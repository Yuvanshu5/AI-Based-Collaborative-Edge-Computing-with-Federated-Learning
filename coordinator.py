"""Task scheduler, workload generator, baselines, and experiment runner."""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, time, threading
from functools import lru_cache
from pathlib import Path
from edge_network.common import DATA, FEATURES, csv_append, send_request, task_features, federated_task_features
from edge_network.tasks import generate_tasks

NODES=[("node1","127.0.0.1",9100),("node2","127.0.0.1",9101),("node3","127.0.0.1",9102),("node4","127.0.0.1",9103)]
KINDS=["sort","matrix","preprocess","numeric"]
CPU_LOAD_PENALTY_MS_PER_PERCENT=25.0
def central_features(task,metrics):
    return {"size":float(task.get("size",1)),"complexity":float(task.get("complexity",1)),
            "cpu":float(metrics.get("cpu_percent",0)),"ram":float(metrics.get("ram_percent",0)),
            "queue":float(metrics.get("queue",0)),"latency_ms":float(metrics.get("latency_ms",0)),
            "bandwidth_mbps":float(metrics.get("bandwidth_mbps",100)),"kind":task.get("kind","sort")}
def collect():
    rows=[]
    for name,host,port in NODES:
        before=time.perf_counter(); status=send_request(host,port,{"action":"status"}); latency=(time.perf_counter()-before)*500
        status["latency_ms"]=latency; status["bandwidth_mbps"]=100.0
        status["node" ]=name; rows.append((name,host,port,status))
        csv_append(DATA/"metrics.csv", {"timestamp":time.time(),"node":name,**{k:status.get(k,0) for k in ("cpu_percent","ram_percent","memory_mb","queue","running","latency_ms","bandwidth_mbps")}})
    return rows
def train_rf():
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from edge_network.dataset import load_workload, make_manifest
        rows=load_workload()
        if len(rows)<20: raise ValueError("Need at least 20 measured tasks for a train/test split. Collect more with: python coordinator.py collect-dataset")
        mapping={str(i+1):kind for i,kind in enumerate(KINDS)}
        clean=[]
        for row in rows:
            row=dict(row); row["kind"]=row.get("kind") or mapping.get(row.get("complexity"),"sort")
            clean.append(row)
        x=pd.DataFrame([{k:float(r[k]) for k in FEATURES} | {"kind":r["kind"]} for r in clean])
        y=[float(r["execution_ms"]) for r in clean]
        split=make_manifest(clean)
        is_train=[split[r["task_id"]]=="train" for r in clean]
        train_x=x.loc[is_train]; test_x=x.loc[[not value for value in is_train]]
        train_y=[value for value,keep in zip(y,is_train) if keep]
        test_y=[value for value,keep in zip(y,is_train) if not keep]
        numeric=list(FEATURES); prep=ColumnTransformer([("numeric",SimpleImputer(strategy="median"),numeric),
            ("kind",OneHotEncoder(handle_unknown="ignore"),["kind"])])
        model=Pipeline([("features",prep),("forest",RandomForestRegressor(n_estimators=200,random_state=42,min_samples_leaf=2,n_jobs=-1))])
        model.fit(train_x,train_y); pred=model.predict(test_x)
        import joblib; joblib.dump(model,DATA/"central_model.joblib")
        report={"samples":len(rows),"train_samples":len(train_y),"test_samples":len(test_y),
                "test_sizes":sorted({int(float(r["size"])) for r in clean if split[r["task_id"]]=="test"}),
                "test_mae_ms":mean_absolute_error(test_y,pred),"test_r2":r2_score(test_y,pred),
                "split":f"held-out task sizes {sorted({int(float(r['size'])) for r in clean if split[r['task_id']]=='test'})}",
                "target":"measured execution_ms"}
        (DATA/"central_model.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        csv_append(DATA/"model_comparison.csv",{"model":"central_ml","timestamp":time.time(),"test_mae_ms":report["test_mae_ms"],"test_samples":len(test_y),"test_r2":report["test_r2"]})
        print(json.dumps(report,indent=2))
    except ImportError as e: print(f"ML dependencies missing: {e}")

@lru_cache(maxsize=1)
def load_central_predictor():
    import joblib
    return joblib.load(DATA/"central_model.joblib")

@lru_cache(maxsize=1)
def load_federated_predictor():
    import torch
    from train_federated import Predictor
    model=Predictor(); ck=torch.load(DATA/"federated_model.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(ck); model.eval()
    return model

def dispatch(task, policy, rr_index=0, verbose=True):
    rows=collect(); chosen=rows[0]
    if policy=="round_robin": chosen=rows[rr_index%len(rows)]
    elif policy=="least_loaded": chosen=min(rows,key=lambda r:r[3]["cpu_percent"]+r[3]["ram_percent"]*.25+r[3]["queue"]*20+r[3]["running"]*30)
    elif policy in ("central_ml","federated_ml"):
        model_path=DATA/("federated_model.pt" if policy=="federated_ml" else "central_model.joblib")
        if model_path.exists():
            if policy=="central_ml":
                import pandas as pd; model=load_central_predictor()
                predicted_exec=[float(model.predict(pd.DataFrame([central_features(task,r[3])]))[0]) for r in rows]
            else:
                from train_federated import infer
                model=load_federated_predictor()
                predicted_exec=[infer(model,federated_task_features(task,r[3])) for r in rows]
            # Estimate total completion from predicted execution plus current network and queue state.
            scores=[]
            for predicted,r in zip(predicted_exec,rows):
                metrics=r[3]; queued=float(metrics["queue"]); running=float(metrics["running"])
                queue_wait=max(0.0,queued+running-1.0)*predicted/2.0
                if float(task.get("size",0)) < 16000:
                    # For tiny tasks, transfer latency dominates execution time.
                    execution_weight=.02; latency_weight=4.0; load_penalty=0.0
                else:
                    execution_weight=1.0
                    latency_weight=2.0
                    # Both learned policies share a runtime-load prior for heavy tasks.
                    load_penalty=CPU_LOAD_PENALTY_MS_PER_PERCENT*max(0.0,float(metrics["cpu_percent"]))
                scores.append(execution_weight*predicted+latency_weight*float(metrics["latency_ms"])+queue_wait+load_penalty)
            chosen=rows[min(range(len(rows)),key=lambda i:scores[i])]
            if chosen[0]!="node1" and scores[0] <= scores[rows.index(chosen)]: chosen=rows[0]
        else: raise FileNotFoundError(f"Missing {policy} model: {model_path}. Train the model before selecting this policy.")
    name,host,port,metrics=chosen
    start_timestamp=time.time(); started=time.perf_counter(); result=send_request(host,port,{"action":"run","task":task}); total=(time.perf_counter()-started)*1000
    finish_timestamp=time.time(); transfer=max(0,total-result.get("execution_ms",0)); row={"timestamp":finish_timestamp,"start_timestamp":start_timestamp,"finish_timestamp":finish_timestamp,"task_id":task["task_id"],"method":policy,"node":name,
        "kind":task["kind"],"size":task["size"],"execution_ms":result.get("execution_ms",0),"completion_ms":total,
        "transfer_ms":transfer,"ok":result.get("ok",False)}
    row["experiment_id"]=task.get("experiment_id",""); row["run_id"]=task.get("run_id","")
    csv_append(DATA/"results.csv",row)
    # Keep comparison tasks out of training data; otherwise rerunning a benchmark leaks its evaluation rows.
    if not task.get("experiment_id"):
        csv_append(DATA/"workload.csv", {**dict(zip(FEATURES,task_features(task,metrics))),"kind":task["kind"],
            "condition":task.get("condition","normal"),"execution_ms":result.get("execution_ms",0),"completion_ms":total,
            "node":name,"task_id":task["task_id"]})
    if verbose: print(f"{task['task_id']}: {policy} to {name}, execution {row['execution_ms']:.1f} ms, completion {total:.1f} ms")
    return row

def start_node_load(name, host, port, duration_s, memory_mb, cpu_duty=1.0, label="heavy-demo"):
    try:
        response=send_request(host,port,{"action":"start_load","duration_s":duration_s,"memory_mb":memory_mb,"cpu_duty":cpu_duty})
        ok=response.get("ok",False)
    except Exception as error:
        response={"error":str(error)}; ok=False
    started_at=response.get("load_started_at",time.time()); finished_at=response.get("load_finishes_at",started_at+duration_s)
    csv_append(DATA/f"{name}_loads.csv",{"timestamp":started_at,"start_timestamp":started_at,
        "finish_timestamp":finished_at,"task_id":f"load-{label}-{name}-{time.time_ns()}","method":f"load_for_{label}",
        "node":name,"kind":"CPU and memory load","size":memory_mb,"execution_ms":duration_s*1000,
        "completion_ms":duration_s*1000,"transfer_ms":0,"memory_load_mb":memory_mb,
        "cpu_duty":cpu_duty,"ok":ok})
    print(f"{name}: {cpu_duty:.0%} CPU + {memory_mb} MB memory load for {duration_s}s ({label}) {'started' if ok else 'failed'}")

def start_load_profile(duration_s, max_memory_mb, label):
    # Deliberately give the workers different loads so schedulers have a real choice.
    profiles=[(1.0,1.0),(0.8,.75),(.55,.5),(.3,.3)]
    started_at=time.time(); threads=[]
    for (name,host,port),(duty,memory_fraction) in zip(NODES,profiles):
        memory_mb=max(4,int(max_memory_mb*memory_fraction))
        worker=threading.Thread(target=start_node_load,args=(name,host,port,duration_s,memory_mb,duty,label),daemon=True)
        worker.start(); threads.append(worker)
    for worker in threads: worker.join()
    time.sleep(.4)
    return {"threads":threads,"finish_at":started_at+duration_s}

def summarize_heavy_comparison(experiment_id, rows):
    grouped={}
    for row in rows: grouped.setdefault(row["method"],[]).append(row)
    baseline=statistics.mean(float(row["completion_ms"]) for row in grouped.get("least_loaded",[]))
    out=[]
    for policy,items in grouped.items():
        completion=sorted(float(item["completion_ms"]) for item in items)
        execution=[float(item["execution_ms"]) for item in items]
        mean=statistics.mean(completion)
        out.append({"experiment_id":experiment_id,"policy":policy,"tasks":len(items),
            "avg_completion_ms":mean,"median_completion_ms":statistics.median(completion),
            "p95_completion_ms":completion[min(len(completion)-1,math.ceil(.95*len(completion))-1)],
            "avg_execution_ms":statistics.mean(execution),"success_rate":sum(bool(item["ok"]) for item in items)/len(items),
            "speedup_vs_least_loaded":baseline/mean if baseline>0 else ""})
    for row in out: csv_append(DATA/"heavy_comparison.csv",row)
    print(json.dumps(out,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("status"); q=sub.add_parser("run"); q.add_argument("--policy",choices=["local","round_robin","least_loaded","central_ml","federated_ml"],default="least_loaded"); q.add_argument("--count",type=int,default=12)
    q=sub.add_parser("benchmark"); q.add_argument("--count",type=int,default=20); q.add_argument("--seed",type=int,default=7)
    q=sub.add_parser("train-ml")
    q=sub.add_parser("collect-dataset"); q.add_argument("--repeats",type=int,default=3); q.add_argument("--seed",type=int,default=101)
    q=sub.add_parser("collect-heavy-dataset"); q.add_argument("--repeats",type=int,default=4); q.add_argument("--seed",type=int,default=101)
    q=sub.add_parser("demo-heavy"); q.add_argument("--kind",choices=KINDS,default="matrix"); q.add_argument("--size",type=int,default=500000); q.add_argument("--count",type=int,default=2); q.add_argument("--duration",type=int,default=12); q.add_argument("--memory-mb",type=int,default=96); q.add_argument("--policy",choices=["compare","local","round_robin","least_loaded","central_ml","federated_ml"],default="compare"); q.add_argument("--seed",type=int,default=101)
    a=p.parse_args()
    if a.cmd=="status":
        for r in collect(): print(json.dumps(r[3],indent=2))
    elif a.cmd=="train-ml": train_rf()
    elif a.cmd=="collect-dataset":
        task_no=0; collection_id=time.time_ns()
        for repeat in range(a.repeats):
            for kind_idx,kind in enumerate(KINDS):
                for size in [500,1000,2000,4000,8000,16000]:
                    condition=["idle","moderate","heavy"][repeat%3]
                    task={"task_id":f"dataset-{collection_id}-{repeat}-{kind}-{size}","kind":kind,"size":size,
                          "complexity":kind_idx+1,"seed":a.seed+task_no,"condition":condition}
                    stress=None
                    if condition!="idle":
                        node=NODES[task_no%len(NODES)]; duration=1.2 if condition=="moderate" else 2.0
                        stress=threading.Thread(target=send_request,args=(node[1],node[2],{"action":"stress","duration_s":duration}),daemon=True)
                        stress.start()
                    dispatch(task,"round_robin",task_no,verbose=False); task_no+=1
                    if stress: stress.join()
        print(f"Collected {task_no} measured tasks across idle/moderate/heavy conditions; dataset: {DATA/'workload.csv'}")
    elif a.cmd=="collect-heavy-dataset":
        task_no=0; collection_id=time.time_ns(); sizes=[16000,64000,250000,500000]
        for repeat in range(a.repeats):
            condition=["idle","moderate","heavy"][repeat%3]
            for kind_idx,kind in enumerate(KINDS):
                for size in sizes:
                    node_index=(task_no+repeat)%len(NODES); node=NODES[node_index]; stress=None
                    if condition!="idle":
                        stress_duration=3.0 if condition=="moderate" else 5.0
                        stress_memory=16 if condition=="moderate" else 48
                        stress_duty=.45 if condition=="moderate" else .9
                        stress=threading.Thread(target=send_request,args=(node[1],node[2],
                            {"action":"stress","duration_s":stress_duration,"memory_mb":stress_memory,"cpu_duty":stress_duty}),
                            kwargs={"timeout":stress_duration+15},daemon=True)
                        stress.start(); time.sleep(.35)
                    task={"task_id":f"heavy-data-{collection_id}-{repeat}-{kind}-{size}","kind":kind,"size":size,
                          "complexity":kind_idx+1,"seed":a.seed+task_no,"condition":f"heavy-training-{condition}"}
                    dispatch(task,"round_robin",task_no+repeat,verbose=False); task_no+=1
                    if stress: stress.join()
        print(f"Collected {task_no} large-task measurements, including held-out size 500000. Retrain both schedulers before comparing.")
    elif a.cmd=="demo-heavy":
        policies=["local","round_robin","least_loaded","central_ml","federated_ml"] if a.policy=="compare" else [a.policy]
        missing=[]
        if "central_ml" in policies and not (DATA/"central_model.joblib").exists(): missing.append("central model (run python coordinator.py train-ml)")
        if "federated_ml" in policies and not (DATA/"federated_model.pt").exists(): missing.append("federated model (train it with the Flower server and four clients)")
        if missing: raise SystemExit("Cannot run comparison: missing " + "; ".join(missing))
        experiment_id=f"heavy-{time.time_ns()}"; policies_order=list(policies); random.Random(a.seed).shuffle(policies_order)
        print(f"Experiment {experiment_id}: same {a.kind}/{a.size} task set for {', '.join(policies_order)}")
        print("Each policy gets a fresh, varied four-node load profile: node1 is busiest; node4 is least busy.")
        rows=[]
        task_defs=[{"kind":a.kind,"size":a.size,"complexity":KINDS.index(a.kind)+1,
                    "seed":a.seed+i,"condition":"heavy-comparison"} for i in range(a.count)]
        load_duration=max(a.duration,a.count*7+2 if a.kind=="matrix" else a.count*2+2)
        for policy in policies_order:
            print(f"\nRunning policy: {policy}")
            load_profile=start_load_profile(load_duration,a.memory_mb,policy)
            for i,base_task in enumerate(task_defs):
                task={**base_task,"task_id":f"{experiment_id}-{policy}-{i}","experiment_id":experiment_id}
                rows.append(dispatch(task,policy,i))
            time.sleep(max(0,load_profile["finish_at"]-time.time()))
        summarize_heavy_comparison(experiment_id,rows)
    elif a.cmd=="run":
        run_id=f"normal-{time.time_ns()}"
        for i,task in enumerate(generate_tasks(a.count)):
            task["run_id"]=run_id; dispatch(task,a.policy,i)
    else:
        # Warm model files before timing so one-time loading is not charged to the first task.
        load_central_predictor(); load_federated_predictor()
        benchmark_started=time.time()
        run_id=f"normal-{time.time_ns()}"
        tasks=list(generate_tasks(a.count,a.seed))
        policies=["local","round_robin","least_loaded","central_ml","federated_ml"]
        for i,base_task in enumerate(tasks):
            # Rotate scheduler order for each matching task; avoid always measuring FL last.
            offset=i%len(policies); order=policies[offset:]+policies[:offset]
            for policy in order:
                task={**base_task,"run_id":run_id}; dispatch(task,policy,i)
        summarize(benchmark_started)
def summarize(since=None):
    rows=list(csv.DictReader((DATA/"results.csv").open(encoding="utf-8")))
    if since is not None: rows=[r for r in rows if float(r["timestamp"])>=since]
    methods=sorted(set(r["method"] for r in rows)); out=[]
    for method in methods:
        values=[float(r["completion_ms"]) for r in rows if r["method"]==method]; ordered=sorted(values)
        out.append({"method":method,"tasks":len(values),"avg_completion_ms":statistics.mean(values),"p95_ms":ordered[min(len(ordered)-1,math.ceil(.95*len(ordered))-1)],"throughput_per_s":1000/statistics.mean(values),"deadline_success_rate":sum(v<=5000 for v in values)/len(values)})
    with (DATA/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
