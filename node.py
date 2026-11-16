"""TCP worker process. Run once per node: python node.py --node 1"""
from __future__ import annotations
import argparse, json, socketserver, threading, time
from pathlib import Path
import psutil
from edge_network.common import DATA, csv_append
from edge_network.tasks import execute

class State:
    def __init__(self, name):
        self.name=name; self.waiting=0; self.running=0; self.lock=threading.Lock(); self.slots=threading.BoundedSemaphore(2)
        self.process=psutil.Process(); self.metric_lock=threading.Lock(); self.process.cpu_percent(interval=None)
        self.last_sample={"timestamp":time.time(),"node":name,"cpu_percent":0.0,"ram_percent":self.process.memory_percent(),
                          "memory_mb":self.process.memory_info().rss/(1024*1024),"queue":0,"running":0,"source":"process_sampler"}
    def sample(self):
        with self.metric_lock:
            cpu=self.process.cpu_percent(interval=None)
            ram=self.process.memory_percent(); memory_mb=self.process.memory_info().rss/(1024*1024)
        with self.lock: waiting,running=self.waiting,self.running
        sample={"timestamp":time.time(),"node":self.name,"cpu_percent":cpu,"ram_percent":ram,
                "memory_mb":memory_mb,"queue":waiting,"running":running,"source":"process_sampler"}
        with self.metric_lock: self.last_sample=sample
        return sample
    def latest(self):
        with self.metric_lock: sample=dict(self.last_sample)
        with self.lock: sample["queue"],sample["running"]=self.waiting,self.running
        return sample
state=None

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        task_log=None
        try:
            request=json.loads(self.rfile.readline().decode())
            if request.get("action")=="status":
                with state.lock: waiting,running=state.waiting,state.running
                response=self.status(waiting,running)
            elif request.get("action")=="run":
                with state.lock: state.waiting+=1
                state.slots.acquire()
                with state.lock: state.waiting-=1; state.running+=1
                started=time.perf_counter()
                try:
                    result,elapsed=execute(request["task"])
                    response={"ok":True,"node":state.name,"result":result,"execution_ms":elapsed,
                              "task":request["task"],"queue":state.waiting}
                    task_log={"timestamp":time.time(),"task_id":request["task"].get("task_id"),
                        "kind":request["task"]["kind"],"size":request["task"]["size"],"execution_ms":elapsed}
                except Exception as e: response={"ok":False,"error":str(e),"node":state.name}
                finally:
                    with state.lock: state.running-=1
                    state.slots.release()
            elif request.get("action")=="stress":
                duration=max(0.1,min(120.0,float(request.get("duration_s",1.0))))
                memory_mb=max(0,min(256,int(request.get("memory_mb",0))))
                cpu_duty=max(0.0,min(1.0,float(request.get("cpu_duty",1.0))))
                with state.lock: state.waiting+=1
                state.slots.acquire()
                with state.lock: state.waiting-=1; state.running+=1
                deadline=time.perf_counter()+duration; value=0.0
                try:
                    # Allocate and touch each page so the OS commits memory to this worker.
                    memory=bytearray(memory_mb*1024*1024)
                    for offset in range(0,len(memory),4096): memory[offset]=1
                    cycle_s=0.1
                    while time.perf_counter()<deadline:
                        cycle_end=min(deadline,time.perf_counter()+cycle_s)
                        busy_until=min(cycle_end,time.perf_counter()+cycle_s*cpu_duty)
                        while time.perf_counter()<busy_until:
                            for i in range(5000): value+=(i%97)*0.000001
                        idle=cycle_end-time.perf_counter()
                        if idle>0: time.sleep(idle)
                    response={"ok":True,"node":state.name,"stress_s":duration,"memory_mb":memory_mb,"cpu_duty":cpu_duty,"checksum":value}
                finally:
                    if "memory" in locals(): del memory
                    with state.lock: state.running-=1
                    state.slots.release()
            elif request.get("action")=="start_load":
                duration=max(0.1,min(120.0,float(request.get("duration_s",1.0))))
                memory_mb=max(0,min(256,int(request.get("memory_mb",0))))
                cpu_duty=max(0.0,min(1.0,float(request.get("cpu_duty",1.0))))
                if not state.slots.acquire(blocking=False):
                    response={"ok":False,"error":"No worker slot is available for background load","node":state.name}
                else:
                    try:
                        memory=bytearray(memory_mb*1024*1024)
                        for offset in range(0,len(memory),4096): memory[offset]=1
                        memory_holder=[memory]
                        started_at=time.time()
                        with state.lock: state.running+=1
                        def background_load():
                            deadline=time.perf_counter()+duration; value=0.0; cycle_s=.1
                            try:
                                while time.perf_counter()<deadline:
                                    cycle_end=min(deadline,time.perf_counter()+cycle_s)
                                    busy_until=min(cycle_end,time.perf_counter()+cycle_s*cpu_duty)
                                    while time.perf_counter()<busy_until:
                                        for i in range(5000): value+=(i%97)*.000001
                                    idle=cycle_end-time.perf_counter()
                                    if idle>0: time.sleep(idle)
                            finally:
                                memory_holder.clear()
                                with state.lock: state.running-=1
                                state.slots.release()
                        thread=threading.Thread(target=background_load,daemon=True); thread.start()
                        response={"ok":True,"node":state.name,"load_started_at":started_at,
                                  "load_finishes_at":started_at+duration,"duration_s":duration,
                                  "memory_mb":memory_mb,"cpu_duty":cpu_duty}
                    except Exception:
                        state.slots.release()
                        raise
            elif request.get("action")=="ping": response={"ok":True,"node":state.name,"time":time.time()}
            else: response={"ok":False,"error":"Unknown action"}
            self.wfile.write((json.dumps(response)+"\n").encode())
            # Reply first: local OneDrive/antivirus file latency should not inflate task completion.
            if task_log is not None:
                try: csv_append(DATA/f"{state.name}_tasks.csv",task_log)
                except Exception as e: print(f"task log error: {e}",flush=True)
        except Exception as e:
            self.wfile.write((json.dumps({"ok":False,"error":str(e)})+"\n").encode())
    @staticmethod
    def status(waiting,running):
        sample=state.latest()
        return {"ok":True,"node":state.name,"cpu_percent":sample["cpu_percent"],"ram_percent":sample["ram_percent"],
                "memory_mb":sample["memory_mb"],"queue":waiting,"running":running,"timestamp":time.time()}

def monitor():
    while True:
        try: csv_append(DATA/f"{state.name}_metrics.csv",state.sample())
        except Exception as e: print(f"metrics sampler error: {e}",flush=True)
        time.sleep(.25)

class Server(socketserver.ThreadingTCPServer): allow_reuse_address=True; daemon_threads=True

def main():
    global state
    p=argparse.ArgumentParser(); p.add_argument("--node",type=int,choices=range(1,5),required=True)
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--base-port",type=int,default=9100)
    a=p.parse_args(); state=State(f"node{a.node}"); port=a.base_port+a.node-1
    threading.Thread(target=monitor,daemon=True).start()
    print(f"{state.name} listening on {a.host}:{port}",flush=True)
    with Server((a.host,port),Handler) as server: server.serve_forever()
if __name__=="__main__": main()
