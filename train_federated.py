"""Train a small execution-time regressor with Flower FedAvg over four clients.

Start the server, then one client per node (see README). Raw rows remain on clients.
"""
from __future__ import annotations
import argparse, csv, json, time
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import flwr as fl
from edge_network.common import DATA, FEATURES, FL_FEATURES, federated_row_features
from edge_network.dataset import load_workload, make_manifest, read_manifest

class Predictor(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(len(FL_FEATURES),32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x)

def normalize_x(x): return np.log1p(np.maximum(x,0)).astype(np.float32)
def normalize_y(y): return np.log1p(np.maximum(y,0)).astype(np.float32)
def infer(model, features):
    model.eval()
    with torch.no_grad(): return max(0.0,float(torch.expm1(model(torch.tensor(normalize_x(np.asarray([features])))).squeeze()).item()))
def load_local(node):
    path=DATA/"workload.csv"
    if not path.exists(): return np.empty((0,len(FL_FEATURES)),dtype=np.float32),np.empty((0,),dtype=np.float32),[]
    rows=load_workload(path); split=read_manifest()
    if not split or any(r["task_id"] not in split for r in rows): split=make_manifest(rows)
    x=[]; y=[]; labels=[]
    for row in rows:
        if row.get("node")==node:
            try:
                x.append(federated_row_features(row))
                y.append(float(row["execution_ms"]))
                labels.append(split[row["task_id"]])
            except (TypeError,ValueError): continue
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),labels

class Client(fl.client.NumPyClient):
    def __init__(self,node):
        self.node=node; self.x,self.y,labels=load_local(node); self.model=Predictor()
        train_idx=np.asarray([i for i,label in enumerate(labels) if label=="train"],dtype=int)
        test_idx=np.asarray([i for i,label in enumerate(labels) if label=="test"],dtype=int)
        self.x_train,self.y_train=self.x[train_idx],self.y[train_idx]
        self.x_test,self.y_test=self.x[test_idx],self.y[test_idx]
    def get_parameters(self,config): return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]
    def set_parameters(self,params):
        state=self.model.state_dict()
        self.model.load_state_dict({k:torch.tensor(v) for k,v in zip(state,params)})
    def fit(self,params,config):
        self.set_parameters(params)
        if len(self.y_train):
            x=torch.tensor(normalize_x(self.x_train)); y=torch.tensor(normalize_y(self.y_train)).reshape(-1,1)
            optim=torch.optim.Adam(self.model.parameters(),lr=.01); self.model.train()
            for _ in range(int(config.get("local_epochs",5))):
                optim.zero_grad(); loss=nn.functional.mse_loss(self.model(x),y); loss.backward(); optim.step()
        return self.get_parameters({}),max(len(self.y_train),1),{"node":self.node,"local_train_rows":len(self.y_train),"local_test_rows":len(self.y_test)}
    def evaluate(self,params,config):
        self.set_parameters(params)
        if not len(self.y_test): return 0.0,0,{"mae_ms":0.0}
        with torch.no_grad(): pred=torch.expm1(self.model(torch.tensor(normalize_x(self.x_test))).squeeze()).numpy()
        return float(np.mean((np.log1p(np.maximum(pred,0))-normalize_y(self.y_test))**2)),len(self.y_test),{"mae_ms":float(np.mean(np.abs(pred-self.y_test)))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["server","client"]); p.add_argument("--node",choices=[f"node{i}" for i in range(1,5)]); p.add_argument("--address",default="127.0.0.1:8080"); p.add_argument("--rounds",type=int,default=10); p.add_argument("--local-epochs",type=int,default=10); a=p.parse_args()
    if a.mode=="server" and (DATA/"workload.csv").exists():
        rows=load_workload(); split=read_manifest()
        if not split or any(r["task_id"] not in split for r in rows): make_manifest(rows)
    import flwr as fl
    if a.mode=="client":
        if not a.node: p.error("client mode requires --node")
        fl.client.start_client(server_address=a.address,client=Client(a.node).to_client())
        return
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg
    import torch
    initial=[v.detach().numpy() for v in Predictor().state_dict().values()]
    class SaveFedAvg(FedAvg):
        def configure_fit(self,server_round,parameters,client_manager):
            self.round_started=time.perf_counter()
            return super().configure_fit(server_round,parameters,client_manager)
        def aggregate_fit(self,server_round,results,failures):
            result=super().aggregate_fit(server_round,results,failures)
            if result and result[0]:
                arrays=fl.common.parameters_to_ndarrays(result[0]); model=Predictor()
                model.load_state_dict({k:torch.tensor(v) for k,v in zip(model.state_dict(),arrays)})
                DATA.mkdir(exist_ok=True); torch.save(model.state_dict(),DATA/"federated_model.pt")
                model_bytes=sum(a.nbytes for a in arrays)
                self.pending_round_metrics={"round":server_round,"clients":len(results),"timestamp":time.time(),
                         "duration_s":time.perf_counter()-getattr(self,"round_started",time.perf_counter()),
                         "estimated_communication_bytes":model_bytes*max(1,len(results))*2}
                print(f"FedAvg round {server_round}: aggregated {len(results)} clients",flush=True)
            return result
        def aggregate_evaluate(self,server_round,results,failures):
            outcome=super().aggregate_evaluate(server_round,results,failures)
            evaluated=[(float(client_result.metrics["mae_ms"]),int(client_result.num_examples))
                       for _,client_result in results if "mae_ms" in client_result.metrics and client_result.num_examples>0]
            from edge_network.common import csv_append
            round_metrics=getattr(self,"pending_round_metrics",None)
            if round_metrics:
                csv_append(DATA/"fl_rounds.csv",round_metrics)
            if evaluated:
                count=sum(n for _,n in evaluated); mae=sum(value*n for value,n in evaluated)/count
                csv_append(DATA/"fl_eval.csv",{"round":server_round,"timestamp":time.time(),"client_mae_ms":mae,"test_samples":count})
                csv_append(DATA/"model_comparison.csv",{"model":"federated_ml","timestamp":time.time(),"test_mae_ms":mae,"test_samples":count,"test_r2":""})
            return outcome
    strategy=SaveFedAvg(fraction_fit=1.0,min_fit_clients=4,min_available_clients=4,initial_parameters=ndarrays_to_parameters(initial),on_fit_config_fn=lambda r:{"local_epochs":a.local_epochs})
    start_server(server_address=a.address,config=ServerConfig(num_rounds=a.rounds),strategy=strategy)
if __name__=="__main__": main()
