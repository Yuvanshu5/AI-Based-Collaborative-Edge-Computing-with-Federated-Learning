# Four Node Edge Computing and Task Offloading

A runnable local demonstration of a four node TCP edge network. It covers task execution, resource/network telemetry, Local/Round Robin/Least Loaded baselines, a centralized Random Forest scheduler, Flower FedAvg training with a small PyTorch predictor, and a Streamlit dashboard.

## Requirements

Python 3.10 or newer. From this folder, create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Start the four worker processes

Open four terminals in this folder and run one command per terminal:

```powershell
python node.py --node 1
python node.py --node 2
python node.py --node 3
python node.py --node 4
```

The default ports are 9100–9103 on `127.0.0.1`. To use multiple computers, start each node with `--host 0.0.0.0`, update the node addresses at the top of `coordinator.py`, and ensure the worker ports are reachable through the network firewall.

## Run tasks and baselines

```powershell
python coordinator.py status
python coordinator.py run --policy local --count 12
python coordinator.py run --policy round_robin --count 12
python coordinator.py run --policy least_loaded --count 12
```

Each task uses an allow-listed workload (`sort`, `matrix`, `preprocess`, or `numeric`). The coordinator records the selected node, execution time, completion time and measured transfer/round-trip overhead. `local` means execution on node1, the coordinator's default source node.

To demonstrate longer workloads and visible resource changes, use the heavier task command while the four nodes are running:

```powershell
python coordinator.py demo-heavy --kind matrix --size 500000 --count 4
python coordinator.py demo-heavy --kind preprocess --size 500000 --count 4
```

Both demos apply CPU work and committed memory allocations to all four node workers while tasks run. `matrix` adds a long CPU-bound task; `preprocess` allocates and transforms a large data set in the worker. In a comparison, workers receive different, repeatable CPU and memory load levels so schedulers can choose among them. Nodes sample CPU, memory, queue, and running-task count four times per second into `data/node*_metrics.csv`. The dashboard refreshes when you press **Refresh dashboard** and shows both background-load and task intervals.

For a fair comparison that includes large workloads, first collect heavy-task training examples and retrain both models. Then run the same task set under every scheduler:

```powershell
python coordinator.py collect-heavy-dataset --repeats 4 --seed 101
python coordinator.py train-ml
```

Retrain the federated model with the Flower server and four clients below, then run:

```powershell
python coordinator.py demo-heavy --kind matrix --size 500000 --policy compare --count 2
```

`compare` runs Local, Round Robin, Least Loaded, Central ML, and Federated ML under fresh copies of the same four-node load profile and the same task definitions. Results are saved in `data/heavy_comparison.csv` and shown in the dashboard. A speedup above 1 means that policy had lower average completion time than Least Loaded in that run. The benchmark reports measured results; it does not guarantee either ML scheduler will win. Repeat with more tasks and seeds before drawing a conclusion.

## Centralized ML scheduler

Collect a balanced measured dataset across task types and six sizes (500 to 16,000) using the worker nodes. Repeats rotate through idle, moderate, and heavy CPU conditions. Then train and run the model:

```powershell
python coordinator.py collect-dataset --repeats 5 --seed 101
python coordinator.py train-ml
python coordinator.py run --policy central_ml --count 12
```

The dataset collection command runs 120 measured tasks, balancing four task types and six input sizes over the nodes and varying CPU load. `collect-heavy-dataset` adds larger sizes and idle/moderate/heavy worker conditions. Both models predict measured execution time from task type, size, and node resources. The scheduler estimates total completion by adding the current round-trip estimate and queue wait. Training uses a shared, reproducible size-grouped split: sizes 2000 and 8000 are held out, and size 500000 is also held out when heavy data is present. The exact row assignments are saved in `data/split_manifest.csv`. Test MAE and R² are saved to `data/central_model.json` and `data/model_comparison.csv`. The FL clients use the same manifest and target, so both models are evaluated on the same held-out task rows.

## Federated model and scheduler

Collect a reasonably sized measured sample before training. Each Flower client filters to its node's rows and uses the shared size-grouped split (sizes 2000 and 8000 held out). It trains on training-size rows and evaluates the global model on held-out-size rows; the server receives model parameters and aggregates them with FedAvg.

1. Terminal A: `python train_federated.py server --rounds 10 --local-epochs 10`
2. After the server is ready, terminals B–E:

```powershell
python train_federated.py client --node node1
python train_federated.py client --node node2
python train_federated.py client --node node3
python train_federated.py client --node node4
```

The four clients use the shared `data/workload.csv` file in this single-computer demonstration but filter it to their own node. For a real distributed demonstration, place the project/data on each machine and configure separate reachable client/server addresses. When a global model exists, run `python coordinator.py run --policy federated_ml --count 12`.

## Dashboard and experiments

```powershell
streamlit run dashboard.py
python coordinator.py benchmark --count 20 --seed 7
```

The dashboard reads the CSV logs under `data/` and separates the latest normal run, the latest heavy-load scheduler comparison, and model-training details into tabs. The benchmark replays the same deterministic task definitions for each approach and writes per-task records to `results.csv` and summary metrics to `summary.csv`. The deadline success rate currently uses a 5 second example deadline; edit the value in `coordinator.py` to match your experiment. For credible comparisons, use more tasks, repeat with several seeds, and keep node/network conditions comparable.

## Outputs

- `data/metrics.csv`: CPU, RAM, queue/running counts, measured round-trip latency, and the configured bandwidth estimate.
- `data/workload.csv`: measured feature/target rows used by training.
- `data/results.csv`, `data/summary.csv`: per-task decisions and method comparisons.
- `data/heavy_comparison.csv`: same-task heavy scheduler comparison, including mean/p95 completion and speedup against Least Loaded.
- `data/fl_rounds.csv`: completed Flower aggregation rounds.
- `data/fl_eval.csv`: per-round federated client prediction MAE.
- `data/split_manifest.csv`: shared held-out-size assignment used by both models.
- `data/model_comparison.csv`: central and federated MAE records on that same held-out set.
- `data/central_model.joblib`, `data/federated_model.pt`: trained predictors.

Bandwidth is currently a documented 100 Mbps estimate; transfer timing is measured from each request's round trip after subtracting worker execution time. All nodes use TCP and expose only status/ping/run actions. This is a research/demo scaffold, not an authenticated production service.
