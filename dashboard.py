"""Task-focused dashboard: streamlit run dashboard.py"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st
from edge_network.common import send_request

DATA = Path(__file__).resolve().parent / "data"
st.set_page_config(page_title="Edge Network Lab", layout="wide")
st.title("Four Node Edge Computing Network")
st.caption("Compare normal workloads, heavy-load scheduler results, and model training metrics.")


def read(name):
    path = DATA / name
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


NODES = [(f"node{i}", "127.0.0.1", 9100 + i - 1) for i in range(1, 5)]

with st.expander("Node controls · load and availability", expanded=True):
    st.caption("Set a persistent simulated CPU and memory load on each worker, or mark it unavailable to schedulers. This simulates a node outage; the worker stays reachable so you can re-enable it here.")
    current = {}
    for name, host, port in NODES:
        try:
            current[name] = send_request(host, port, {"action": "status"}, timeout=1.0)
        except Exception as error:
            current[name] = {"ok": False, "enabled": False, "load_percent": 0, "error": str(error)}
    with st.form("node_controls"):
        cols = st.columns(4)
        enabled_values, load_values = {}, {}
        for (name, _, _), col in zip(NODES, cols):
            status = current[name]
            online = bool(status.get("ok"))
            with col:
                st.markdown(f"**{name}** · {'connected' if online else 'offline'}")
                enabled_values[name] = st.checkbox("Available to scheduler", value=bool(status.get("enabled", False)) if online else False, key=f"enabled_{name}", disabled=not online)
                load_values[name] = st.slider("Simulated load (%)", 0, 100, int(status.get("load_percent", 0)) if online else 0, 5, key=f"load_{name}", disabled=not online)
                st.caption(f"CPU {float(status.get('cpu_percent', 0)):.0f}% · RAM {float(status.get('ram_percent', 0)):.0f}% · queue {int(status.get('queue', 0))}")
        submitted = st.form_submit_button("Apply node settings")
    if submitted:
        messages = []
        for name, host, port in NODES:
            if not current[name].get("ok"):
                messages.append(f"{name}: worker is not connected")
                continue
            try:
                send_request(host, port, {"action": "set_enabled", "enabled": enabled_values[name]}, timeout=2.0)
                percent = load_values[name] if enabled_values[name] else 0
                send_request(host, port, {"action": "set_load", "percent": percent, "memory_mb": round(percent * .64)}, timeout=2.0)
                messages.append(f"{name}: {'enabled' if enabled_values[name] else 'disabled'}, load {percent}%")
            except Exception as error:
                messages.append(f"{name}: could not apply settings ({error})")
        for message in messages:
            st.write(message)
        st.rerun()


if st.button("Refresh dashboard"):
    st.rerun()

results = read("results.csv")
if not results.empty:
    for column in ("experiment_id", "run_id"):
        if column not in results:
            results[column] = ""
        results[column] = results[column].fillna("").astype(str)

normal_tab, heavy_tab, model_tab = st.tabs(["Normal tasks", "Heavy-load comparison", "ML / FL model"])

with normal_tab:
    st.subheader("Normal workload comparison")
    normal = results[results["experiment_id"] == ""].copy() if not results.empty else pd.DataFrame()
    if not normal.empty:
        task_ids = normal["task_id"].fillna("").astype(str)
        normal = normal[~task_ids.str.startswith(("heavy-", "dataset-"))]
        tracked_runs = normal[normal["run_id"].str.startswith("normal-")]
        if not tracked_runs.empty:
            latest_run = tracked_runs["run_id"].iloc[-1]
            normal = tracked_runs[tracked_runs["run_id"] == latest_run]
    if normal.empty:
        st.info("No normal task run is recorded yet. Run `python coordinator.py benchmark --count 4 --seed 7`.")
    else:
        st.caption("Results are limited to the latest normal run so they do not mix with older experiments.")
        averages = normal.groupby("method")["completion_ms"].mean().sort_values()
        st.bar_chart(averages)
        st.markdown("**Average completion by scheduler**")
        average_table = normal.groupby("method").agg(
            attempts=("completion_ms", "size"),
            avg_completion_ms=("completion_ms", "mean"),
            median_completion_ms=("completion_ms", "median"),
            p95_completion_ms=("completion_ms", lambda values: values.quantile(.95)),
            avg_execution_ms=("execution_ms", "mean"),
        ).sort_values("avg_completion_ms")
        successes = normal[normal["ok"].astype(str).str.lower().isin(["true", "1"])] if "ok" in normal else normal.dropna(subset=["completion_ms"])
        rates = successes.groupby("method").size().div(normal.groupby("method").size()).rename("success_rate")
        average_table = average_table.join(rates)
        st.dataframe(average_table, use_container_width=True)
        st.markdown("**Individual normal task runs**")
        columns = [c for c in ["timestamp", "task_id", "method", "node", "kind", "size",
                                "execution_ms", "completion_ms", "ok"] if c in normal]
        st.dataframe(normal.sort_values("timestamp", ascending=False)[columns], use_container_width=True)

with heavy_tab:
    st.subheader("Heavy-load scheduler comparison")
    comparison = read("heavy_comparison.csv")
    if comparison.empty:
        st.info("No heavy comparison is recorded yet. Run `python coordinator.py demo-heavy --kind matrix --size 500000 --policy compare --count 2`.")
    else:
        latest_experiment = comparison["experiment_id"].dropna().iloc[-1]
        latest = comparison[comparison["experiment_id"] == latest_experiment].copy()
        latest = latest.sort_values("avg_completion_ms")
        st.caption(f"Experiment {latest_experiment}. Lower completion time is faster; speedup above 1.0 beats Least Loaded in this run.")
        st.bar_chart(latest.set_index("policy")["avg_completion_ms"])
        summary_cols = [c for c in ["policy", "tasks", "avg_completion_ms", "median_completion_ms",
                                    "p95_completion_ms", "avg_execution_ms", "success_rate",
                                    "speedup_vs_least_loaded"] if c in latest]
        st.dataframe(latest[summary_cols], use_container_width=True)
        detail = results[results["experiment_id"] == str(latest_experiment)].copy() if not results.empty else pd.DataFrame()
        if not detail.empty:
            st.markdown("**Individual heavy task runs**")
            detail_cols = [c for c in ["task_id", "method", "node", "kind", "size",
                                       "execution_ms", "completion_ms", "ok"] if c in detail]
            st.dataframe(detail.sort_values(["method", "timestamp"])[detail_cols], use_container_width=True)

with model_tab:
    st.subheader("Training data and prediction quality")
    with st.expander("How Central ML and Federated ML choose a node", expanded=True):
        st.markdown("Both methods estimate task execution time for each available node, combine that estimate with load and network state, then choose the node with the lowest score. The models differ in how they learn; the scheduler score is shared.")
        central_col, federated_col = st.columns(2)
        with central_col:
            st.markdown("**Central ML · Random Forest**")
            st.markdown("One model is trained from the combined workload records. For a task and node, its input includes size, complexity, CPU, RAM, queue, latency, bandwidth, and task kind.")
            st.latex(r"\hat{t}_i = f_{RF}(\text{task features},\,\text{node}_i\text{ state})")
            st.markdown("The output `t̂ᵢ` is the predicted execution time in milliseconds.")
        with federated_col:
            st.markdown("**Federated ML · local neural networks + FedAvg**")
            st.markdown("Each of the four clients trains on its own node’s training rows. It transforms feature and target values with `log(1 + x)`, then trains a small neural network using mean squared error:")
            st.latex(r"L_k = \frac{1}{n_k}\sum_j (\hat{y}_{k,j}-\log(1+y_{k,j}))^2")
            st.markdown("The server averages client model weights in proportion to each client’s training-row count. Each client trains only on rows assigned to its node; the aggregation shares model weights rather than combining training batches.")
            st.latex(r"w_{r+1}=\sum_{k=1}^{4}\frac{n_k}{\sum_j n_j}w^{(k)}_{r+1}")
            st.markdown("At prediction time, the network output is converted back to milliseconds with `exp(output) − 1`.")
        st.markdown("**Shared node score**")
        st.markdown("For small tasks (`size < 16,000`), the scheduler emphasizes network latency. For larger tasks, predicted execution and CPU load matter more:")
        st.latex(r"S_i = \begin{cases}0.02\hat{t}_i + 4\ell_i + q_i, & \text{size}<16{,}000\\\\\hat{t}_i + 2\ell_i + q_i + 25\,CPU_i, & \text{otherwise}\end{cases}")
        st.latex(r"q_i=\max(0,\,queue_i+running_i-1)\frac{\hat{t}_i}{2};\qquad node^*=\arg\min_{i\in enabled} S_i")
        st.caption("Here ℓ is the measured half-round-trip latency estimate in milliseconds; CPU is a percentage. This local demo stores node-tagged rows in one workload.csv, then each simulated client filters its own rows; it demonstrates federated aggregation but does not provide separate physical data stores. The dashboard’s average_completion_ms is the measured worker request-to-response time, not this predicted score or model training time. Run-to-run delays can vary, so no learned scheduler can truthfully be guaranteed to have the lowest average every time. Use more tasks and repeat runs to compare robustly.")
    model_path = DATA / "central_model.json"
    if model_path.exists():
        st.markdown("**Central ML test report**")
        st.json(json.loads(model_path.read_text(encoding="utf-8")))
    comparison = read("model_comparison.csv")
    if not comparison.empty:
        central = comparison[comparison["model"] == "central_ml"]
        federated = comparison[comparison["model"] == "federated_ml"]
        latest_rows = []
        if not central.empty:
            latest_rows.append(central.iloc[-1])
        if not federated.empty:
            latest_rows.append(federated.iloc[-1])
        if latest_rows:
            view = pd.DataFrame(latest_rows)[["model", "test_mae_ms", "test_samples"]].set_index("model")
            st.markdown("**Latest held-out prediction error (MAE, ms)**")
            st.bar_chart(view["test_mae_ms"])
            st.dataframe(view, use_container_width=True)
            st.caption("Lower MAE means execution-time predictions were closer to measured times on held-out data.")
    fl_rounds = read("fl_rounds.csv")
    if not fl_rounds.empty:
        st.markdown("**Recent Federated Averaging rounds**")
        st.dataframe(fl_rounds.tail(5), use_container_width=True)
    fl_eval = read("fl_eval.csv")
    if not fl_eval.empty:
        st.markdown("**Federated model MAE by round**")
        st.line_chart(fl_eval.tail(20).set_index("round")["client_mae_ms"])
    dataset = read("workload.csv")
    if not dataset.empty:
        st.markdown("**Measured training samples by task and load condition**")
        if "condition" in dataset:
            st.bar_chart(dataset.groupby(["kind", "condition"]).size().unstack(fill_value=0))
        else:
            st.bar_chart(dataset.groupby("kind").size())
