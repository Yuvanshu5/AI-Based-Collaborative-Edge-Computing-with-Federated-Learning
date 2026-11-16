"""Task-focused dashboard: streamlit run dashboard.py"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

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
            tasks=("completion_ms", "size"),
            avg_completion_ms=("completion_ms", "mean"),
            median_completion_ms=("completion_ms", "median"),
            p95_completion_ms=("completion_ms", lambda values: values.quantile(.95)),
            avg_execution_ms=("execution_ms", "mean"),
        ).sort_values("avg_completion_ms")
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
