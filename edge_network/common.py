"""Shared task definitions, wire protocol, and telemetry paths."""
from __future__ import annotations
import json, socket, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
FEATURES = ["size", "complexity", "cpu", "ram", "queue", "latency_ms", "bandwidth_mbps"]
TASK_KINDS = ["sort", "matrix", "preprocess", "numeric"]
FL_FEATURES = FEATURES + [f"kind_{kind}" for kind in TASK_KINDS]

def send_request(host: str, port: int, payload: dict, timeout: float = 120) -> dict:
    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps(payload) + "\n").encode())
        chunks = bytearray()
        while b"\n" not in chunks:
            part = sock.recv(65536)
            if not part: break
            chunks.extend(part)
    if not chunks: raise ConnectionError(f"No response from {host}:{port}")
    result = json.loads(bytes(chunks).split(b"\n", 1)[0])
    result["round_trip_ms"] = (time.perf_counter() - started) * 1000
    return result

def csv_append(path: Path, row: dict) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file=not path.exists() or path.stat().st_size==0
    if not new_file:
        with path.open(newline="",encoding="utf-8") as f:
            reader=csv.DictReader(f); fields=list(reader.fieldnames or []); old_rows=list(reader)
        extra=[key for key in row if key not in fields]
        if extra:
            fields.extend(extra)
            with path.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(old_rows)
    else:
        fields=list(row)
    with path.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if new_file: writer.writeheader()
        writer.writerow(row)

def task_features(task: dict, metrics: dict) -> list[float]:
    return [float(task.get("size", 1)), float(task.get("complexity", 1)),
            float(metrics.get("cpu_percent", 0)), float(metrics.get("ram_percent", 0)),
            float(metrics.get("queue", 0)), float(metrics.get("latency_ms", 0)),
            float(metrics.get("bandwidth_mbps", 100))]

def federated_task_features(task: dict, metrics: dict) -> list[float]:
    kind=task.get("kind","sort")
    return task_features(task,metrics)+[float(kind==name) for name in TASK_KINDS]

def federated_row_features(row: dict) -> list[float]:
    kind=row.get("kind") or TASK_KINDS[max(0,min(3,int(float(row.get("complexity",1)))-1))]
    return [float(row[k]) for k in FEATURES]+[float(kind==name) for name in TASK_KINDS]
