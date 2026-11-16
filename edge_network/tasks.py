"""Deterministic, allow-listed CPU workloads; tasks never execute arbitrary code."""
from __future__ import annotations
import random, time

def execute(task: dict):
    kind, size = task["kind"], max(1, int(task.get("size", 1000)))
    start = time.perf_counter()
    if kind == "sort":
        rng = random.Random(int(task.get("seed", 1)))
        data = [rng.randrange(size * 10) for _ in range(size)]
        result = {"count": len(sorted(data)), "checksum": sum(data)}
    elif kind == "matrix":
        n = min(240, max(2, int(size ** 0.5)))
        repeats=max(1,min(10,int(task.get("iterations",4 if size>=250000 else 1))))
        # Pure Python matrix multiplication keeps the demo dependency-light.
        a = [[(i + j) % 7 for j in range(n)] for i in range(n)]
        b = [[(i * j + 1) % 11 for j in range(n)] for i in range(n)]
        checksum = 0
        for _ in range(repeats):
            for i in range(n):
                for j in range(n): checksum += sum(a[i][k] * b[k][j] for k in range(n))
        result = {"dimension": n, "repetitions": repeats, "checksum": checksum}
    elif kind == "preprocess":
        values = [(i % 997) / 997 for i in range(size)]
        mean = sum(values) / len(values)
        normalized = [value - mean for value in values]
        result = {"count": size, "mean": mean, "normalized_checksum": sum(normalized)}
    elif kind == "numeric":
        # Repeated stable numerical work scaled by size.
        acc = 0.0
        for i in range(size * 20): acc += ((i % 101) / 101.0) ** 2
        result = {"iterations": size * 20, "value": acc}
    else: raise ValueError(f"Unsupported task kind: {kind}")
    return result, (time.perf_counter() - start) * 1000

def generate_tasks(count: int, seed: int = 7):
    rng = random.Random(seed)
    kinds = ["sort", "matrix", "preprocess", "numeric"]
    for i in range(count):
        kind = kinds[i % len(kinds)]
        size = rng.choice([500, 1000, 2000, 4000])
        yield {"task_id": f"task-{time.time_ns()}-{i:04d}", "kind": kind,
               "size": size, "complexity": kinds.index(kind) + 1, "seed": seed + i}
