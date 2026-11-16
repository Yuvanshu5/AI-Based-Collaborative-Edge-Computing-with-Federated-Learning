"""Shared, deterministic size-grouped train/test split for central ML and FL."""
from __future__ import annotations
import csv
from pathlib import Path
from .common import DATA

def load_workload(path: Path | None = None) -> list[dict]:
    path=path or DATA/"workload.csv"
    with path.open(encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames or []); rows=list(reader)
    seen={}; changed=False
    for row in rows:
        original=row.get("task_id",""); count=seen.get(original,0); seen[original]=count+1
        if count:
            row["task_id"]=f"{original}-dup{count+1}"; changed=True
    if changed:
        with path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows

def make_manifest(rows: list[dict], path: Path | None = None) -> dict[str,str]:
    """Hold out fixed middle sizes so no measured size leaks across the split."""
    path=path or DATA/"split_manifest.csv"
    ids=[r["task_id"] for r in rows]
    groups=[int(float(r["size"])) for r in rows]
    if len(set(groups))<3: raise ValueError("Need at least three distinct sizes for a size-grouped train/test split")
    held_out={2000,8000}
    # Keep a full unseen heavy size for evaluating heavy-workload models too.
    if 500000 in set(groups): held_out.add(500000)
    if not held_out.issubset(set(groups)):
        raise ValueError("Dataset must include the held-out sizes 2000 and 8000; collect it with collect-dataset")
    split={task_id:("test" if size in held_out else "train") for task_id,size in zip(ids,groups)}
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["task_id","split"]); writer.writeheader()
        writer.writerows({"task_id":k,"split":v} for k,v in split.items())
    return split

def read_manifest(path: Path | None = None) -> dict[str,str]:
    path=path or DATA/"split_manifest.csv"
    if not path.exists(): return {}
    with path.open(encoding="utf-8",newline="") as f: return {r["task_id"]:r["split"] for r in csv.DictReader(f)}
