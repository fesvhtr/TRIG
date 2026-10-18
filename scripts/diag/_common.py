"""Shared driver for the diagnostic scoring scripts: prompt loading, image indexing, resume,
sharding and the concurrent JSONL writer."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Tuple

from tqdm import tqdm


def add_common_args(ap: argparse.ArgumentParser, default_workers: int) -> None:
    ap.add_argument("--model", default="qwen_image_21", help="generator alias (image folder name)")
    ap.add_argument("--prompts", default="data/diag/prompts.jsonl")
    ap.add_argument("--images", default="data/diag/images")
    ap.add_argument("--out", default="data/diag/scores")
    ap.add_argument("--workers", type=int, default=default_workers)
    ap.add_argument("--shard", default="0/1", help="i/n: score every n-th prompt starting at i (parallel jobs)")
    ap.add_argument("--tag", default="", help="extra suffix for the output file name")


def load_prompts(path: str, shard: str = "0/1") -> List[dict]:
    i, n = (int(x) for x in shard.split("/"))
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()][i::n]


def index_images(img_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """data_id -> [(seed, path)] from files named <data_id>_s<seed>.png (one directory listing)."""
    out = defaultdict(list)
    if not os.path.isdir(img_dir):
        return out
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".png") or ".partial" in fn:
            continue
        stem, _, seed = fn[:-4].rpartition("_s")
        if stem and seed.isdigit():
            out[stem].append((int(seed), os.path.join(img_dir, fn)))
    return out


def output_path(args, protocol: str) -> str:
    i, n = (int(x) for x in args.shard.split("/"))
    suffix = (f".shard{i}of{n}" if n > 1 else "") + (f".{args.tag}" if args.tag else "")
    return os.path.join(args.out, f"{args.model}__{protocol}{suffix}.jsonl")


def load_done(out_path: str) -> set:
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            if r.get("score") is not None:
                done.add((r["data_id"], r["seed"], r["dim"]))
    return done


def build_jobs(records: List[dict], images: Dict[str, List[Tuple[int, str]]], done: set) -> List[dict]:
    """One job per (image, dim) not yet scored."""
    jobs = []
    for r in records:
        for seed, path in images.get(r["data_id"], []):
            for dim in r["dimensions"]:
                if (r["data_id"], seed, dim) not in done:
                    jobs.append({"record": r, "seed": seed, "dim": dim, "path": path})
    return jobs


def run_jobs(jobs: Iterable[dict], fn: Callable[[dict], dict], workers: int, out_path: str) -> None:
    """Run ``fn(job) -> result row`` concurrently, appending rows to ``out_path`` as they finish."""
    jobs = list(jobs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(fn, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()


def base_row(job: dict, protocol: str) -> dict:
    r = job["record"]
    return {"data_id": r["data_id"], "seed": job["seed"], "dim": job["dim"], "pair": "_".join(r["dimensions"]),
            "protocol": protocol}
