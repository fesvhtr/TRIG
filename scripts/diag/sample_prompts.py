"""Stratified sample of TRIG text-to-image prompts for the dimension diagnostic.

Drops every subset that involves the excluded dimensions (trig.config_judge_v2.EXCLUDED_DIMS) and takes
`--per-pair` prompts from each remaining dimension pair with a fixed seed.

    python scripts/diag/sample_prompts.py --per-pair 7 --out data/diag/prompts.jsonl
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trig.config_judge_v2 import EXCLUDED_DIMS  # noqa: E402


def subset_of(row):
    dims = row.get("dimensions")
    if isinstance(dims, (list, tuple)) and len(dims) == 2:
        return tuple(dims)
    return tuple(row["data_id"].split("_")[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="RISys-Lab/TRIG")
    ap.add_argument("--split", default="text_to_image")
    ap.add_argument("--per-pair", type=int, default=7, help="prompts per dimension pair; 0 = take all (full benchmark run)")
    ap.add_argument("--exclude", nargs="*", default=list(EXCLUDED_DIMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/diag/prompts.jsonl")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    groups = defaultdict(list)
    for row in ds:
        pair = subset_of(row)
        if any(d in args.exclude for d in pair):
            continue
        groups[pair].append(row)

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for pair in sorted(groups):
            rows = groups[pair]
            picked = list(rows) if args.per_pair <= 0 else rng.sample(rows, min(args.per_pair, len(rows)))
            for r in picked:
                rec = {"data_id": r["data_id"], "prompt": r["prompt"], "dimensions": list(pair)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(f"{'_'.join(pair):12s} pool={len(rows):4d} picked={len(picked)}")
    print(f"wrote {n} prompts from {len(groups)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
