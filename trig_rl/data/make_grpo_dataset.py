"""TRIG text-to-image prompts -> Flow-Factory online dataset (train.jsonl / test.jsonl).

Rows: {"prompt", "data_id", "pair", "relation", "dominant"}; the non-prompt columns reach the
reward as the per-sample `metadata` JSON. `relation`/`dominant` come from a DTM json (build_dtm.py);
without --dtm every pair is "undefined" (plain mean reward).

    python trig_rl/data/make_grpo_dataset.py --dtm trig_rl/data/dtm.json --out trig_rl/data/grpo
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rewards.relations import DTM  # noqa: E402
from trig.config_judge_v2 import EXCLUDED_DIMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="RISys-Lab/TRIG")
    ap.add_argument("--split", default="text_to_image")
    ap.add_argument("--exclude", nargs="*", default=list(EXCLUDED_DIMS))
    ap.add_argument("--dtm", default=None)
    ap.add_argument("--test-per-pair", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="trig_rl/data/grpo")
    args = ap.parse_args()

    dtm = DTM.load(args.dtm) if args.dtm else None
    rows = defaultdict(list)
    for r in load_dataset(args.dataset, split=args.split):
        a, b = r["dimensions"]
        if a in args.exclude or b in args.exclude:
            continue
        rel, dom = dtm.relation(a, b, default="undefined") if dtm else ("undefined", None)
        rows[f"{a}_{b}"].append({"prompt": r["prompt"], "data_id": r["data_id"], "pair": f"{a}_{b}", "relation": rel, "dominant": dom})

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    counts = {"train": 0, "test": 0}
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as ftr, open(os.path.join(args.out, "test.jsonl"), "w", encoding="utf-8") as fte:
        for pair in sorted(rows):
            items = rows[pair]; rng.shuffle(items)
            for split, f, chunk in (("test", fte, items[:args.test_per_pair]), ("train", ftr, items[args.test_per_pair:])):
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n"); counts[split] += 1
    print(f"{len(rows)} pairs -> {counts} in {args.out}")


if __name__ == "__main__":
    main()
