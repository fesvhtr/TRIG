"""Judge scores -> dtm.json: relation type per dimension pair, using the same classifier as the
paper's Relationator (trig/evaluation/relation_rules.py).

    python trig_rl/relations/build_dtm.py --scores "data/diag/scores/qwen_image_21__v1*.jsonl" --seed 0 --out trig_rl/data/dtm.json
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trig.evaluation.relation_rules import THRESHOLDS, classify_pair  # noqa: E402
from trig.evaluation.score_io import load_scores, pair_arrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="score jsonl files or globs (one protocol)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="only this generation seed (default: all)")
    for k, v in THRESHOLDS.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    args = ap.parse_args()
    thr = {k: getattr(args, k) for k in THRESHOLDS}

    out = {"thresholds": thr, "sources": args.scores, "pairs": {}}
    for pair, pts in sorted(pair_arrays(load_scores(args.scores, seed=args.seed)).items()):
        a, b = pair.split("_")
        rel, dom, info = classify_pair(pts[:, 0], pts[:, 1], thr)
        out["pairs"][pair] = {**info, "type": rel, "dominant": [a, b][dom] if dom is not None else None}
        print(f"{pair:12s} n={info['n']:5d} rho={info['corr']:+.2f} syn={info['synergy_density']:.2f} bot={info['bottleneck_density']:.2f} -> {rel}" + (f" (dominant {[a, b][dom]})" if dom is not None else ""))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("summary:", dict(Counter(v["type"] for v in out["pairs"].values())), "->", args.out)


if __name__ == "__main__":
    main()
