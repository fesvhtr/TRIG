"""Diagnostic images + judge scores -> offline-DPO preference manifest (Flow-Factory V2) with
relation-aware Pareto pairing (rules: trig_rl/rewards/relations.py, RELATIONS[*].prefers).

    python trig_rl/data/make_dpo_pairs.py --scores "data/diag/scores/qwen_image_21__v2*.jsonl" \
        --images data/diag/images/qwen_image_21 --prompts data/diag/prompts_sub.jsonl \
        --dtm trig_rl/data/dtm.json --out trig_rl/data/dpo_pairs
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rewards.relations import DTM, prefers  # noqa: E402
from trig.evaluation.score_io import load_scores, pair_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dtm", default=None)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-pairs-per-prompt", type=int, default=2)
    ap.add_argument("--out", default="trig_rl/data/dpo_pairs")
    args = ap.parse_args()

    prompts = {r["data_id"]: r for r in (json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip())}
    dtm = DTM.load(args.dtm) if args.dtm else None
    img_root = os.path.abspath(args.images)
    by_prompt = defaultdict(list)  # data_id -> [(seed, (sa, sb))]
    for pair, pts in pair_points(load_scores(args.scores)).items():
        for did, seed, sa, sb in pts:
            if did in prompts and os.path.exists(os.path.join(img_root, f"{did}_s{seed}.png")):
                by_prompt[did].append((seed, (sa, sb)))

    os.makedirs(args.out, exist_ok=True)
    stats = Counter()
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for did, items in sorted(by_prompt.items()):
            if len(items) < 2:
                continue
            a, b = prompts[did]["dimensions"]
            rel, dom = dtm.relation(a, b, default="undefined") if dtm else ("undefined", None)
            cands = [(sum(w) - sum(l), sw, sl, w, l) for (sw, w), (sl, l) in itertools.permutations(items, 2) if prefers(w, l, rel, dom, args.margin)]
            for m, sw, sl, w, l in sorted(cands, key=lambda x: -x[0])[:args.max_pairs_per_prompt]:
                stats[rel] += 1
                f.write(json.dumps({
                    "schema_version": 2, "input": {"prompt": prompts[did]["prompt"], "media": []},
                    "supervision": {"type": "preference",
                                    "chosen": {"media": [{"type": "image", "path": os.path.join(img_root, f"{did}_s{sw}.png")}]},
                                    "rejected": {"media": [{"type": "image", "path": os.path.join(img_root, f"{did}_s{sl}.png")}]}},
                    "metadata": {"data_id": did, "pair": f"{a}_{b}", "relation": rel, "dominant": dom,
                                 "chosen_scores": {a: round(w[0], 2), b: round(w[1], 2)}, "rejected_scores": {a: round(l[0], 2), b: round(l[1], 2)},
                                 "margin": round(m, 2)}}, ensure_ascii=False) + "\n")
    print(f"prompts with >=2 images: {sum(len(v) >= 2 for v in by_prompt.values())}, pairs: {sum(stats.values())}, by relation: {dict(stats)} -> {args.out}/train.jsonl")


if __name__ == "__main__":
    main()
