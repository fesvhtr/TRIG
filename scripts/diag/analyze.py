"""Analyse diagnostic scores: how coupled the dimensions are, and how much of the coupling is
prompt difficulty vs. judge halo vs. protocol. Reads data/diag/scores/<model>__*.jsonl, groups rows
by their `protocol` field, writes data/diag/report/<model>.md and one CSV per protocol.

Per pair: rho cross-prompt (per-prompt means; what the DTM uses), rho within-prompt (prompt mean
removed, multi-seed prompts only), synergy / bottleneck densities. Per dimension: mean, std,
prompt-variance share and seed std (multi-seed prompts only). Plus Spearman agreement between
every pair of protocols.

    python scripts/diag/analyze.py --model qwen_image_21
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trig.evaluation.relation_rules import THRESHOLDS, spearman  # noqa: E402
from trig.evaluation.score_io import load_scores, pair_points  # noqa: E402


def rho(x, y):
    return spearman(x, y) if len(x) >= 4 else float("nan")


def md_table(headers, rows):
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def pair_stats(by_img):
    table = {}
    for pair, pts in sorted(pair_points(by_img).items()):
        s1 = np.array([p[2] for p in pts]); s2 = np.array([p[3] for p in pts])
        per_prompt = defaultdict(lambda: ([], []))
        for did, _, x, y in pts:
            per_prompt[did][0].append(x); per_prompt[did][1].append(y)
        m1 = np.array([np.mean(v[0]) for v in per_prompt.values()]); m2 = np.array([np.mean(v[1]) for v in per_prompt.values()])
        multi = {d for d, v in per_prompt.items() if len(v[0]) >= 2}
        w1 = np.array([x - np.mean(per_prompt[d][0]) for d, _, x, _ in pts if d in multi])
        w2 = np.array([y - np.mean(per_prompt[d][1]) for d, _, _, y in pts if d in multi])
        table[pair] = dict(
            n_prompts=len(per_prompt), multi_seed=len(multi), mean_a=s1.mean(), mean_b=s2.mean(),
            rho_cross=rho(m1, m2), rho_within=rho(w1, w2) if len(multi) >= 3 else float("nan"), rho_image=rho(s1, s2),
            synergy=float(np.mean((s1 > THRESHOLDS["synergy"]) & (s2 > THRESHOLDS["synergy"]))),
            bottleneck=float(np.mean((s1 < THRESHOLDS["bottleneck"]) & (s2 < THRESHOLDS["bottleneck"]))))
    return table


def dim_stats(by_img):
    per_dim = defaultdict(lambda: defaultdict(list))  # dim -> data_id -> scores over seeds
    for (_, did, _), d in by_img.items():
        for dim, s in d.items():
            per_dim[dim][did].append(s)
    table = {}
    for dim, pp in sorted(per_dim.items()):
        allv = np.array([s for v in pp.values() for s in v])
        multi = [v for v in pp.values() if len(v) > 1]
        mv = np.array([s for v in multi for s in v]); mm = np.array([np.mean(v) for v in multi])
        table[dim] = dict(n_img=int(allv.size), mean=allv.mean(), std=allv.std(),
                          prompt_var_share=float(mm.var() / mv.var()) if mv.size and mv.var() > 0 else float("nan"),
                          seed_std=float(np.mean([np.std(v) for v in multi])) if multi else float("nan"),
                          ge_0_8=float(np.mean(allv >= 0.8)), le_0_5=float(np.mean(allv <= 0.5)))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen_image_21")
    ap.add_argument("--scores", default="data/diag/scores")
    ap.add_argument("--out", default="data/diag/report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.scores, f"{args.model}__*.jsonl")))  # deterministic first-wins dedupe
    protocols = sorted({json.loads(l)["protocol"] for f in files for l in open(f) if l.strip()})
    if not protocols:
        raise SystemExit(f"no score files for {args.model} in {args.scores}")
    data = {p: load_scores(files, protocol=p) for p in protocols}

    md = [f"# Diagnostic report: {args.model}\n"]
    for p, by_img in data.items():
        pairs, dims = pair_stats(by_img), dim_stats(by_img)
        md += [f"\n## Protocol `{p}`  ({sum(len(d) for d in by_img.values())} scores)\n", "### Per dimension\n",
               md_table(["dim", "n_img", "mean", "std", "prompt-var share (multi-seed)", "seed std", "≥0.8", "≤0.5"],
                        [[k, *d.values()] for k, d in dims.items()]),
               "\n### Per pair\n",
               md_table(["pair", "n_prompts", "multi-seed", "mean A", "mean B", "rho cross-prompt", "rho within-prompt", "rho per-image", "synergy", "bottleneck"],
                        [[k, *d.values()] for k, d in pairs.items()])]
        agg = defaultdict(lambda: ([], []))
        for pair, d in pairs.items():
            for dim in pair.split("_"):
                agg[dim][0].append(d["rho_cross"])
                if not np.isnan(d["rho_within"]):
                    agg[dim][1].append(d["rho_within"])
        md += ["\n### Average rho per dimension (cross-prompt / within-prompt)\n",
               md_table(["dim", "avg rho cross", "avg rho within", "n pairs"],
                        [[dim, float(np.nanmean(c)), float(np.mean(w)) if w else float("nan"), len(c)] for dim, (c, w) in sorted(agg.items())])]
        with open(os.path.join(args.out, f"{args.model}__{p}__pairs.csv"), "w", newline="") as f:
            w = csv.writer(f); keys = list(next(iter(pairs.values())).keys()); w.writerow(["pair", *keys])
            for pair, d in pairs.items():
                w.writerow([pair, *[f"{v:.2f}" if isinstance(v, float) else v for v in d.values()]])

    for i, pa in enumerate(protocols):
        for pb in protocols[i + 1:]:
            per_dim = defaultdict(lambda: ([], []))
            for key, d in data[pa].items():
                for dim, s in d.items():
                    if key in data[pb] and dim in data[pb][key]:
                        per_dim[dim][0].append(s); per_dim[dim][1].append(data[pb][key][dim])
            md += [f"\n## Protocol agreement ({pa} vs {pb}), Spearman per dimension\n",
                   md_table(["dim", "rho", "n", f"mean {pa}", f"mean {pb}"],
                            [[dim, rho(x, y), len(x), float(np.mean(x)), float(np.mean(y))] for dim, (x, y) in sorted(per_dim.items())])]

    md.append("\n## How to read\n- `rho cross-prompt` high but `rho within-prompt` low: coupling is prompt difficulty, not the judge.\n"
              "- both high under `baseline`, lower under `v1`: judge halo, fixed by the protocol.\n"
              "- both high under every protocol: the two dimensions are one thing for this generator+judge; merge or redefine.\n"
              "- `prompt-var share` near 1 with tiny `seed std`: the dimension is decided by the prompt, generation randomness barely matters.\n")
    path = os.path.join(args.out, f"{args.model}.md")
    open(path, "w", encoding="utf-8").write("\n".join(md))
    print(f"written {path}")


if __name__ == "__main__":
    main()
