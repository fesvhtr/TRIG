"""Readers for the judge score files written by scripts/diag (one JSON line per image x dim)."""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from trig.config_judge_v2 import EXCLUDED_DIMS, pair_allowed

Key = Tuple[str, str, int]  # (pair, data_id, seed)


def load_scores(paths_or_globs: Iterable[str], seed: Optional[int] = None,
                protocol: Optional[str] = None, exclude=EXCLUDED_DIMS) -> Dict[Key, Dict[str, float]]:
    """{(pair, data_id, seed): {dim: score}}; rows without a score or from an excluded pair are
    dropped; the first occurrence of a key wins (subset and shard files overlap)."""
    out: Dict[Key, Dict[str, float]] = defaultdict(dict)
    for pattern in paths_or_globs:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("score") is None or not pair_allowed(r["pair"], exclude) or (seed is not None and r["seed"] != seed) \
                            or (protocol is not None and r.get("protocol") != protocol):
                        continue
                    out[(r["pair"], r["data_id"], r["seed"])].setdefault(r["dim"], r["score"])
    return out


def pair_points(by_img: Dict[Key, Dict[str, float]]) -> Dict[str, List[Tuple[str, int, float, float]]]:
    """{pair: [(data_id, seed, score_a, score_b)]} for images that have both dimensions."""
    pairs = defaultdict(list)
    for (pair, did, seed), d in by_img.items():
        a, b = pair.split("_")
        if a in d and b in d:
            pairs[pair].append((did, seed, d[a], d[b]))
    return dict(pairs)


def pair_arrays(by_img) -> Dict[str, np.ndarray]:
    return {p: np.array([(x, y) for _, _, x, y in pts], float) for p, pts in pair_points(by_img).items()}
