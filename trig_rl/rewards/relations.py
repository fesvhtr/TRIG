"""Relation types between two TRIG dimensions and what each means for training.

Each relation knows two things: how to turn a sample's two scores into one scalar reward
(`scalar`), and when one sample should be preferred over another as a DPO pair (`prefers`).
Pareto utilities for group rewards live at the bottom. Relations are identified by name
("synergy", "bottleneck", "tilt", "dispersion", "undefined"); for "tilt", `dominant` is the
index (0/1) of the dimension that tends to win.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np


def dominates(w: Sequence[float], l: Sequence[float], margin: float = 0.0) -> bool:
    """w is no worse than l on every dimension and better by >= margin on at least one."""
    w, l = np.asarray(w, float), np.asarray(l, float)
    return bool(np.all(w >= l) and np.any(w - l >= max(margin, 1e-9)))


@dataclass(frozen=True)
class Relation:
    name: str
    scalar: Callable[[float, float, Optional[int]], float]          # (a, b, dominant) -> reward
    prefers: Callable[[np.ndarray, np.ndarray, Optional[int], float], bool]  # (w, l, dominant, eps)


def _mean(a, b, _=None):
    return 0.5 * (a + b)


def _bottleneck_scalar(a, b, _=None, mix=0.7):
    return mix * min(a, b) + (1 - mix) * _mean(a, b)


def _tilt_scalar(a, b, dominant, weak_weight=0.7, floor=0.5, penalty=1.0):
    dom, weak = (a, b) if dominant == 0 else (b, a)
    return weak_weight * weak + (1 - weak_weight) * dom - penalty * max(0.0, floor - dom)


def _tilt_prefers(w, l, dominant, eps):
    weak = 1 - dominant
    return dominates(w, l, eps) and w[weak] >= l[weak] + eps and w[dominant] >= l[dominant]


RELATIONS: Dict[str, Relation] = {
    # both dimensions move together: just push the mean; any non-worsening gain is a valid pair
    "synergy": Relation("synergy", _mean, lambda w, l, d, eps: (w.sum() - l.sum()) >= 2 * eps and bool(np.all(w >= l - 1e-9))),
    # both tend to be low: reward the floor; a pair must lift the weaker dimension
    "bottleneck": Relation("bottleneck", _bottleneck_scalar, lambda w, l, d, eps: dominates(w, l, eps) and w.min() >= l.min() + eps),
    # one dimension wins at the other's expense: favour the weak one, keep the dominant above a floor
    "tilt": Relation("tilt", _tilt_scalar, _tilt_prefers),
    # weakly related: plain mean; only strict dominance with a wide margin counts as a pair
    "dispersion": Relation("dispersion", _mean, lambda w, l, d, eps: bool(np.any(np.asarray(w) - np.asarray(l) >= 2 * eps)) and dominates(w, l, eps)),
    "undefined": Relation("undefined", _mean, lambda w, l, d, eps: dominates(w, l, eps)),
}


def combine(a: float, b: float, relation: str, dominant: Optional[int] = None) -> float:
    rel = RELATIONS[relation]
    if relation == "tilt" and dominant is None:
        rel = RELATIONS["undefined"]
    return float(rel.scalar(a, b, dominant))


def prefers(w, l, relation: str, dominant: Optional[int], eps: float) -> bool:
    rel = RELATIONS[relation]
    if relation == "tilt" and dominant is None:
        rel = RELATIONS["undefined"]
    return bool(rel.prefers(np.asarray(w, float), np.asarray(l, float), dominant, eps))


class DTM:
    """pair -> relation lookup from a dtm.json (built by trig_rl/relations/build_dtm.py). Used by
    the dataset builders to stamp `relation` / `dominant` into each row; rewards read those."""

    def __init__(self, pairs: Dict[str, dict]):
        self.map = {}
        for pair, info in pairs.items():
            a, b = pair.split("_")
            dom = info.get("dominant")
            self.map[(a, b)] = (info["type"], {a: 0, b: 1}.get(dom))
            self.map[(b, a)] = (info["type"], {b: 0, a: 1}.get(dom))

    @classmethod
    def load(cls, path: str) -> "DTM":
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("pairs", d))

    def relation(self, dim_a: str, dim_b: str, default: Optional[str] = None):
        """(relation, dominant_index) for the ordered pair; KeyError unless a default is given."""
        if (dim_a, dim_b) in self.map:
            return self.map[(dim_a, dim_b)]
        if default is None:
            raise KeyError(f"pair {dim_a}_{dim_b} not in DTM")
        return default, None


def pareto_fronts(points: np.ndarray) -> list:
    """Non-dominated sorting (maximisation). Front index per point, 0 = best."""
    n = len(points); fronts = [-1] * n; remaining = set(range(n)); k = 0
    while remaining:
        front = [i for i in remaining if not any(j != i and dominates(points[j], points[i]) for j in remaining)]
        for i in front:
            fronts[i] = k
        remaining -= set(front); k += 1
    return fronts


def pareto_rewards(scores: np.ndarray, scalar: Sequence[float], front_weight: float = 0.7) -> np.ndarray:
    """Group reward in [0, 1]: Pareto-front rank (dominant term) mixed with the relation scalar."""
    fronts = np.array(pareto_fronts(scores), float)
    return front_weight * (1.0 - fronts / max(1.0, fronts.max() + 1.0)) + (1.0 - front_weight) * np.asarray(scalar, float)
