"""Pairwise relation classification shared by the paper's Relationator and trig_rl's DTM builder.

Priority (as in the paper): synergy (both high) -> bottleneck (both low) -> tilt (one dimension
tends to exceed the other on the trade-off region) -> dispersion (weak correlation) -> undefined.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats

THRESHOLDS: Dict[str, float] = dict(
    synergy=0.8,            # both scores above this -> synergy region
    bottleneck=0.5,         # both scores below this -> bottleneck region
    synergy_density=0.4,    # share of points in the synergy region needed
    bottleneck_density=0.4, # share of points in the bottleneck region needed
    dominance_ratio=1.5,    # |above| / |below| the regression line on the trade-off region
    tradeoff_corr=0.7,      # correlation below this -> dispersion
)
NAME_TO_CODE = {"synergy": 0, "dispersion": 1, "tilt_a": 2, "tilt_b": -2, "bottleneck": 3, "undefined": -1}


def spearman(s1, s2) -> float:
    if np.std(s1) == 0 or np.std(s2) == 0:
        return 0.0
    return float(stats.spearmanr(s1, s2)[0])


def classify_pair(s1: np.ndarray, s2: np.ndarray, thresholds: Optional[Dict[str, float]] = None,
                  corr_fn: Callable = spearman) -> Tuple[str, Optional[int], dict]:
    """Return (relation, dominant_index, info). dominant_index is 0 or 1 for ``tilt`` (which of
    the two dimensions tends to win), else None."""
    thr = {**THRESHOLDS, **(thresholds or {})}
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    syn = (s1 > thr["synergy"]) & (s2 > thr["synergy"])
    bot = (s1 < thr["bottleneck"]) & (s2 < thr["bottleneck"])
    other = ~(syn | bot)
    rho = corr_fn(s1, s2)
    info = {"n": int(len(s1)), "mean_a": round(float(s1.mean()), 2), "mean_b": round(float(s2.mean()), 2),
            "synergy_density": round(float(syn.mean()), 2), "bottleneck_density": round(float(bot.mean()), 2),
            "corr": round(rho, 2)}
    if info["synergy_density"] >= thr["synergy_density"]:
        return "synergy", None, info
    if info["bottleneck_density"] >= thr["bottleneck_density"]:
        return "bottleneck", None, info
    if other.sum() >= 10:
        x, y = s1[other], s2[other]
        slope, intercept = np.polyfit(x, y, 1)
        above = int((y > slope * x + intercept).sum())
        below = int(len(y) - above)
        ratio = max(above, below) / max(1, min(above, below))
        info.update({"above": above, "below": below, "ratio": round(float(ratio), 2)})
        if ratio > thr["dominance_ratio"]:
            # more points above the line: dim B exceeds what A predicts -> B dominates (index 1)
            return "tilt", (1 if above > below else 0), info
    if rho < thr["tradeoff_corr"]:
        return "dispersion", None, info
    return "undefined", None, info
