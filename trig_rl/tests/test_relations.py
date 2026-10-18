"""Offline tests for the relation registry, Pareto ranking and the judge module."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rewards.relations import DTM, combine, dominates, pareto_fronts, pareto_rewards, prefers  # noqa: E402
from trig.judge import build_messages, logprobs_to_score, token_to_rating_word  # noqa: E402


def test_scalars():
    assert combine(0.9, 0.3, "synergy") == 0.6 and combine(0.9, 0.3, "undefined") == 0.6
    assert combine(0.9, 0.3, "bottleneck") == 0.7 * 0.3 + 0.3 * 0.6
    assert abs(combine(0.9, 0.3, "tilt", 0) - (0.7 * 0.3 + 0.3 * 0.9)) < 1e-9
    assert combine(0.4, 0.3, "tilt", 0) < combine(0.9, 0.3, "tilt", 0)   # dominant under the floor is penalised
    assert combine(0.9, 0.3, "tilt", None) == 0.6                          # no dominant known -> mean


def test_pairing():
    eps = 0.05
    assert dominates([0.8, 0.6], [0.7, 0.6], eps) and not dominates([0.8, 0.5], [0.7, 0.6], eps)
    assert prefers([0.8, 0.6], [0.7, 0.6], "undefined", None, eps)
    assert not prefers([0.8, 0.5], [0.7, 0.6], "undefined", None, eps)      # trades one dim for another
    assert prefers([0.8, 0.5], [0.7, 0.5], "synergy", None, eps)
    assert prefers([0.9, 0.6], [0.9, 0.4], "bottleneck", None, eps) and not prefers([0.9, 0.6], [0.7, 0.6], "bottleneck", None, eps)
    assert prefers([0.9, 0.6], [0.9, 0.4], "tilt", 0, eps) and not prefers([0.95, 0.4], [0.9, 0.4], "tilt", 0, eps)
    assert prefers([0.9, 0.6], [0.9, 0.4], "dispersion", None, eps) and not prefers([0.9, 0.48], [0.9, 0.4], "dispersion", None, eps)


def test_pareto():
    pts = np.array([[0.9, 0.9], [0.9, 0.2], [0.2, 0.9], [0.5, 0.5], [0.1, 0.1]])
    assert pareto_fronts(pts) == [0, 1, 1, 1, 2]
    r = pareto_rewards(pts, pts.min(1), 0.7)
    assert r[0] > max(r[1:]) and r[4] < min(r[:4]) and r[3] > r[1] and r[3] > r[2]


def test_dtm_and_judge():
    d = DTM({"IQ-R_TA-S": {"type": "tilt", "dominant": "TA-S"}})
    assert d.relation("IQ-R", "TA-S") == ("tilt", 1) and d.relation("TA-S", "IQ-R") == ("tilt", 0)
    assert d.relation("D-K", "TA-C", default="undefined") == ("undefined", None)
    class T:
        def __init__(s, t, lp): s.token, s.logprob = t, lp
    assert token_to_rating_word("Ter") == "terrible" and token_to_rating_word("<think>") is None
    assert abs(logprobs_to_score([T("Good", math.log(.6)), T("Excellent", math.log(.4))])[0] - 0.85) < 1e-9
    assert build_messages("v1", "IQ-A", "p", "u")[1]["content"][0]["text"] == "Evaluate this image."
    assert "Knowledge" in build_messages("v2", "D-K", "p", "u")[0]["content"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
