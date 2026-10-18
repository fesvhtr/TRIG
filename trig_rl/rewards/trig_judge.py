"""Flow-Factory reward models backed by the TRIGScore judge (trig.judge).

Each sample's metadata (from make_grpo_dataset.py) carries `pair` ("A_B"), `relation` and
`dominant`. Both dimensions are scored once per image (shared client cache); the classes differ
only in what they return:

  TRIGDimReward         one dimension (dim_index 0/1). Two of them + Flow-Factory `gdpo`
                        aggregation = per-dimension group normalisation and sum (Multi-GRPO baseline).
  TRIGPairReward        the relation-aware scalar (relations.combine).
  TRIGParetoGroupReward groupwise: Pareto-front rank within the group, tie-broken by the scalar.

YAML (dotted path; TRIG root on PYTHONPATH):
  rewards:
    - name: trig_pareto
      reward_model: trig_rl.rewards.trig_judge.TRIGParetoGroupReward
      api_base_url: http://127.0.0.1:8000/v1/     # or env TRIG_JUDGE_ENDPOINT
      vlm_model: Qwen/Qwen3.5-9B                   # or env TRIG_JUDGE_MODEL
      protocol: v2
      score_log: trig_rl/saves/<run>/judge_scores.jsonl   # optional per-sample log
"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards.abc import GroupwiseRewardModel, PointwiseRewardModel, RewardModelOutput

_TRIG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TRIG_ROOT not in sys.path:
    sys.path.insert(0, _TRIG_ROOT)
from trig.judge import JudgeClient, resolve_judge  # noqa: E402

from .relations import combine, pareto_rewards  # noqa: E402

_CLIENTS: Dict[tuple, JudgeClient] = {}   # one client (and score cache) per endpoint/model/protocol
_LOCK = threading.Lock()


class _TRIGBase:
    required_fields: Tuple[str, ...] = ("image", "prompt", "metadata")
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        ek = config.extra_kwargs
        cfg = resolve_judge(ek.get("api_base_url"), ek.get("vlm_model"), ek.get("api_key"))
        protocol = ek.get("protocol", "v2")
        with _LOCK:
            key = (cfg.endpoint, cfg.model, protocol)
            if key not in _CLIENTS:
                _CLIENTS[key] = JudgeClient(cfg.endpoint, cfg.model, cfg.api_key, protocol=protocol,
                                            max_concurrent=ek.get("max_concurrent", 32))
            self.judge = _CLIENTS[key]
        self.front_weight = ek.get("front_weight", 0.7)
        # Flow-Factory drops RewardModelOutput.extra_info, so persist per-sample scores ourselves.
        self.score_log = ek.get("score_log")
        if self.score_log:
            self.score_log = self.score_log.replace(".jsonl", f".rank{accelerator.process_index}.jsonl")
            os.makedirs(os.path.dirname(os.path.abspath(self.score_log)), exist_ok=True)

    @staticmethod
    def _meta(md: str) -> Tuple[Tuple[str, str], str, Optional[int]]:
        d = json.loads(md)
        a, b = d["pair"].split("_")
        return (a, b), d.get("relation", "undefined"), d.get("dominant")

    def _score(self, prompt, image, metadata):
        """pairs, (n, 2) scores in pair order, relation-aware scalars."""
        metas = [self._meta(m) for m in metadata]
        items = [(img, p, dim) for img, p, ((a, b), _, _) in zip(image, prompt, metas) for dim in (a, b)]
        scores = np.asarray(self.judge.score(items), float).reshape(len(image), 2)
        scalars = [combine(s[0], s[1], rel, dom) for s, (_, rel, dom) in zip(scores, metas)]
        return metas, scores, scalars

    def _log(self, tag, prompt, metas, scores, scalars, rewards):
        if not self.score_log:
            return
        with open(self.score_log, "a", encoding="utf-8") as f:
            for p, ((a, b), rel, _), s, sc, r in zip(prompt, metas, scores, scalars, rewards):
                f.write(json.dumps({"tag": tag, "pair": f"{a}_{b}", "relation": rel, a: round(float(s[0]), 2), b: round(float(s[1]), 2),
                                    "scalar": round(float(sc), 2), "reward": round(float(r), 2), "prompt": p[:200]}, ensure_ascii=False) + "\n")

    def _output(self, tag, prompt, metas, scores, scalars, rewards):
        self._log(tag, prompt, metas, scores, scalars, rewards)
        return RewardModelOutput(rewards=torch.tensor(np.asarray(rewards, float), dtype=torch.float32))


class TRIGDimReward(_TRIGBase, PointwiseRewardModel):
    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.dim_index = int(config.extra_kwargs.get("dim_index", 0))

    @torch.no_grad()
    def __call__(self, prompt: List[str], image: Optional[List[Image.Image]] = None, metadata: Optional[List[str]] = None) -> RewardModelOutput:
        metas, scores, scalars = self._score(prompt, image, metadata)
        return self._output(f"dim{self.dim_index}", prompt, metas, scores, scalars, scores[:, self.dim_index])


class TRIGPairReward(_TRIGBase, PointwiseRewardModel):
    @torch.no_grad()
    def __call__(self, prompt: List[str], image: Optional[List[Image.Image]] = None, metadata: Optional[List[str]] = None) -> RewardModelOutput:
        metas, scores, scalars = self._score(prompt, image, metadata)
        return self._output("pair", prompt, metas, scores, scalars, scalars)


class TRIGParetoGroupReward(_TRIGBase, GroupwiseRewardModel):
    @torch.no_grad()
    def __call__(self, prompt: List[str], image: Optional[List[Image.Image]] = None, metadata: Optional[List[str]] = None) -> RewardModelOutput:
        metas, scores, scalars = self._score(prompt, image, metadata)
        return self._output("pareto", prompt, metas, scores, scalars, pareto_rewards(scores, scalars, self.front_weight))
