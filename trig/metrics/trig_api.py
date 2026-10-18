"""TRIGScore through any OpenAI-compatible chat endpoint (vLLM, OpenAI, DashScope, ...).

Thin BaseMetric adapter over :mod:`trig.judge`, which owns the rating scale, the prompts for
the ``baseline`` / ``v1`` / ``v2`` protocols, endpoint resolution (``TRIG_JUDGE_ENDPOINT``,
``TRIG_JUDGE_MODEL`` override the yaml) and the concurrent client.
"""
import os

from trig.judge import JudgeClient, resolve_judge
from trig.metrics.base import BaseMetric
from trig.utils.utils import image_data_url


class TRIGAPIMetric(BaseMetric):
    def __init__(self, API_KEY=None, endpoint=None, model_name=None, protocol="baseline", top_logprobs=10,
                 num_workers=8, max_retries=3, timeout=120.0, task=None, **kwargs):
        super().__init__(**kwargs)
        cfg = resolve_judge(endpoint, model_name, API_KEY)
        self.judge = JudgeClient(cfg.endpoint, cfg.model, cfg.api_key, protocol=protocol, task=task or "t2i",
                                 top_logprobs=top_logprobs, max_concurrent=num_workers, max_retries=max_retries,
                                 timeout=timeout)
        self.task = task
        self.model_name = cfg.model
        self.metric_name = "TRIGAPIMetric_" + cfg.model
        print(f"Initializing TRIGAPIMetric: endpoint={cfg.endpoint}, model={cfg.model}, protocol={protocol}, dimension={self.dimension}")

    def compute(self, image_path, prompt, *args, return_details=False):
        """Score one image on ``self.dimension``. ``args`` = () for t2i, (src_img,) for p2p,
        (src_img, item) for s2p; the source image is not shown to the judge (paper protocol)."""
        self.judge.task = self.task
        item = args[1] if len(args) == 2 else None
        return self.judge.score_one(image_data_url(image_path), prompt, self.dimension, item=item,
                                    return_details=return_details)

    def compute_batch(self, task, promp_data):
        if task is None:
            raise ValueError("task is None")
        self.task = task
        self.judge.task = task
        self.judge.protocol = self.judge.protocol
        ids = [d["data_id"] for d in promp_data]
        # The judge scores dimension `self.dimension` for every image; s2p passes the subject item.
        if task == "s2p":
            scores = [self.judge.score_one(image_data_url(d["gen_image_path"]), d["prompt"], self.dimension, item=d["item"])
                      for d in promp_data]
        else:
            scores = self.judge.score([(d["gen_image_path"], d["prompt"], self.dimension) for d in promp_data])
        return dict(zip(ids, [round(s, 3) for s in scores]))


if __name__ == "__main__":
    # Smoke check against a running judge:  TRIG_JUDGE_ENDPOINT=http://host:8000/v1/ python -m trig.metrics.trig_api
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = TRIGAPIMetric(dimension="TA-C", task="t2i")
    print(m.compute(os.path.join(here, "demo.jpg"), "A beautiful sunset with a beach", return_details=True))
