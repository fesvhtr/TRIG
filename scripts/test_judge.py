"""Score demo.jpg on all ten TRIG dimensions against a running judge and print the raw
rating-word probabilities, to verify the judge answers with a rating token (not <think>).

    TRIG_JUDGE_ENDPOINT=$(cat data/judge_endpoint.txt) python scripts/test_judge.py [image] [prompt]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trig.metrics.trig_api import TRIGAPIMetric  # noqa: E402
from trig.config import gpt_logit_dimension_msg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "demo.jpg")
prompt = sys.argv[2] if len(sys.argv) > 2 else "A beautiful sunset with a beach"
model = os.environ.get("TRIG_JUDGE_MODEL", "Qwen/Qwen3.5-9B")

for dim in gpt_logit_dimension_msg["t2i"]:
    metric = TRIGAPIMetric(model_name=model, dimension=dim)
    metric.task = "t2i"
    score, details = metric.compute(image, prompt, return_details=True)
    probs = {k: round(v, 3) for k, v in details.get("word_probs", {}).items()}
    print(f"{dim}: score={score:.3f} first_token={details.get('first_token')!r} probs={probs}")
