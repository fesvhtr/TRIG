"""Score diagnostic images with a System-One decision model served by JEV's serve_decide.py
(POST /v1/decide, e.g. autotrust/JEV-27B-VL). Same output schema as judge_protocols.py.

Protocols: jev_choice (five TRIG rating words as options; expectation over [0, .25, .5, .75, 1]),
jev_score (JEV's 0-5 ordinal head, E[level]/5); add the suffix _v1 to hide the prompt for the
image-quality dimensions, as in Qwen protocol v1.

    TRIG_JEV_ENDPOINT=http://127.0.0.1:8000 python scripts/diag/judge_jev.py --protocols jev_choice jev_score
"""
import argparse
import os
import sys
from functools import lru_cache

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.diag._common import add_common_args, base_row, build_jobs, index_images, load_done, load_prompts, output_path, run_jobs  # noqa: E402
from trig.config import gpt_logit_dimension_msg  # noqa: E402
from trig.judge import IMAGE_ONLY_DIMS, RATING_WEIGHTS, RATING_WORDS  # noqa: E402
from trig.utils.utils import image_data_url  # noqa: E402

PROTOCOLS = ("jev_choice", "jev_score", "jev_choice_v1", "jev_score_v1")
RATING_DESC = {
    "terrible": "terrible (many errors or clearly fails this dimension)",
    "bad": "bad (some errors)",
    "medium": "medium (acceptable)",
    "good": "good (very good performance)",
    "excellent": "excellent (accurate, nothing to fault on this dimension)",
}


def build_request(protocol, dim, prompt, data_url):
    kind, v1 = protocol.replace("_v1", "").replace("jev_", ""), protocol.endswith("_v1")
    if v1 and dim in IMAGE_ONLY_DIMS:
        state = [{"image": data_url}, "This is an AI generated image. The prompt is deliberately withheld; judge the image itself."]
    else:
        state = [{"image": data_url}, f"This is an AI generated image. Prompt used to generate it: {prompt}"]
    question = ("Evaluate this generated image on ONE dimension only, using the definition and criteria below. "
                "Ignore other qualities of the image.\n" + gpt_logit_dimension_msg["t2i"][dim] +
                "\nHow does the image perform on this dimension?")
    if kind == "choice":
        return {"kind": "choice", "state": state, "question": question, "options": [RATING_DESC[w] for w in RATING_WORDS]}
    return {"kind": "score", "state": state, "question": question + " Give 0 for terrible and 5 for excellent."}


def to_score(protocol, probabilities):
    if "choice" in protocol:
        return sum(p * RATING_WEIGHTS[w] for w, p in zip(RATING_WORDS, probabilities)), dict(zip(RATING_WORDS, [round(x, 4) for x in probabilities]))
    return sum(p * i for i, p in enumerate(probabilities)) / 5.0, {str(i): round(x, 4) for i, x in enumerate(probabilities)}


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_workers=8)
    ap.add_argument("--protocols", nargs="+", default=["jev_choice"], choices=PROTOCOLS)
    ap.add_argument("--endpoint", default=os.environ.get("TRIG_JEV_ENDPOINT", "http://127.0.0.1:8000"))
    ap.add_argument("--max-side", type=int, default=448, help="JEV recommends ~448 px for latency")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    records = load_prompts(args.prompts, args.shard)
    images = index_images(os.path.join(args.images, args.model))
    session = requests.Session()
    url = args.endpoint.rstrip("/") + "/v1/decide"
    data_url = lru_cache(maxsize=256)(lambda path: image_data_url(path, max_side=args.max_side))

    for protocol in args.protocols:
        out = output_path(args, protocol)
        jobs = build_jobs(records, images, load_done(out))
        print(f"[jev] protocol={protocol} jobs={len(jobs)}", flush=True)
        if not jobs:
            continue

        def score(job):
            row = base_row(job, protocol)
            try:
                req = build_request(protocol, job["dim"], job["record"]["prompt"], data_url(job["path"]))
                resp = session.post(url, json=req, timeout=args.timeout)
                resp.raise_for_status()
                body = resp.json()
                s, probs = to_score(protocol, body["probabilities"])
                row.update({"score": round(s, 4), "word_probs": probs, "choice": body.get("choice"), "elapsed": body.get("elapsed_seconds")})
            except Exception as e:
                row.update({"score": None, "error": f"{type(e).__name__}: {e}"})
            return row

        run_jobs(jobs, score, args.workers, out)
        print(f"[jev] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
