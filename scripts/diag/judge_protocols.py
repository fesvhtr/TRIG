"""Score diagnostic images on their two dimensions with the TRIGScore judge under one or more
protocols (baseline / v1 / v2, see trig/judge.py). Output: one JSON line per (image, dim) in
data/diag/scores/<model>__<protocol>[.shardIofN].jsonl; resumable.

    TRIG_JUDGE_ENDPOINT=http://host:8000/v1/ python scripts/diag/judge_protocols.py \
        --model qwen_image_21 --protocols baseline v1 v2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.diag._common import add_common_args, base_row, build_jobs, index_images, load_done, load_prompts, output_path, run_jobs  # noqa: E402
from trig.judge import PROTOCOLS, JudgeClient  # noqa: E402
from trig.utils.utils import image_data_url  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, default_workers=16)
    ap.add_argument("--protocols", nargs="+", default=["baseline", "v1"], choices=PROTOCOLS)
    ap.add_argument("--judge", default=None, help="judge model (default: TRIG_JUDGE_MODEL env)")
    ap.add_argument("--endpoint", default=None, help="judge endpoint (default: TRIG_JUDGE_ENDPOINT env)")
    args = ap.parse_args()

    records = load_prompts(args.prompts, args.shard)
    images = index_images(os.path.join(args.images, args.model))
    for protocol in args.protocols:
        out = output_path(args, protocol)
        jobs = build_jobs(records, images, load_done(out))
        print(f"[judge] protocol={protocol} jobs={len(jobs)}", flush=True)
        if not jobs:
            continue
        judge = JudgeClient(args.endpoint, args.judge, protocol=protocol, max_concurrent=args.workers)

        def score(job):
            row = base_row(job, protocol)
            try:
                s, details = judge.score_one(image_data_url(job["path"]), job["record"]["prompt"], job["dim"], return_details=True)
                row.update({"score": round(s, 4), "first_token": details["first_token"],
                            "word_probs": {k: round(v, 4) for k, v in details["word_probs"].items()}})
            except Exception as e:
                row.update({"score": None, "error": str(e)})
            return row

        run_jobs(jobs, score, args.workers, out)
        print(f"[judge] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
