"""Generate several seeds per prompt with one model alias (any _DiffusersT2IModel subclass).
Output <out>/<model>/<data_id>_s<seed>.png; existing files are skipped, so it resumes.

    python scripts/diag/generate_seeds.py --model qwen_image_21 --seeds 0 1 2 3 --shard 0/4 --batch-size 8
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trig.models import import_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen_image_21")
    ap.add_argument("--prompts", default="data/diag/prompts.jsonl")
    ap.add_argument("--out", default="data/diag/images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th prompt starting at i")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int); ap.add_argument("--steps", type=int)
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    records = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()][i::n]
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    todo = [(r, s) for r in records for s in args.seeds if not os.path.exists(os.path.join(out_dir, f"{r['data_id']}_s{s}.png"))]
    print(f"[gen] model={args.model} shard={args.shard} prompts={len(records)} todo={len(todo)}", flush=True)
    if not todo:
        return

    cls = import_model(args.model)
    if args.width: cls.width = args.width
    if args.height: cls.height = args.height
    if args.steps: cls.call_kwargs = {**cls.call_kwargs, "num_inference_steps": args.steps}
    model = cls()

    t0 = time.time()
    for b in range(0, len(todo), args.batch_size):
        chunk = todo[b:b + args.batch_size]
        try:
            images = model.generate([r["prompt"] for r, _ in chunk], seeds=[s for _, s in chunk])
            for (r, s), img in zip(chunk, images):
                path = os.path.join(out_dir, f"{r['data_id']}_s{s}.png")
                img.save(path + ".partial.png")
                os.replace(path + ".partial.png", path)
        except Exception as e:  # keep the shard alive
            print(f"[gen] FAILED {', '.join(f'{r['data_id']}:s{s}' for r, s in chunk)}: {type(e).__name__}: {e}", flush=True)
        done = b + len(chunk)
        if (b // args.batch_size) % 10 == 9 or done == len(todo):
            print(f"[gen] {done}/{len(todo)} done, {(time.time() - t0) / done:.1f}s per image (batch {args.batch_size})", flush=True)


if __name__ == "__main__":
    main()
