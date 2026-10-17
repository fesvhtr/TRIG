"""Instantiate each model alias, generate one image for a fixed TRIG-style prompt, report
wall time and peak GPU memory. Images land in data/output/smoke/<alias>.png.

    python scripts/smoke_gen.py zimage_base flux2_klein_base_9b qwen_image_21
"""
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trig.models import import_model  # noqa: E402

PROMPT = ("A detailed watercolor painting of a red fox reading a newspaper on a park bench "
          "under autumn maple trees, soft morning light.")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "output", "smoke")


def main(aliases):
    os.makedirs(OUT_DIR, exist_ok=True)
    report = []
    for alias in aliases:
        print(f"\n===== {alias} =====", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            cls = import_model(alias)
            repo = getattr(cls, "model_id", None)
            if repo:
                # Resume / wait for any in-progress download so the job never races the cache.
                from huggingface_hub import snapshot_download
                snapshot_download(repo)
            model = cls()
            t_load = time.time() - t0
            t1 = time.time()
            image = model.generate(PROMPT)
            t_gen = time.time() - t1
            path = os.path.join(OUT_DIR, f"{alias}.png")
            image.save(path)
            peak = torch.cuda.max_memory_allocated() / 2**30
            report.append((alias, "ok", f"load {t_load:.0f}s gen {t_gen:.1f}s peak {peak:.1f}GiB"))
            print(f"[{alias}] saved {path} | {report[-1][2]}", flush=True)
            del model
        except Exception as e:  # keep going so one failure does not hide the others
            traceback.print_exc()
            report.append((alias, "FAILED", f"{type(e).__name__}: {e}"))
        torch.cuda.empty_cache()
    print("\n===== summary =====")
    for row in report:
        print(" | ".join(row))


if __name__ == "__main__":
    main(sys.argv[1:] or ["zimage_base"])
