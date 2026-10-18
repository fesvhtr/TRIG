# trig_rl — optimising generators with TRIG's multi-dimensional judge

Everything for the "use the eval results to train the model" stage lives here. The trainers are
**not** written from scratch: we build on [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
(vendored as a git submodule in `third_party/Flow-Factory`, Apache-2.0), which already supports
Qwen-Image-2.1 / Z-Image, Flow-GRPO/DanceGRPO-style GRPO, online and offline Diffusion-DPO, LoRA,
DeepSpeed/FSDP2, and custom reward classes. Our code is the thin TRIG-specific layer.

## Layout

| Path | What |
|---|---|
| `rewards/relations.py` | `RELATIONS` registry: per relation type the reward scalar and the DPO pairing rule; Pareto fronts / rewards; DTM loader |
| `rewards/trig_judge.py` | Flow-Factory reward classes: `TRIGDimReward` (one dim), `TRIGPairReward` (relation scalar), `TRIGParetoGroupReward` (Pareto-GRPO, groupwise) |
| `relations/build_dtm.py` | Scores JSONL → `dtm.json` via the shared classifier `trig/evaluation/relation_rules.py` |
| `data/make_grpo_dataset.py` | TRIG prompts → `data/grpo/{train,test}.jsonl`; `--dtm` stamps `relation`/`dominant` per row (rewards read these, no DTM at train time) |
| `data/make_dpo_pairs.py` | Diagnostic images + scores → offline-DPO V2 manifest with relation-aware Pareto pairing |
| `configs/*.yaml` | GRPO and offline-DPO recipes for Qwen-Image-2.1 (LoRA) |
| `scripts/*.sh` | Slurm launchers (judge on GPU 0 + training on the rest). Large jobs: ask before submitting |
| `tests/test_relations.py` | Offline tests for the rules |

## Environment

`/dpc/kuin0164/zsc/venv/ff` (Python 3.12, torch 2.13 cu130, Flow-Factory + its pinned diffusers
submodule, transformers 5.20). Run everything from the TRIG repo root with
`PYTHONPATH=/dpc/kuin0164/zsc/TRIG` so that both `trig` and `trig_rl` import. Judge server:
`venv/llm` (vLLM 0.26) via `scripts/slurm/serve_judge.sh` or inline in `trig_rl/scripts/train_grpo.sh`.

## Pipeline

The judge itself (scale, prompts, client) lives in `trig/judge.py` and is shared with the evaluation scripts.

1. **Scores → DTM**: `python trig_rl/relations/build_dtm.py --scores data/diag/scores/qwen_image_21__v2*.jsonl --out trig_rl/data/dtm.json`
1b. **GRPO data**: `python trig_rl/data/make_grpo_dataset.py --dtm trig_rl/data/dtm.json --out trig_rl/data/grpo`
2. **Offline DPO data**: `python trig_rl/data/make_dpo_pairs.py --scores ... --images data/diag/images/qwen_image_21 --prompts data/diag/prompts_sub.jsonl --dtm trig_rl/data/dtm.json --out trig_rl/data/dpo_pairs`
   (needs ≥2 seeds per prompt; the 196-prompt subset has 4, the full set has 1 → generate more seeds for training prompts first)
3. **Offline DPO**: `ff-train trig_rl/configs/offline_dpo_qwen_image_21_trig.yaml`
4. **GRPO** (warm-start from the DPO LoRA via `model.resume_path`): `ff-train trig_rl/configs/grpo_qwen_image_21_trig.yaml`
   Reward variants are switched in the `rewards:` block: Pareto-GRPO (default), two per-dimension rewards with `gdpo` (Multi-GRPO baseline), or the relation scalar.
5. **Evaluate**: regenerate the TRIG test prompts with the LoRA and score all dimensions with `scripts/diag/judge_protocols.py`; compare DTMs with `build_dtm.py`.

## Dimensions

Seven dimensions, 21 pairs: IQ-R, IQ-O, IQ-A, TA-C, TA-R, TA-S, D-K (`trig.config_judge_v2.DIMS`). R-T, R-B and D-A from the paper are excluded everywhere (`EXCLUDED_DIMS`); every loader filters on it.

## Relation rules (from the DTM)

| relation | reward scalar (`combine`) | DPO pair rule (on top of Pareto dominance by ≥ eps) |
|---|---|---|
| synergy | mean | sum improves by ≥ 2 eps, no dim worse |
| bottleneck | 0.7·min + 0.3·mean | winner's min ≥ loser's min + eps |
| tilt (D dominant) | 0.7·weak + 0.3·D − penalty·max(0, floor − D) | weak dim ≥ +eps, dominant not worse |
| dispersion | mean (Pareto rank does the work) | strict dominance with margin 2 eps |
| undefined | mean | plain dominance |

Upstream commit used: see `git submodule status` (Flow-Factory `fb96d991`, 2026-10-18).
