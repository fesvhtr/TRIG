#!/usr/bin/env bash
# One allocation, one GPU, up to four stages in sequence, each in its own environment:
#   1. generate images (venv/t2i)    2. Qwen3.5 TRIGScore, chosen protocols (venv/llm)
#   3. JEV-27B-VL decision model (venv/jev)    4. analyse (venv/t2i)
# Usage: sbatch [--array=0-N] scripts/slurm/diag_all_in_one.sh <gen_alias> <prompts.jsonl> [seeds...]
#   A Slurm array shards the prompt file across tasks (every N-th prompt); score files get a .shardIofN suffix.
#   env: DIAG_BATCH=8  DIAG_PROTOCOLS="baseline v1 v2"  JEV_PROTOCOLS="jev_choice jev_score"
#        SKIP_GEN=1 / SKIP_QWEN=1 / SKIP_JEV=1 to skip stages. Every stage is resumable.
#SBATCH --account=kuin0164
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=trig_diag_all
#SBATCH --output=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%A_%a.out
#SBATCH --error=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%A_%a.err
set -euo pipefail
source /dpc/kuin0164/zsc/TRIG/scripts/slurm/lib.sh
GEN="${1:-qwen_image_21}"; PROMPTS="${2:-data/diag/prompts_sub.jsonl}"; shift 2 || true
SEEDS=("$@"); [[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0 1 2 3)
JUDGE="${JUDGE:-Qwen/Qwen3.5-9B}"; JEV_MODEL="${JEV_MODEL:-autotrust/JEV-27B-VL}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
stage() { echo; echo "================ [$(date +%H:%M:%S)] $* ================"; }

if [[ "${SKIP_GEN:-0}" != "1" ]]; then
  stage "generate $GEN prompts=$PROMPTS shard=$SHARD seeds=${SEEDS[*]} batch=${DIAG_BATCH:-8}"
  ( trig_env t2i
    python scripts/diag/generate_seeds.py --model "$GEN" --prompts "$PROMPTS" --seeds "${SEEDS[@]}" --shard "$SHARD" --batch-size "${DIAG_BATCH:-8}" )
fi
if [[ "${SKIP_QWEN:-0}" != "1" ]]; then
  stage "judge $JUDGE protocols=${DIAG_PROTOCOLS:-baseline v1}"
  ( trig_env llm; PORT=$(free_port); LOG="slurm_logs/vllm_qwen_${SLURM_JOB_ID}.log"
    snapshot_dir "$JUDGE" >/dev/null
    start_vllm_judge "$JUDGE" "$PORT" "$LOG"; trap 'stop_server $SERVER_PID' EXIT
    wait_healthy "$PORT" "$SERVER_PID" "$LOG"
    TRIG_JUDGE_ENDPOINT="http://127.0.0.1:${PORT}/v1/" TRIG_JUDGE_MODEL="$JUDGE" \
      python scripts/diag/judge_protocols.py --model "$GEN" --prompts "$PROMPTS" --shard "$SHARD" --protocols ${DIAG_PROTOCOLS:-baseline v1} --workers 32 )
fi
if [[ "${SKIP_JEV:-0}" != "1" ]]; then
  stage "judge $JEV_MODEL protocols=${JEV_PROTOCOLS:-jev_choice jev_score}"
  ( trig_env jev; PORT=$(free_port); LOG="slurm_logs/jev_${SLURM_JOB_ID}.log"
    start_jev "$JEV_MODEL" "$PORT" "$LOG"; trap 'stop_server $SERVER_PID' EXIT
    wait_healthy "$PORT" "$SERVER_PID" "$LOG" 240
    TRIG_JEV_ENDPOINT="http://127.0.0.1:${PORT}" \
      python scripts/diag/judge_jev.py --model "$GEN" --prompts "$PROMPTS" --shard "$SHARD" --protocols ${JEV_PROTOCOLS:-jev_choice jev_score} --workers 8 )
fi
stage "analyze (partial while other shards run)"
( trig_env t2i; python scripts/diag/analyze.py --model "$GEN" ) || true
