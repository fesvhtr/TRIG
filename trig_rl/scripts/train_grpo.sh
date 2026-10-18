#!/usr/bin/env bash
# One node, N GPUs: GPU 0 serves the Qwen3.5 judge (venv/llm), GPUs 1..N-1 train with Flow-Factory (venv/ff).
# Large multi-GPU job: ask before submitting.   Usage: sbatch --gres=gpu:4 trig_rl/scripts/train_grpo.sh [config.yaml]
#SBATCH --account=kuin0164
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=trig_grpo
#SBATCH --output=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.out
#SBATCH --error=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.err
set -euo pipefail
source /dpc/kuin0164/zsc/TRIG/scripts/slurm/lib.sh
CONFIG="${1:-trig_rl/configs/grpo_qwen_image_21_trig.yaml}"; JUDGE="${JUDGE:-Qwen/Qwen3.5-9B}"
NGPU=$(nvidia-smi -L | wc -l); PORT=$(free_port); LOG="slurm_logs/vllm_judge_${SLURM_JOB_ID}.log"
( trig_env llm; snapshot_dir "$JUDGE" >/dev/null; CUDA_VISIBLE_DEVICES=0 start_vllm_judge "$JUDGE" "$PORT" "$LOG"; wait "$SERVER_PID" ) &
JOB=$!; trap 'kill $JOB 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; kill -0 $JOB 2>/dev/null || { echo "judge died"; tail -40 "$LOG"; exit 1; }; sleep 10; done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "judge never healthy"; exit 1; }
export TRIG_JUDGE_ENDPOINT="http://127.0.0.1:${PORT}/v1/" TRIG_JUDGE_MODEL="$JUDGE"
trig_env ff
CUDA_VISIBLE_DEVICES="$(seq -s, 1 $((NGPU - 1)))" ff-train "$CONFIG"
