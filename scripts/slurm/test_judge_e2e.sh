#!/usr/bin/env bash
# One-GPU end-to-end judge check: serve, score demo.jpg on all dimensions (scripts/test_judge.py), stop.
# Usage: sbatch scripts/slurm/test_judge_e2e.sh [HF_MODEL_ID]
#SBATCH --account=kuin0164
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=trig_judge_test
#SBATCH --output=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.out
#SBATCH --error=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.err
set -euo pipefail
source /dpc/kuin0164/zsc/TRIG/scripts/slurm/lib.sh
trig_env llm
MODEL="${1:-Qwen/Qwen3.5-9B}"; PORT=$(free_port)
snapshot_dir "$MODEL" >/dev/null
start_vllm_judge "$MODEL" "$PORT" "slurm_logs/vllm_${SLURM_JOB_ID}.log"; trap 'stop_server $SERVER_PID' EXIT
wait_healthy "$PORT" "$SERVER_PID" "slurm_logs/vllm_${SLURM_JOB_ID}.log"
TRIG_JUDGE_ENDPOINT="http://127.0.0.1:${PORT}/v1/" TRIG_JUDGE_MODEL="$MODEL" python scripts/test_judge.py
