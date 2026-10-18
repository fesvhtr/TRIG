#!/usr/bin/env bash
# Serve the TRIGScore judge (default Qwen/Qwen3.5-9B) with vLLM on one H200 for the job's lifetime.
# Usage: sbatch scripts/slurm/serve_judge.sh [HF_MODEL_ID]   -> export TRIG_JUDGE_ENDPOINT=$(cat data/judge_endpoint.txt)
#SBATCH --account=kuin0164
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=trig_judge
#SBATCH --output=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.out
#SBATCH --error=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.err
set -euo pipefail
source /dpc/kuin0164/zsc/TRIG/scripts/slurm/lib.sh
trig_env llm
MODEL="${1:-Qwen/Qwen3.5-9B}"; PORT=$(free_port)
echo "http://$(hostname -f):${PORT}/v1/" > data/judge_endpoint.txt
echo "[judge] model=$MODEL endpoint=$(cat data/judge_endpoint.txt)"
snapshot_dir "$MODEL" >/dev/null
exec vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 2}' --served-model-name "$MODEL"
