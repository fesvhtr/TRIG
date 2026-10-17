#!/usr/bin/env bash
# Serve the TRIGScore judge (default Qwen/Qwen3.5-9B) with vLLM on one H200.
# Usage: sbatch scripts/slurm/serve_judge.sh [HF_MODEL_ID] [PORT]
# The endpoint URL is written to data/judge_endpoint.txt; export it as TRIG_JUDGE_ENDPOINT.
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
MODEL="${1:-Qwen/Qwen3.5-9B}"
PORT="${2:-8000}"
REPO_DIR=/dpc/kuin0164/zsc/TRIG

module load intel_h200_gpu
module load cuda/13.0
module load gcc/14.1.0
source /dpc/kuin0164/zsc/venv/llm/bin/activate
export HF_HOME=/dpc/kuin0164/zsc/hf_home
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
# Compile caches must live on node-local disk: Triton's atomic-rename writes race on /dpc (Lustre).
LOCAL_CACHE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}/vllm_${SLURM_JOB_ID:-local}"
mkdir -p "$LOCAL_CACHE"
export VLLM_CACHE_ROOT="$LOCAL_CACHE/vllm"
export TRITON_CACHE_DIR="$LOCAL_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE/inductor"

ENDPOINT="http://$(hostname -f):${PORT}/v1/"
mkdir -p "$REPO_DIR/data"
echo "$ENDPOINT" > "$REPO_DIR/data/judge_endpoint.txt"
echo "[judge] model=$MODEL endpoint=$ENDPOINT"

# Thinking is switched off per request by TRIGAPIMetric (chat_template_kwargs), so no
# reasoning parser is needed. Two images per prompt covers the p2p/s2p variants.
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 2}' \
  --served-model-name "$MODEL"
