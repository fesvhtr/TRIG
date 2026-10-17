#!/usr/bin/env bash
# End-to-end judge test on one H200: start vLLM, wait until healthy, score demo.jpg on all
# ten dimensions with scripts/test_judge.py, then shut the server down.
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
MODEL="${1:-Qwen/Qwen3.5-9B}"
PORT=8000
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
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

# Make sure the weights are complete before serving (resumes / waits on an in-progress download).
hf download "$MODEL" >/dev/null

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 2}' --served-model-name "$MODEL" \
  > "$REPO_DIR/slurm_logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then echo "[judge] server healthy after ${i}0s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[judge] vllm exited early"; tail -50 "$REPO_DIR/slurm_logs/vllm_${SLURM_JOB_ID}.log"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "[judge] server never became healthy"; exit 1; }

export TRIG_JUDGE_ENDPOINT="http://127.0.0.1:${PORT}/v1/"
export TRIG_JUDGE_MODEL="$MODEL"
echo "[judge] --- thinking disabled (default) ---"
python scripts/test_judge.py
echo "[judge] --- raw first token with thinking enabled, for comparison ---"
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from trig.metrics.trig_api import TRIGAPIMetric
m = TRIGAPIMetric(model_name=os.environ["TRIG_JUDGE_MODEL"], dimension="TA-C", enable_thinking=True)
m.task = "t2i"
print(m.compute("demo.jpg", "A beautiful sunset with a beach", return_details=True))
PY
echo "[judge] done"
