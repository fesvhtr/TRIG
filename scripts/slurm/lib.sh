#!/usr/bin/env bash
# Shared pieces for the Slurm scripts: cluster environment, judge servers, health wait.
# Usage: source "$(dirname "$0")/lib.sh"   (REPO_DIR is set here)
REPO_DIR=/dpc/kuin0164/zsc/TRIG
VENVS=/dpc/kuin0164/zsc/venv

trig_env() {  # trig_env <venv name: llm | t2i | jev | ff>
  module load intel_h200_gpu; module load cuda/13.0; module load gcc/14.1.0
  source "$VENVS/$1/bin/activate"
  export HF_HOME=/dpc/kuin0164/zsc/hf_home HF_HUB_CACHE=/dpc/kuin0164/zsc/hf_home/hub HF_HUB_DISABLE_XET=1
  export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # Compile caches must live on node-local disk: Triton's atomic-rename writes race on /dpc (Lustre).
  local cache="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}/trig_${SLURM_JOB_ID:-local}"; mkdir -p "$cache"
  export VLLM_CACHE_ROOT="$cache/vllm" TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
  cd "$REPO_DIR"
}

free_port() { echo $((20000 + ${SLURM_JOB_ID:-$$} % 20000)); }   # unique per job; jobs may share a node

snapshot_dir() {  # snapshot_dir <repo id> -> local snapshot path (downloads only if missing)
  python - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    print(snapshot_download(sys.argv[1], local_files_only=True))
except Exception:
    print(snapshot_download(sys.argv[1]))
PY
}

start_vllm_judge() {  # start_vllm_judge <model> <port> <log>  (sets SERVER_PID)
  vllm serve "$1" --host 127.0.0.1 --port "$2" --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 2}' --served-model-name "$1" > "$3" 2>&1 &
  SERVER_PID=$!
}

start_jev() {  # start_jev <model> <port> <log>  (JEV's own vLLM server with /v1/decide; sets SERVER_PID)
  local dir; dir="$(snapshot_dir "$1")"
  python "$dir/serve_decide.py" --model "$dir" --served-model-name "$1" \
    --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$dir/adapter_vllm" \
    --logprobs-mode processed_logprobs --max-model-len 32768 --enable-prefix-caching --mamba-cache-mode align \
    --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 8 --trust-request-chat-template \
    --gpu-memory-utilization 0.90 --host 127.0.0.1 --port "$2" > "$3" 2>&1 &
  SERVER_PID=$!
}

wait_healthy() {  # wait_healthy <port> <pid> <log> [max_tries=180]
  for _ in $(seq 1 "${4:-180}"); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || { echo "server exited early:"; tail -60 "$3"; return 1; }
    sleep 10
  done
  echo "server never became healthy"; tail -60 "$3"; return 1
}

stop_server() { kill "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; sleep 5; }
