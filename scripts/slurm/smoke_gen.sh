#!/usr/bin/env bash
# Generate one image per model alias with the TRIG prompt set, as a smoke test on one H200.
# Usage: sbatch scripts/slurm/smoke_gen.sh zimage_base flux2_klein_base_9b qwen_image_21 ...
#SBATCH --account=kuin0164
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --job-name=trig_smoke_gen
#SBATCH --output=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.out
#SBATCH --error=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.err
set -euo pipefail
REPO_DIR=/dpc/kuin0164/zsc/TRIG
module load intel_h200_gpu
module load cuda/13.0
module load gcc/14.1.0
source /dpc/kuin0164/zsc/venv/t2i/bin/activate
export HF_HOME=/dpc/kuin0164/zsc/hf_home
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO_DIR"
python scripts/smoke_gen.py "$@"
