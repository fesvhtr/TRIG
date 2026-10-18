#!/usr/bin/env bash
# Offline DPO needs no judge at train time. Large job: ask before submitting.
# Usage: sbatch --gres=gpu:2 trig_rl/scripts/train_offline_dpo.sh [config.yaml]
#SBATCH --account=kuin0164
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=trig_dpo
#SBATCH --output=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.out
#SBATCH --error=/dpc/kuin0164/zsc/TRIG/slurm_logs/%x_%j.err
set -euo pipefail
source /dpc/kuin0164/zsc/TRIG/scripts/slurm/lib.sh
trig_env ff
ff-train "${1:-trig_rl/configs/offline_dpo_qwen_image_21_trig.yaml}"
