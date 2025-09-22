#!/usr/bin/env bash
set -euo pipefail

# Example usage (CUDA 12.1 build):
#   bash scripts/linux/setup_env.sh venv cu121
#   source venv/bin/activate
#   bash scripts/linux/train_full.sh

EXP_DIR="experiments_logs/full_run_pro6000_$(date +%Y%m%d_%H%M%S)"

python scripts/cache_wfdb_to_pt.py || true

python train_full.py \
  --sequence_len 5000 \
  --batch_size 256 \
  --workers 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --epochs 30 \
  --accum_steps 1 \
  --mixed_precision \
  --cache_dir datos/pt_cache \
  --exp_dir "${EXP_DIR}"

echo "[train] Logs in ${EXP_DIR}"


