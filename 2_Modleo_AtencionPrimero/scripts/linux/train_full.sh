#!/usr/bin/env bash
set -euo pipefail

# Example usage (CUDA 12.1 build):
#   bash scripts/linux/setup_env.sh venv cu121
#   source venv/bin/activate
#   bash scripts/linux/train_full.sh

# Resolve repo root and ensure PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_DIR="${SCRIPT_DIR}/../.."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

EXP_DIR="experiments_logs/full_run_pro6000_$(date +%Y%m%d_%H%M%S)"

python scripts/cache_wfdb_to_pt.py || true

# Ensure labels hierarchy exists from real headers (top-K configurable via env TOPK)
if [[ ! -f datos/labels_hierarchy.json ]]; then
  python scripts/generate_labels_hierarchy.py --top_k "${TOPK:-30}"
fi

python train_full.py \
  --sequence_len 1000 \
  --batch_size 256 \
  --workers 16 \
  --lr 1e-3 \
  --weight_decay 5e-4 \
  --epochs 60 \
  --accum_steps 1 \
  --mixed_precision \
  --dropout 0.35 \
  --early_stopping_patience 8 \
  --early_stopping_min_delta 5e-4 \
  --gamma_pos 2.0 \
  --gamma_neg 4.0 \
  --asl_clip 0.0 \
  --sampler_power 0.3 \
  --aug_jitter_std 0.01 \
  --aug_shift_max 50 \
  --aug_lead_drop_prob 0.05 \
  --aug_amp_scale_min 0.9 \
  --aug_amp_scale_max 1.1 \
  --cache_dir datos/pt_cache \
  --exp_dir "${EXP_DIR}"

echo "[train] Logs in ${EXP_DIR}"


