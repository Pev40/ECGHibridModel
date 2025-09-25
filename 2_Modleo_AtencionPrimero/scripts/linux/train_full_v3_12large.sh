#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and ensure PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_DIR="${SCRIPT_DIR}/../.."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Experiment directory
EXP_DIR="experiments_logs/full_run_v3_$(date +%Y%m%d_%H%M%S)"

# Defaults (override with env vars)
SEQ_LEN=${SEQ_LEN:-5000}
BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-16}
LR=${LR:-5e-5}
WD=${WD:-1e-4}
EPOCHS=${EPOCHS:-120}
ACCUM=${ACCUM:-1}
DROPOUT=${DROPOUT:-0.2}
HMST_D=${HMST_D:-256}
HMST_HEADS=${HMST_HEADS:-8}
HMST_LAYERS=${HMST_LAYERS:-6}
HMST_STAGES=${HMST_STAGES:-3}

# Filters
BANDPASS_LOW=${BANDPASS_LOW:-0.5}
BANDPASS_HIGH=${BANDPASS_HIGH:-45.0}
TARGET_FS=${TARGET_FS:-500.0}
NOTCH_HZ=${NOTCH_HZ:-0}

python train_full_v3.py \
  --sequence_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WD}" \
  --epochs "${EPOCHS}" \
  --accum_steps "${ACCUM}" \
  --mixed_precision \
  --dropout "${DROPOUT}" \
  --target_fs "${TARGET_FS}" \
  --bandpass_low "${BANDPASS_LOW}" \
  --bandpass_high "${BANDPASS_HIGH}" \
  $( [[ "${NOTCH_HZ}" == "50" || "${NOTCH_HZ}" == "60" ]] && echo --notch_hz "${NOTCH_HZ}" ) \
  --hmst_d_model "${HMST_D}" \
  --hmst_heads "${HMST_HEADS}" \
  --hmst_layers "${HMST_LAYERS}" \
  --hmst_stages "${HMST_STAGES}" \
  --exp_dir "${EXP_DIR}"

echo "[train_v3] Logs en ${EXP_DIR}"


