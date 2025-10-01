#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and ensure PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_DIR="${SCRIPT_DIR}/../.."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Experiment directory
EXP_DIR="experiments_logs/full_run_v3_ddp_$(date +%Y%m%d_%H%M%S)"

# Optimized defaults for 2 GPUs
SEQ_LEN=${SEQ_LEN:-7500}
BATCH_SIZE=${BATCH_SIZE:-32}  # Per GPU
WORKERS=${WORKERS:-8}
LR=${LR:-5e-5}
WD=${WD:-1e-4}
EPOCHS=${EPOCHS:-120}
ACCUM=${ACCUM:-4}  # Effective batch size = 32 * 4 * 2 = 256
DROPOUT=${DROPOUT:-0.2}
HMST_D=${HMST_D:-128}
HMST_HEADS=${HMST_HEADS:-4}
HMST_LAYERS=${HMST_LAYERS:-4}
HMST_STAGES=${HMST_STAGES:-2}

# Filters
BANDPASS_LOW=${BANDPASS_LOW:-0.5}
BANDPASS_HIGH=${BANDPASS_HIGH:-45.0}
TARGET_FS=${TARGET_FS:-500.0}
NOTCH_HZ=${NOTCH_HZ:-0}

echo "=== CONFIGURACIÓN DDP 2 GPUs ==="
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * ACCUM * 2))"
echo "d_model: ${HMST_D}"
echo "Heads: ${HMST_HEADS}"
echo "Layers: ${HMST_LAYERS}"
echo "Stages: ${HMST_STAGES}"
echo "Sequence len: ${SEQ_LEN}"
echo "================================"

# Launch with torchrun
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 \
  train_full_v3_ddp.py \
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

echo "[train_v3_ddp] Logs en ${EXP_DIR}"