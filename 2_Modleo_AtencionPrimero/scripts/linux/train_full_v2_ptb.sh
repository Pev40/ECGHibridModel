#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash scripts/linux/setup_env.sh venv cu121
#   source venv/bin/activate
#   bash scripts/linux/train_full_v2.sh
#
# Override defaults via env, e.g.:
#   DATASET=ptbxl BATCH_SIZE=256 EPOCHS=80 bash scripts/linux/train_full_v2.sh

# Resolve repo root and ensure PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_DIR="${SCRIPT_DIR}/../.."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Experiment directory
EXP_DIR="experiments_logs/full_run_v2_$(date +%Y%m%d_%H%M%S)"

# Optional: cache conversion (ignore errors)
python scripts/cache_wfdb_to_pt.py || true

# Defaults (override with env vars)
SEQ_LEN=${SEQ_LEN:-5000}
BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-16}
LR=${LR:-5e-5}
WD=${WD:-1e-3}
EPOCHS=${EPOCHS:-150}
ACCUM=${ACCUM:-1}
DROPOUT=${DROPOUT:-0.4}
ATTN_DROPOUT=${ATTN_DROPOUT:-0.4}
TRANS_DROPOUT=${TRANS_DROPOUT:-0.2}
GAMMA_POS=${GAMMA_POS:-2.5}
GAMMA_NEG=${GAMMA_NEG:-6.0}
ASL_CLIP=${ASL_CLIP:-0.02}
SAMPLER_POWER=${SAMPLER_POWER:-0.3}
SAMPLER_POWER_RARE=${SAMPLER_POWER_RARE:-1.5}
RARE_THRESH=${RARE_THRESH:-0.01}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-25}
SCHED_METRIC=${SCHED_METRIC:-val_loss}
DATASET=${DATASET:-ptbxl}  # one of: 12large, ptbxl, georgia, incart

# Augmentations
AUG_JITTER_STD=${AUG_JITTER_STD:-0.01}
AUG_SHIFT_MAX=${AUG_SHIFT_MAX:-50}
AUG_LEAD_DROP=${AUG_LEAD_DROP:-0.05}
AUG_AMP_MIN=${AUG_AMP_MIN:-0.9}
AUG_AMP_MAX=${AUG_AMP_MAX:-1.1}
AUG_LEAD_NOISE_MAX=${AUG_LEAD_NOISE_MAX:-3.0}
AUG_TIME_WARP_MAX=${AUG_TIME_WARP_MAX:-0.01}
AUG_TIME_WARP_P=${AUG_TIME_WARP_P:-0.7}

# Mixup
MIXUP_ALPHA=${MIXUP_ALPHA:-0.2}
MIXUP_P=${MIXUP_P:-0.5}

# Notch and bandpass
BANDPASS_LOW=${BANDPASS_LOW:-0.5}
BANDPASS_HIGH=${BANDPASS_HIGH:-45.0}
TARGET_FS=${TARGET_FS:-500.0}
NOTCH_HZ=${NOTCH_HZ:-0}  # 50 or 60; 0 to disable
LABEL_SMOOTH=${LABEL_SMOOTH:-0.15}
FREQ_LOW=${FREQ_LOW:-1}
FREQ_HIGH=${FREQ_HIGH:-48}
SWIN_FREEZE=${SWIN_FREEZE:-0}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-15}

python train_full_v2.py \
  --sequence_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WD}" \
  --epochs "${EPOCHS}" \
  --accum_steps "${ACCUM}" \
  --mixed_precision \
  --dropout "${DROPOUT}" \
  --attn_dropout "${ATTN_DROPOUT}" \
  --trans_dropout "${TRANS_DROPOUT}" \
  --gamma_pos "${GAMMA_POS}" \
  --gamma_neg "${GAMMA_NEG}" \
  --asl_clip "${ASL_CLIP}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --label_smoothing "${LABEL_SMOOTH}" \
  --sampler_power "${SAMPLER_POWER}" \
  --sampler_power_rare "${SAMPLER_POWER_RARE}" \
  --rare_class_thresh "${RARE_THRESH}" \
  --aug_jitter_std "${AUG_JITTER_STD}" \
  --aug_shift_max "${AUG_SHIFT_MAX}" \
  --aug_lead_drop_prob "${AUG_LEAD_DROP}" \
  --aug_amp_scale_min "${AUG_AMP_MIN}" \
  --aug_amp_scale_max "${AUG_AMP_MAX}" \
  --aug_lead_noise_scale_max "${AUG_LEAD_NOISE_MAX}" \
  --aug_time_warp_max "${AUG_TIME_WARP_MAX}" \
  --aug_time_warp_p "${AUG_TIME_WARP_P}" \
  --mixup_alpha "${MIXUP_ALPHA}" \
  --mixup_p "${MIXUP_P}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --scheduler_metric "${SCHED_METRIC}" \
  --target_fs "${TARGET_FS}" \
  --bandpass_low "${BANDPASS_LOW}" \
  --bandpass_high "${BANDPASS_HIGH}" \
  $( [[ "${NOTCH_HZ}" == "50" || "${NOTCH_HZ}" == "60" ]] && echo --notch_hz "${NOTCH_HZ}" ) \
  --freq_bins_low "${FREQ_LOW}" \
  --freq_bins_high "${FREQ_HIGH}" \
  --swin_freeze_stages "${SWIN_FREEZE}" \
  --dataset "${DATASET}" \
  --cache_dir datos/pt_cache \
  --exp_dir "${EXP_DIR}"

echo "[train_v2] Logs in ${EXP_DIR}"


