#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash scripts/linux/setup_env.sh venv cu121   # opcional
#   source venv/bin/activate                     # opcional
#   bash scripts/linux/train_all_datasets.sh

# Resolver root del repo y PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_DIR="${SCRIPT_DIR}/../.."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="experiments_logs/all_datasets_${TIMESTAMP}"
mkdir -p "${EXP_ROOT}" datos/pt_cache || true

# Opcional: pre-cachear (si el script existe)
python scripts/cache_wfdb_to_pt.py || true

DATASETS=(12large ptbxl georgia incart)

for ds in "${DATASETS[@]}"; do
  EXP_DIR="${EXP_ROOT}/${ds}"
  mkdir -p "${EXP_DIR}"
  echo "[train] Iniciando dataset=${ds} (logs en ${EXP_DIR})"

  if python train_full.py \
    --dataset "${ds}" \
    --sequence_len 10000 \
    --batch_size 256 \
    --workers 16 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --epochs 2 \
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
    --exp_dir "${EXP_DIR}"; then
      echo "[train] Dataset ${ds} completado."
  else
      echo "[train][WARN] Dataset ${ds} falló. Continuando con el siguiente..." >&2
      continue
  fi
done

echo "[train] Finalizado. Revisa ${EXP_ROOT} para los logs."


