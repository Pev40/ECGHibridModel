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

DATASETS=(12large ptbxl georgia)

check_dataset_ready() {
  local ds="$1"
  case "$ds" in
    12large)
      # Requiere .hea bajo datos/12Large/WFDBRecords
      if find datos/12Large/WFDBRecords -type f -name '*.hea' -print -quit 2>/dev/null | grep -q .; then
        return 0
      else
        echo "[train][SKIP] 12large: no hay .hea en datos/12Large/WFDBRecords" >&2
        return 1
      fi
      ;;
    ptbxl)
      # Requiere el CSV principal
      if [[ -f datos/PTBXL/ptbxl_database.csv ]]; then
        return 0
      else
        echo "[train][SKIP] ptbxl: falta datos/PTBXL/ptbxl_database.csv" >&2
        return 1
      fi
      ;;
    georgia)
      # Requiere .hea bajo datos/Georgia12LeadECGDatabase
      if find datos/Georgia12LeadECGDatabase -type f -name '*.hea' -print -quit 2>/dev/null | grep -q .; then
        return 0
      else
        echo "[train][SKIP] georgia: no hay .hea en datos/Georgia12LeadECGDatabase" >&2
        return 1
      fi
      ;;
    incart)
      # Requiere carpeta files con .hea
      if find datos/StPetersburgIncart12LeadArrhythmiaDatabase/files -type f -name '*.hea' -print -quit 2>/dev/null | grep -q .; then
        return 0
      else
        echo "[train][SKIP] incart: falta carpeta 'files' o .hea en datos/StPetersburgIncart12LeadArrhythmiaDatabase" >&2
        return 1
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

for ds in "${DATASETS[@]}"; do
  EXP_DIR="${EXP_ROOT}/${ds}"
  mkdir -p "${EXP_DIR}"
  echo "[train] Iniciando dataset=${ds} (logs en ${EXP_DIR})"

  # Flags específicos por dataset
  EXTRA_FLAGS=()
  case "${ds}" in
    incart)
      # Si la jerarquía de INCART no existe o tiene 0 fine codes, copiar fallback de 12Large
      INCART_HIER="datos/StPetersburgIncart12LeadArrhythmiaDatabase/labels_hierarchy.json"
      if [[ ! -f "${INCART_HIER}" ]]; then
        if [[ -f datos/12Large/labels_hierarchy.json ]]; then
          cp -f datos/12Large/labels_hierarchy.json "${INCART_HIER}" || true
          echo "[train] Copiada jerarquía fallback desde 12Large para INCART"
        fi
      else
        fine_len=$(python - <<'PY'
import json
try:
    with open('datos/StPetersburgIncart12LeadArrhythmiaDatabase/labels_hierarchy.json','r',encoding='utf-8') as f:
        d=json.load(f)
    print(len(d.get('fine_codes', [])))
except Exception:
    print(-1)
PY
)
        if [[ "${fine_len}" -le 0 && -f datos/12Large/labels_hierarchy.json ]]; then
          cp -f datos/12Large/labels_hierarchy.json "${INCART_HIER}" || true
          echo "[train] Reemplazada jerarquía vacía por fallback de 12Large para INCART"
        fi
      fi
      EXTRA_FLAGS+=(--no_sampler --no_auto_hierarchy)
      # Verificar si existen etiquetas positivas utilizables; si no, omitir entrenamiento supervisado
      HAS_LABELS=$(python - <<'PY'
import json, os, numpy as np
try:
    from datasets.incart import INCART12Lead
    hier_path = os.path.join('datos','StPetersburgIncart12LeadArrhythmiaDatabase','labels_hierarchy.json')
    root = os.path.join('datos','StPetersburgIncart12LeadArrhythmiaDatabase')
    ds = INCART12Lead(root, split='train', hierarchy_path=hier_path, eval_mode=True)
    s = 0.0
    for rec in ds.samples:
        y_fine = rec[2]
        if y_fine is not None:
            s += float(np.sum(y_fine))
    print(1 if s > 0 else 0)
except Exception:
    print(0)
PY
)
      if [[ "${HAS_LABELS}" != "1" ]]; then
        echo "[train][SKIP] incart: no hay etiquetas positivas utilizables (y_fine suma=0). Omitiendo entrenamiento supervisado para este dataset."
        continue
      fi
      ;;
  esac

  if check_dataset_ready "${ds}" && python train_full.py \
    --dataset "${ds}" \
    --sequence_len 5000 \
    --batch_size 256 \
    --workers 16 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --epochs 100 \
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
    --exp_dir "${EXP_DIR}" \
    "${EXTRA_FLAGS[@]}"; then
      echo "[train] Dataset ${ds} completado."
  else
      echo "[train][WARN] Dataset ${ds} falló. Continuando con el siguiente..." >&2
      continue
  fi
done

echo "[train] Finalizado. Revisa ${EXP_ROOT} para los logs."


