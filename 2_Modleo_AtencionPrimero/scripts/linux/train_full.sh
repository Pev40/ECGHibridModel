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

# Ensure labels hierarchy exists
if [[ ! -f datos/labels_hierarchy.json ]]; then
  python - << 'PY'
import json,os
os.makedirs('datos', exist_ok=True)
hier={"fine_codes":["426177001","426783006","164890007","427084000","164934002","55827005","55930002","59931005","427393009","164889003","429622005","39732003","284470004","10370003","428750005","270492004","713427006","427172004","164917005","251146004","47665007","164930006","698252002","426761007","61721007","59118001","164873001","365413008","111975006","6374002"],"coarse_groups":{"rhythm":["426177001","426783006","427084000","164890007","164889003","426761007","713422000","713427006","427172004"],"conduction":["270492004","195042002","27885002","59118001","164909002","698252002","426995002","251164006","74390002"],"st_t_ischemia_mi":["429622005","164930006","164934002","59931005","164917005","164865005","111975006"],"hypertrophy":["164873001","89792004","446358003"],"ectopy":["284470004","17338001","75532003"],"axis_rotation_voltage":["39732003","47665007","251146004","251198002","251199005","365413008"],"other":["55827005","55930002","10370003","61721007","6374002","428417006"]}}
with open('datos/labels_hierarchy.json','w',encoding='utf-8') as f: json.dump(hier,f,ensure_ascii=False,indent=2)
print('Created datos/labels_hierarchy.json')
PY
fi

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


