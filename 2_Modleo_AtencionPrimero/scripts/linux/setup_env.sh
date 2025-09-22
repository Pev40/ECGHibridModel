#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/linux/setup_env.sh [venv_name] [cu121|cu122|cpu]
# Defaults: venv_name=venv, torch channel=cu121

VENV_NAME="${1:-venv}"
TORCH_CHANNEL="${2:-cu129}"

echo "[setup] Creating venv: ${VENV_NAME}"
python3 -m venv "${VENV_NAME}"
source "${VENV_NAME}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

if [[ "${TORCH_CHANNEL}" != "cpu" ]]; then
  echo "[setup] Installing PyTorch with ${TORCH_CHANNEL}"
  python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
else
  echo "[setup] Installing CPU-only PyTorch"
  python -m pip install torch torchvision
fi

echo "[setup] Installing project requirements"
python -m pip install -r requirements.txt

python - << 'PY'
import torch
print('cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY

echo "[setup] Done. Activate with: source ${VENV_NAME}/bin/activate"


