#!/usr/bin/env bash
set -euo pipefail

# Script para ejecutar múltiples corridas del modelo HMST v3 con diferentes semillas
# Usage: bash scripts/linux/multi_run_v3.sh

# Resolve repo root and ensure PYTHONPATH
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
REPO_DIR="${SCRIPT_DIR}/../.."
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Configuración de las corridas
SEEDS=(123 456 789 999 2025)  # Semillas a ejecutar (ya tienes 42)
BASE_EXP_DIR="experiments_logs/multi_run_v3_$(date +%Y%m%d_%H%M%S)"

# Parámetros por defecto (puedes modificar aquí)
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

# Filtros
BANDPASS_LOW=${BANDPASS_LOW:-0.5}
BANDPASS_HIGH=${BANDPASS_HIGH:-45.0}
TARGET_FS=${TARGET_FS:-500.0}
NOTCH_HZ=${NOTCH_HZ:-0}

echo "=== Iniciando múltiples corridas del modelo HMST v3 ==="
echo "Semillas a ejecutar: ${SEEDS[*]}"
echo "Directorio base: ${BASE_EXP_DIR}"
echo "Épocas por corrida: ${EPOCHS}"
echo ""

# Crear directorio base
mkdir -p "${BASE_EXP_DIR}"

# Archivo para recopilar resultados
SUMMARY_FILE="${BASE_EXP_DIR}/summary_all_runs.csv"
echo "seed,exp_dir,auroc_macro,auprc_macro,f1_macro,auroc_micro,auprc_micro" > "${SUMMARY_FILE}"

# Ejecutar cada corrida
for seed in "${SEEDS[@]}"; do
    echo "--- Ejecutando corrida con semilla: ${seed} ---"
    
    # Directorio específico para esta semilla
    SEED_EXP_DIR="${BASE_EXP_DIR}/seed_${seed}"
    
    # Ejecutar entrenamiento
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
      --seed "${seed}" \
      --exp_dir "${SEED_EXP_DIR}"
    
    # Extraer métricas del test_metrics.json si existe
    if [[ -f "${SEED_EXP_DIR}/test_metrics.json" ]]; then
        echo "Extrayendo métricas para semilla ${seed}..."
        
        # Usar python para extraer métricas del JSON
        python -c "
import json
import sys

try:
    with open('${SEED_EXP_DIR}/test_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    auroc_macro = metrics.get('auroc_macro', 'nan')
    auprc_macro = metrics.get('auprc_macro', 'nan')
    f1_macro = metrics.get('f1_macro', 'nan')
    auroc_micro = metrics.get('auroc_micro', 'nan')
    auprc_micro = metrics.get('auprc_micro', 'nan')
    
    # Escribir al archivo de resumen
    with open('${SUMMARY_FILE}', 'a') as f:
        f.write(f'${seed},${SEED_EXP_DIR},{auroc_macro},{auprc_macro},{f1_macro},{auroc_micro},{auprc_micro}\n')
    
    print(f'Semilla ${seed} - AUROC: {auroc_macro:.4f}, AUPRC: {auprc_macro:.4f}, F1: {f1_macro:.4f}')
except Exception as e:
    print(f'Error extrayendo métricas para semilla ${seed}: {e}')
    with open('${SUMMARY_FILE}', 'a') as f:
        f.write(f'${seed},${SEED_EXP_DIR},nan,nan,nan,nan,nan\n')
"
    else
        echo "No se encontró test_metrics.json para semilla ${seed}"
        echo "${seed},${SEED_EXP_DIR},nan,nan,nan,nan,nan" >> "${SUMMARY_FILE}"
    fi
    
    echo "Corrida con semilla ${seed} completada."
    echo ""
done

echo "=== Todas las corridas completadas ==="
echo "Resultados guardados en: ${BASE_EXP_DIR}"
echo "Resumen de métricas en: ${SUMMARY_FILE}"

# Generar estadísticas finales
echo "--- Calculando estadísticas finales ---"
python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${SUMMARY_FILE}')
    df = df.dropna()  # Remover corridas fallidas
    
    if len(df) > 0:
        metrics = ['auroc_macro', 'auprc_macro', 'f1_macro', 'auroc_micro', 'auprc_micro']
        
        print('\\n=== ESTADÍSTICAS FINALES ===')
        print(f'Corridas exitosas: {len(df)}/{len(df)}')
        print('\\nResultados por métrica:')
        
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                min_val = df[metric].min()
                max_val = df[metric].max()
                print(f'{metric:12s}: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})')
        
        # Guardar estadísticas
        stats = df[metrics].describe()
        stats.to_csv('${BASE_EXP_DIR}/statistics_summary.csv')
        print(f'\\nEstadísticas detalladas guardadas en: ${BASE_EXP_DIR}/statistics_summary.csv')
        
        # Mostrar tabla completa
        print('\\n=== TABLA COMPLETA DE RESULTADOS ===')
        print(df.to_string(index=False, float_format='%.4f'))
    else:
        print('No se pudieron extraer métricas de ninguna corrida.')
        
except Exception as e:
    print(f'Error calculando estadísticas: {e}')
"

echo ""
echo "Proceso completado. Revisa los directorios individuales para logs detallados."
