## Organización por dataset

- 12Large
  - Datos: `datos/12Large/WFDBRecords`
  - Jerarquía: `datos/12Large/labels_hierarchy.json`
  - Entrenar: `python train_full.py --dataset 12large --exp_dir experiments_logs`

- PTBXL
  - Datos: `datos/PTBXL/records500` y `ptbxl_database.csv`
  - Generar jerarquía: `python scripts/generate_ptbxl_labels_hierarchy.py --root datos/PTBXL`
  - Entrenar: `python train_full.py --dataset ptbxl --exp_dir experiments_logs`

- Georgia12LeadECGDatabase
  - Datos: `datos/Georgia12LeadECGDatabase/*.hea|.mat`
  - Usa jerarquía de 12Large por defecto
  - Entrenar: `python train_full.py --dataset georgia --exp_dir experiments_logs`

- StPetersburg INCART
  - Datos: `datos/StPetersburgIncart12LeadArrhythmiaDatabase/files`
  - Usa jerarquía de 12Large por defecto (no hay SNOMED en headers)
  - Entrenar: `python train_full.py --dataset incart --exp_dir experiments_logs`

EDA por dataset:
```bash
python scripts/eda_by_dataset.py --dataset 12large --outdir eda_outputs
python scripts/eda_by_dataset.py --dataset ptbxl   --outdir eda_outputs
python scripts/eda_by_dataset.py --dataset georgia --outdir eda_outputs
python scripts/eda_by_dataset.py --dataset incart  --outdir eda_outputs
```

Resultados y métricas por dataset se guardan en `experiments_logs/<dataset>/`.


