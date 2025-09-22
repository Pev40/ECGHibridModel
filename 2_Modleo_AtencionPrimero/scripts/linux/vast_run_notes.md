## Vast.ai quick notes

### 1) Seleccionar instancia (ejemplo RTX PRO 6000 96GB)
- Filtra por VRAM >= 96GB, CUDA >= 12.9, fiabilidad alta.
- Apunta el ID de la instancia ofertada.

### 2) Lanzar y conectar por SSH
```bash
# login si no lo hiciste
vast login

# crear/rentar (ejemplo, ajusta --image según tu preferencia)
vast create instance <OFFER_ID> \
  --image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \
  --disk 200 --onstart-cmd "bash"

# listar instancias
vast show instances

# túnel/ssh
vast ssh <INSTANCE_ID>
```

### 3) Setup proyecto dentro de la instancia
```bash
git clone <TU_REPO>
cd <TU_REPO>
bash scripts/linux/setup_env.sh venv cu121
source venv/bin/activate
```

### 4) Entrenar y monitorear
```bash
bash scripts/linux/train_full.sh

# En otra terminal local:
vast ssh <INSTANCE_ID>
nvidia-smi -l 5
tail -f experiments_logs/full_run*/train_log.csv
```

### 5) Persistencia y checkpointing
- Los checkpoints y logs se guardan en `experiments_logs/` dentro de la instancia.
- Para descargar resultados:
```bash
vast scp <INSTANCE_ID>:~/<TU_REPO>/experiments_logs ./experiments_logs
```


