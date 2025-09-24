## EDA de datasets ECG en esta carpeta (12Large y PTBXL)

Esta carpeta contiene herramientas para generar metadatos y ejecutar un EDA directamente aquí (no en `scripts/`).

### 1) Construir metadatos

Genera CSVs ligeros con información clave de cada registro para facilitar el EDA.

```bash
# 12Large
python dataWrangling/12Electro/build_metadata.py --dataset 12large --root RUTA_AL_12LARGE --out dataWrangling/12Electro/12large_meta.csv

# PTBXL (alta resolución 500 Hz)
python dataWrangling/12Electro/build_metadata.py --dataset ptbxl --root RUTA_AL_PTBXL --out dataWrangling/12Electro/ptbxl_meta.csv

# PTBXL (baja resolución 100 Hz)
python dataWrangling/12Electro/build_metadata.py --dataset ptbxl --root RUTA_AL_PTBXL --out dataWrangling/12Electro/ptbxl_meta_lr.csv --lowres
```

### 2) Ejecutar EDA sobre los CSVs de metadatos

Usa el CLI de EDA localizado en esta misma carpeta:

```bash
python dataWrangling/12Electro/eda_cli.py --input dataWrangling/12Electro/12large_meta.csv --file-type csv --target None --time-col None --corr spearman --max-categories 50 --id-cols patient_id

python dataWrangling/12Electro/eda_cli.py --input dataWrangling/12Electro/ptbxl_meta.csv --file-type csv --target None --time-col None --corr spearman --max-categories 50 --id-cols patient_id
```

Las salidas se guardan en `reports/eda/` con subcarpetas por timestamp.

### 3) EDA manual (opcional)

Abre `dataWrangling/12Electro/EDA_Plantilla.ipynb` y ajusta las rutas a los CSVs generados (o directamente a las carpetas de datasets si prefieres) para explorar.

### Dependencias

Usa el `requirements.txt` del repositorio raíz. Ya incluye `pandas`, `seaborn`, `matplotlib`, `pyarrow`, `plotly`, etc.

## EDA (Exploratory Data Analysis) para datos de ECG / tabulares

Este módulo provee un script CLI de EDA rápido para inspeccionar datasets CSV/Parquet: tamaños, tipos, nulos, estadísticas descriptivas, outliers, correlaciones y gráficos básicos.

### Requisitos

Instala las dependencias en un entorno virtual (Windows PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r scripts/eda/requirements.txt
```

### Uso

```bash
python scripts/eda/eda_cli.py --input RUTA_AL_ARCHIVO_O_CARPETA \
  --file-type auto \
  --sep , \
  --encoding utf-8 \
  --limit-rows 500000 \
  --glob "*.csv" \
  --target etiqueta \
  --time-col timestamp \
  --corr spearman \
  --max-categories 50 \
  --id-cols id_paciente,id_registro
```

- `--input`: archivo CSV/Parquet o carpeta con múltiples archivos homogéneos.
- `--file-type`: `auto|csv|parquet`. Si es `auto`, se infiere por extensión.
- `--limit-rows`: limita filas cargadas para proteger memoria.
- `--glob`: cuando `--input` es carpeta, filtra archivos (p. ej. `"*.csv"`).
- `--target`: nombre de la columna objetivo (si aplica) para balance de clases.
- `--time-col`: columna temporal (si aplica). Si no se indica, se intenta inferir.
- `--corr`: método de correlación `pearson` o `spearman`.
- `--max-categories`: top de categorías a resumir por columna.
- `--no-plots`: si se agrega, no genera gráficos.
 - `--id-cols`: columnas separadas por coma para detectar duplicados por identificador.

Los resultados se guardan en `reports/eda/YYYYMMDD_HHMMSS/`:

- `overview.csv` — filas, columnas y memoria.
- `schema_dtypes.csv` — dtypes, tipo inferido y cardinalidad.
- `na_summary.csv` — nulos por columna.
- `numeric_summary.csv` — estadísticas, sesgo, curtosis y outliers IQR.
- `value_counts/` — top categorías por columna categórica.
- `correlation_numeric.csv` — matriz de correlación (si aplica).
- `class_balance.csv` — balance de clases (si `--target`).
- `outliers/` — índices de outliers por IQR (muestra por columna).
- `duplicates_summary.json` — conteo de duplicados globales y por subset.
- `duplicates/` — muestras de duplicados (global y por subset si se indicó).
- `plots/` — histogramas, boxplots, barras, heatmap de correlación y nulos.

### Notas

- Para CSV grandes, usa `--limit-rows` para proteger memoria.
- Para Parquet, se carga cada archivo completo y luego se trunca con `--limit-rows` si se indicó.
- Las pruebas de normalidad usan D'Agostino sobre una muestra (máx. 50k valores) por columna.
- Si tus datos de ECG están en formato de series temporales por muestra (un registro por trazo), puedes usar `--time-col` para habilitar parsing de timestamps si existe. Si hay columna temporal, se generan gráficos por día (conteos y balance de clases si `--target`).


