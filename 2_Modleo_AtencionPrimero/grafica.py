import pandas as pd
import matplotlib.pyplot as plt

# Cargar el log
# Asegúrate de que esta ruta apunte a tu archivo de log.
# Ejemplo: log_path = "experiments_logs/full_run_pro6000_.../train_log.csv"
log_path = "train_log.csv"

# Leer CSV.
df = pd.read_csv(log_path)

# Asegurarse de que todas las columnas numéricas sean tratadas como tales,
# convirtiendo los 'nan' de string a NaN de numpy y manejando errores.
numeric_cols = ['train_loss', 'val_loss', 'val_auroc_macro', 'val_auprc_macro', 'val_f1_macro', 'lr']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Crear una figura con 3 subplots verticales, compartiendo el eje X
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
fig.suptitle('Análisis Completo del Entrenamiento', fontsize=16)

# --- Subplot 1: Pérdidas (Loss) ---
ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", linestyle='-')
ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="x", linestyle='--')
ax1.set_ylabel("Loss")
ax1.set_title("Evolución de la Pérdida (Loss)")
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Subplot 2: Métricas de Validación ---
if 'val_auroc_macro' in df.columns:
    ax2.plot(df["epoch"], df["val_auroc_macro"], label="AUROC Macro", marker="s")
if 'val_auprc_macro' in df.columns:
    ax2.plot(df["epoch"], df["val_auprc_macro"], label="AUPRC Macro", marker="^")
if 'val_f1_macro' in df.columns:
    ax2.plot(df["epoch"], df["val_f1_macro"], label="F1 Macro", marker="d")
ax2.set_ylabel("Puntuación (Score)")
ax2.set_title("Métricas de Validación")
ax2.legend()
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.set_ylim(bottom=0, top=1.05) # Las métricas suelen estar entre 0 y 1

# --- Subplot 3: Tasa de Aprendizaje (Learning Rate) ---
if 'lr' in df.columns:
    ax3.plot(df["epoch"], df["lr"], label="Learning Rate", marker=".", color="purple")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.set_title("Decaimiento de la Tasa de Aprendizaje")
ax3.legend()
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

# Ajustar el layout y mostrar la gráfica
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Ajuste para el supertítulo
plt.show()
