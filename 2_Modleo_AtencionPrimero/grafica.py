import pandas as pd
import matplotlib.pyplot as plt

# Cargar el log
log_path = "train.csv"

# Leer CSV ignorando los "nan"
df = pd.read_csv(log_path)

# Reemplazar "nan" string por NaN real de numpy
df = df.replace("nan", float("nan"))

# Convertir a float por si acaso quedaron strings
df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")

# Graficar
plt.figure(figsize=(10,6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Evolución del entrenamiento")
plt.legend()
plt.grid(True)
plt.show()
