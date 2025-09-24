# -*- coding: utf-8 -*-
"""
Script para el Análisis Exploratorio de Datos (EDA) y Limpieza de Datos
del dataset 12-lead electrocardiogram database.
"""

# --- 1. CONFIGURACIÓN E IMPORTACIÓN DE LIBRERÍAS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("--- Iniciando el análisis exploratorio del dataset de ECG ---")

# --- 2. CARGA E INSPECCIÓN INICIAL DE LOS DATOS ---
# Carga el archivo CSV en un DataFrame de pandas.
# Asegúrate de que '12large_meta.csv' esté en el mismo directorio.
try:
    df = pd.read_csv('12large_meta.csv')
    print("\n[PASO 1/4] Archivo CSV cargado exitosamente.")
except FileNotFoundError:
    print("\n[ERROR] No se encontró el archivo '12large_meta.csv'.")
    print("Por favor, asegúrate de que el archivo esté en el mismo directorio que este script.")
    exit()

# Muestra las primeras 5 filas para tener una idea de la estructura.
print("\n--- Primeras 5 filas del dataset ---")
print(df.head())

# Muestra información general: columnas, tipos de datos y valores no nulos.
print("\n--- Información general del DataFrame ---")
df.info()

# Muestra estadísticas descriptivas para las columnas numéricas.
print("\n--- Estadísticas descriptivas ---")
print(df.describe())


# --- 3. LIMPIEZA DE DATOS (DATA WRANGLING) ---
print("\n[PASO 2/4] Realizando limpieza y validación de datos...")

# Verificar si hay valores nulos en alguna columna.
missing_values = df.isnull().sum()
print("\n--- Conteo de valores nulos por columna ---")
print(missing_values)
if missing_values.sum() == 0:
    print("¡Excelente! No hay valores nulos en el dataset.")
else:
    print("Se encontraron valores nulos. Se recomienda tratarlos (eliminar o imputar).")

# Verificar si hay filas duplicadas.
duplicate_rows = df.duplicated().sum()
print(f"\n--- Número de filas duplicadas: {duplicate_rows} ---")
if duplicate_rows == 0:
    print("No se encontraron filas duplicadas.")
else:
    # Si se encuentran duplicados, se pueden eliminar con:
    # df = df.drop_duplicates()
    pass

# Verificar la consistencia de los datos.
# Por la descripción del dataset, esperamos que la mayoría de los ECGs
# tengan 12 derivaciones (leads) y una frecuencia de muestreo (fs) de 500 Hz.
print("\n--- Conteo de valores únicos en 'num_leads' ---")
print(df['num_leads'].value_counts())

print("\n--- Conteo de valores únicos en 'fs' (frecuencia de muestreo) ---")
print(df['fs'].value_counts())


# --- 4. ANÁLISIS EXPLORATORIO DE DATOS (EDA) Y VISUALIZACIÓN ---
print("\n[PASO 3/4] Creando visualizaciones para el análisis...")

# Configuración del estilo de los gráficos.
sns.set(style="whitegrid", palette="viridis")

# 1. Distribución del número de etiquetas por registro de ECG
plt.figure(figsize=(10, 6))
sns.countplot(x='num_labels', data=df)
plt.title('Distribución del Número de Diagnósticos (Etiquetas) por ECG', fontsize=16)
plt.xlabel('Cantidad de Etiquetas', fontsize=12)
plt.ylabel('Número de Registros de ECG', fontsize=12)
plt.savefig('distribucion_etiquetas.png')
print("\nGráfico 'distribucion_etiquetas.png' guardado.")

# 2. Análisis de las etiquetas de diagnóstico (la parte más importante)
# La columna 'labels' es un string con códigos separados por comas.
# Necesitamos separar estos códigos para analizarlos individualmente.
# Primero, eliminamos filas donde las etiquetas puedan ser nulas (si las hubiera).
df_labels = df.dropna(subset=['labels'])

# Creamos una lista que contenga TODAS las etiquetas de todas las filas.
all_labels = []
for index, row in df_labels.iterrows():
    labels_list = str(row['labels']).split(',')
    all_labels.extend(labels_list)

# Contamos la frecuencia de cada etiqueta.
label_counts = Counter(all_labels)

# Convertimos el contador a un DataFrame para facilitar la visualización.
df_label_counts = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])
df_label_counts = df_label_counts.sort_values(by='Count', ascending=False)

# Visualizamos las 20 etiquetas más comunes.
top_n = 20
plt.figure(figsize=(12, 10))
sns.barplot(x='Count', y='Label', data=df_label_counts.head(top_n), palette='viridis')
plt.title(f'Top {top_n} Diagnósticos (Etiquetas) Más Comunes', fontsize=16)
plt.xlabel('Frecuencia (Número de Ocurrencias)', fontsize=12)
plt.ylabel('Código de Diagnóstico (SNOMED CT)', fontsize=12)
plt.tight_layout()
plt.savefig('top_20_diagnosticos.png')
print("Gráfico 'top_20_diagnosticos.png' guardado.")

print("\n[PASO 4/4] Análisis finalizado.")
print("Se han guardado dos gráficos en el directorio: 'distribucion_etiquetas.png' y 'top_20_diagnosticos.png'.")

# Mostrar los gráficos si estás en un entorno que lo permita (como Jupyter).
plt.show()