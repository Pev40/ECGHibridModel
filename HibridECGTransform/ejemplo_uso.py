"""
Ejemplo de uso del modelo ECGTransForm con Atención Variable Temporal

Este archivo muestra cómo usar el modelo actualizado con las nuevas características
de atención variable temporal para mejorar la detección de ECG.
"""

import torch
import numpy as np
from models import ecgTransForm
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class

def ejemplo_uso_modelo():
    """
    Ejemplo de cómo usar el modelo ECGTransForm con atención variable temporal
    """
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Obtener configuraciones para el dataset MIT
    dataset_configs = get_dataset_class('mit')()
    hparams_class = get_hparams_class('supervised')()
    hparams = hparams_class.train_params
    
    print("Configuraciones del modelo:")
    print(f"- Número de clases: {dataset_configs.num_classes}")
    print(f"- Longitud de secuencia: {dataset_configs.sequence_len}")
    print(f"- Dimensiones de embedding: {dataset_configs.embedding_dims}")
    print(f"- Número de cabezas de atención: {dataset_configs.num_heads}")
    print(f"- Número de bloques transformer: {dataset_configs.num_transformer_blocks}")
    
    # Crear el modelo
    modelo = ecgTransForm(configs=dataset_configs, hparams=hparams)
    modelo.to(device)
    
    # Crear datos de ejemplo (batch_size=4, channels=1, length=186)
    batch_size = 4
    datos_ejemplo = torch.randn(batch_size, dataset_configs.input_channels, dataset_configs.sequence_len)
    datos_ejemplo = datos_ejemplo.to(device)
    
    print(f"\nForma de datos de entrada: {datos_ejemplo.shape}")
    
    # Evaluar modelo sin visualizar atención
    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(datos_ejemplo, use_attn=False)
        print(f"Forma de predicciones: {predicciones.shape}")
        print(f"Predicciones (logits): {predicciones}")
        
        # Convertir a probabilidades
        probabilidades = torch.softmax(predicciones, dim=1)
        print(f"Probabilidades: {probabilidades}")
        
        # Obtener clases predichas
        clases_predichas = torch.argmax(probabilidades, dim=1)
        print(f"Clases predichas: {clases_predichas}")
        print(f"Nombres de clases: {[dataset_configs.class_names[i] for i in clases_predichas]}")

    # Evaluar modelo CON visualización de atención
    print("\n=== ANÁLISIS DE ATENCIÓN ===")
    with torch.no_grad():
        predicciones, pesos_atencion = modelo(datos_ejemplo, use_attn=True)
        
        variable_attn_weights, temporal_attn_weights = pesos_atencion
        
        print(f"Número de capas de atención: {len(variable_attn_weights)}")
        print(f"Forma de pesos de atención variable: {variable_attn_weights[0].shape if variable_attn_weights[0] is not None else 'None'}")
        print(f"Forma de pesos de atención temporal: {temporal_attn_weights[0].shape if temporal_attn_weights[0] is not None else 'None'}")
        
        # Analizar la atención promedio por capa
        if variable_attn_weights[0] is not None:
            for i, pesos in enumerate(variable_attn_weights):
                atencion_promedio = pesos.mean().item()
                print(f"Atención variable promedio capa {i+1}: {atencion_promedio:.4f}")
                
        if temporal_attn_weights[0] is not None:
            for i, pesos in enumerate(temporal_attn_weights):
                atencion_promedio = pesos.mean().item()
                print(f"Atención temporal promedio capa {i+1}: {atencion_promedio:.4f}")

def comparar_arquitecturas():
    """
    Compara la nueva arquitectura con atención variable temporal vs la original
    """
    print("\n=== COMPARACIÓN DE ARQUITECTURAS ===")
    
    # Configuraciones
    dataset_configs = get_dataset_class('mit')()
    hparams_class = get_hparams_class('supervised')()
    hparams = hparams_class.train_params
    
    # Crear modelo
    modelo = ecgTransForm(configs=dataset_configs, hparams=hparams)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Mostrar componentes principales
    print("\nComponentes del modelo:")
    print("1. Convoluciones multi-escala (original)")
    print("2. Módulo de recalibración de canales - SE blocks (original)")
    print("3. ✨ NUEVO: Convoluciones causales multi-escala")
    print("4. ✨ NUEVO: Embeddings posicionales avanzados")
    print("5. ✨ NUEVO: Atención Variable Temporal (Variable + Temporal)")
    print("6. Clasificador final (original)")
    
    print("\nBeneficios de la atención variable temporal:")
    print("- Captura dependencias tanto entre variables como en el tiempo")
    print("- Utiliza convoluciones causales para preservar orden temporal")
    print("- Embeddings más sofisticados para mejor representación")
    print("- Doble atención: variable (entre características) y temporal (entre timesteps)")

if __name__ == "__main__":
    print("=== EJEMPLO DE USO: ECGTransForm con Atención Variable Temporal ===")
    ejemplo_uso_modelo()
    comparar_arquitecturas()
    print("\n¡Modelo actualizado exitosamente! 🎉") 