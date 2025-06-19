"""
Script para verificar el uso de memoria antes del entrenamiento completo
"""

import torch
import gc
from models import ecgTransForm
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class

def limpiar_memoria():
    """Limpia la memoria CUDA"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def verificar_memoria_gpu():
    """Verifica el estado actual de la memoria GPU"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9
        free_memory = total_memory - allocated_memory
        
        print(f"=== ESTADO DE MEMORIA GPU ===")
        print(f"Memoria total: {total_memory:.2f} GB")
        print(f"Memoria reservada: {reserved_memory:.2f} GB")
        print(f"Memoria asignada: {allocated_memory:.2f} GB")
        print(f"Memoria libre: {free_memory:.2f} GB")
        return free_memory > 1.0  # Necesitamos al menos 1GB libre
    else:
        print("CUDA no disponible")
        return False

def probar_modelo_memoria(dataset='ptb', batch_size=32):
    """
    Prueba el modelo con diferentes tamaños de batch para verificar memoria
    """
    print(f"\n=== PROBANDO MODELO CON DATASET {dataset.upper()} ===")
    
    # Configuraciones
    dataset_configs = get_dataset_class(dataset)()
    hparams_class = get_hparams_class('supervised')()
    hparams = hparams_class.train_params
    
    # Modificar batch size para prueba
    hparams['batch_size'] = batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    try:
        # Limpiar memoria antes de comenzar
        limpiar_memoria()
        
        # Verificar memoria inicial
        if not verificar_memoria_gpu():
            print("⚠️ Advertencia: Poca memoria disponible")
        
        # Crear modelo
        print("\n1. Creando modelo...")
        modelo = ecgTransForm(configs=dataset_configs, hparams=hparams)
        modelo.to(device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in modelo.parameters())
        print(f"   Parámetros totales: {total_params:,}")
        
        # Verificar memoria después de cargar modelo
        print("\n2. Memoria después de cargar modelo:")
        verificar_memoria_gpu()
        
        # Crear datos de prueba
        print(f"\n3. Probando con batch_size={batch_size}...")
        datos_prueba = torch.randn(batch_size, dataset_configs.input_channels, dataset_configs.sequence_len)
        datos_prueba = datos_prueba.to(device)
        
        # Forward pass
        modelo.eval()
        with torch.no_grad():
            predicciones = modelo(datos_prueba, use_attn=False)
            print(f"   ✅ Forward pass exitoso!")
            print(f"   Forma de salida: {predicciones.shape}")
        
        # Verificar memoria después del forward pass
        print("\n4. Memoria después de forward pass:")
        verificar_memoria_gpu()
        
        # Probar con atención
        print(f"\n5. Probando con análisis de atención...")
        with torch.no_grad():
            predicciones, pesos_atencion = modelo(datos_prueba, use_attn=True)
            print(f"   ✅ Forward pass con atención exitoso!")
        
        # Verificar memoria final
        print("\n6. Memoria final:")
        verificar_memoria_gpu()
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ ERROR DE MEMORIA con batch_size={batch_size}")
            print(f"Error: {e}")
            limpiar_memoria()
            return False
        else:
            print(f"\n❌ ERROR: {e}")
            return False
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        return False

def encontrar_batch_size_optimo(dataset='ptb'):
    """
    Encuentra el batch size óptimo para el hardware disponible
    """
    print(f"\n=== BUSCANDO BATCH SIZE ÓPTIMO PARA {dataset.upper()} ===")
    
    batch_sizes = [16, 32, 48, 64, 80, 96, 112, 128]
    batch_optimo = 16
    
    for batch_size in batch_sizes:
        print(f"\n--- Probando batch_size = {batch_size} ---")
        limpiar_memoria()
        
        if probar_modelo_memoria(dataset, batch_size):
            batch_optimo = batch_size
            print(f"✅ Batch size {batch_size} funciona correctamente")
        else:
            print(f"❌ Batch size {batch_size} falla por memoria")
            break
    
    print(f"\n🎯 BATCH SIZE ÓPTIMO: {batch_optimo}")
    return batch_optimo

def recomendaciones_optimizacion():
    """
    Proporciona recomendaciones para optimizar memoria
    """
    print("\n=== RECOMENDACIONES PARA OPTIMIZAR MEMORIA ===")
    print("1. 🔧 Parámetros ya optimizados:")
    print("   - embedding_dims: 256 → 128")
    print("   - num_transformer_blocks: 3 → 2") 
    print("   - batch_size: 128 → 64")
    print("\n2. 📋 Opciones adicionales si aún hay problemas:")
    print("   - Reducir batch_size a 32 o 16")
    print("   - Usar precisión mixta (fp16)")
    print("   - Activar PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("\n3. 💡 Comandos recomendados:")
    print("   # GPU con poca memoria (6GB):")
    print("   python main.py --dataset ptb --device cuda:0")
    print("   ")
    print("   # Si persisten los problemas:")
    print("   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("   python main.py --dataset ptb --device cuda:0")

if __name__ == "__main__":
    print("🔍 VERIFICACIÓN DE MEMORIA PARA ECGTransForm")
    print("=" * 50)
    
    # Limpiar memoria inicial
    limpiar_memoria()
    
    # Verificar estado inicial
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible. Usando CPU.")
        exit()
    
    # Probar con configuración actual
    print("\n📊 PRUEBA CON CONFIGURACIÓN ACTUAL")
    dataset = 'ptb'
    if probar_modelo_memoria(dataset, 64):
        print(f"\n✅ ¡El modelo debería funcionar con la configuración actual!")
    else:
        print(f"\n⚠️ Problemas de memoria detectados. Buscando configuración óptima...")
        encontrar_batch_size_optimo(dataset)
    
    # Mostrar recomendaciones
    recomendaciones_optimizacion()
    
    print("\n🎉 Verificación completada!") 