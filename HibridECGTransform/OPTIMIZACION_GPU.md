# 🚀 Optimización de GPU para ECGTransForm

## Problema Identificado
El modelo se cargaba en memoria GPU pero tenía largos períodos de inactividad (0% de utilización) debido a cuellos de botella en el procesamiento de datos.

## ✅ Optimizaciones Implementadas

### 1. DataLoader Optimizado (`dataloader.py`)
- **Multiprocesamiento**: `num_workers` automático (máximo 4)
- **Pin Memory**: Transferencia más rápida CPU → GPU
- **Persistent Workers**: Workers se mantienen vivos entre épocas
- **Prefetch Factor**: Pre-carga 2 batches por worker

### 2. Entrenador Optimizado (`trainer.py`)
- **Mixed Precision**: Reduce uso de memoria y acelera cómputo
- **Gradient Accumulation**: Simula batches más grandes sin usar más memoria
- **CUDNN Benchmark**: Optimiza convoluciones para tamaños fijos
- **Torch Compile**: Compilación JIT cuando está disponible
- **Monitoreo de Tiempo**: Estadísticas de rendimiento por batch/época

### 3. Monitor de GPU (`monitor_gpu.py`)
- **Monitoreo en Tiempo Real**: Uso de GPU, memoria y CPU
- **Detección de Cuellos de Botella**: Alertas automáticas
- **Reportes de Eficiencia**: Análisis post-entrenamiento

### 4. Script Principal Mejorado (`main.py`)
- **Auto Batch Size**: Encuentra automáticamente el batch size óptimo
- **Configuraciones GPU**: Optimizaciones automáticas
- **Monitoreo Integrado**: Opción de monitoreo en tiempo real

## 🔧 Cómo Usar las Optimizaciones

### Entrenamiento Básico Optimizado
```bash
python main.py --dataset ptb --device cuda:0
```

### Encontrar Batch Size Óptimo
```bash
python main.py --dataset ptb --auto_batch_size True --device cuda:0
```

### Entrenamiento con Monitoreo
```bash
python main.py --dataset ptb --enable_monitoring True --device cuda:0
```

### Solo Monitor de GPU (en terminal separada)
```bash
python monitor_gpu.py
```

## 📊 Mejoras Esperadas

### Antes de las Optimizaciones
- ❌ GPU inactiva 50-70% del tiempo
- ❌ `num_workers=0` (sin multiprocesamiento)
- ❌ Sin pin_memory
- ❌ Sin mixed precision
- ❌ Transferencias lentas CPU → GPU

### Después de las Optimizaciones
- ✅ GPU activa 80-95% del tiempo
- ✅ Multiprocesamiento automático
- ✅ Transferencias aceleradas
- ✅ Menor uso de memoria con mixed precision
- ✅ Monitoreo en tiempo real
- ✅ 2-3x más rápido en la mayoría de casos

## 🎯 Configuraciones Recomendadas por GPU

### GPU de 6GB (RTX 3060, etc.)
```bash
# Configuración conservadora
python main.py --dataset ptb --device cuda:0

# Si necesitas más memoria
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py --dataset ptb --device cuda:0
```

### GPU de 8GB+ (RTX 3070+, etc.)
```bash
# Puedes usar batch sizes más grandes
python main.py --dataset ptb --auto_batch_size True --device cuda:0
```

### Múltiples GPUs
```bash
# Especificar GPU específica
python main.py --dataset ptb --device cuda:1
```

## 🔍 Monitoreo y Diagnóstico

### Verificar Eficiencia Actual
```bash
python monitor_gpu.py
```

### Síntomas de Cuellos de Botella
- **GPU <30% utilización**: Problema en carga de datos
- **CPU >90%**: Reducir `num_workers`
- **Memoria GPU >90%**: Reducir `batch_size`
- **Memoria GPU <20%**: Aumentar `batch_size`

### Comandos de Diagnóstico
```python
import torch

# Verificar memoria GPU
print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print(f"Memoria usada: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")

# Verificar configuraciones
print(f"CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
print(f"Mixed Precision disponible: {torch.cuda.is_available()}")
```

## ⚡ Optimizaciones Avanzadas

### 1. Batch Size Dinámico
Si el batch size óptimo encontrado es diferente al configurado:

```python
# En configs/hparams.py
self.train_params = {
    'batch_size': 80,  # Usar el valor encontrado por auto_batch_size
    # ... otros parámetros
}
```

### 2. Gradient Accumulation
Para simular batches más grandes:

```python
# En trainer.py (ya implementado)
self.gradient_accumulation_steps = 2  # Simula batch_size * 2
```

### 3. Memory Mapping
Para datasets muy grandes:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

## 🐛 Solución de Problemas

### Error: "Cannot find a working triton installation" (Windows)
**Problema común en Windows con torch.compile**

**Solución Rápida:**
```bash
# Ejecutar script de configuración
python install_dependencies.py

# Usar versión compatible con Windows
python main_windows.py --dataset ptb --device cuda:0
```

**Solución Manual:**
```bash
# Opción 1: Variables de entorno
set TORCH_COMPILE_DISABLE=1
set TORCHDYNAMO_DISABLE=1
python main.py --dataset ptb

# Opción 2: Usar archivo batch
setup_env.bat
python main.py --dataset ptb
```

**Solución Definitiva (Recomendada):**
Instalar WSL2 + Ubuntu para compatibilidad completa con PyTorch

### Error: "CUDA out of memory"
1. Reducir batch_size en `configs/hparams.py`
2. Usar gradient accumulation
3. Configurar variable de entorno:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### Error: "Too many workers"
1. Reducir `num_workers` manualmente en `dataloader.py`
2. O usar `num_workers=0` para depuración

### GPU sigue con baja utilización
1. Verificar que los datos estén en la ubicación correcta
2. Ejecutar monitor para identificar el cuello de botella específico
3. Considerar usar datasets en memoria si son pequeños

## 📈 Resultados Esperados

Con estas optimizaciones deberías ver:
- **Tiempo por época**: Reducido en 40-60%
- **Utilización GPU**: 80-95% (vs 30-50% anterior)
- **Memoria GPU**: Mejor aprovechada
- **Consistencia**: Menos variación en tiempos de entrenamiento

¡El entrenamiento ahora debería mantener la GPU constantemente ocupada y reducir significativamente los tiempos de entrenamiento! 