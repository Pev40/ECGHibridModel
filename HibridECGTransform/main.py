import os
import argparse
import warnings
from trainer import trainer
import sklearn.exceptions
import torch
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()

# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='Exp1',   type=str, help='experiment name')
parser.add_argument('--run_description',        default='run1',     type=str, help='run name')

# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='mit',           type=str, help='mit, ptb')
parser.add_argument('--seed_id',                default='0',             type=str, help='to fix a seed while training')

# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'data',           type=str,   help='Path containing dataset')

parser.add_argument('--num_runs',               default=1,                 type=int,   help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',            type=str,   help='cpu or cuda')

# ========= Optimización de GPU ===============
parser.add_argument('--auto_batch_size',        default=False,             type=bool,  help='Encontrar batch size óptimo automáticamente')
parser.add_argument('--enable_monitoring',      default=False,             type=bool,  help='Habilitar monitoreo de GPU en tiempo real')

args = parser.parse_args()

def optimize_gpu_settings():
    """Optimiza configuraciones de GPU para mejor rendimiento"""
    print("🔧 Optimizando configuraciones de GPU...")
    
    if torch.cuda.is_available():
        # Configuraciones de optimización
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Configurar asignador de memoria si está disponible
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.9)  # Usar 90% de la memoria GPU
            
        # Variables de entorno para optimización - compatibles con Windows
        try:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print("   ✅ Configuración CUDA expandable_segments activada")
        except:
            print("   ⚠️ No se pudo configurar expandable_segments")
            
        # Configuraciones específicas para Windows
        import platform
        if platform.system() == "Windows":
            print("   🪟 Detectado Windows - aplicando configuraciones específicas")
            # Desactivar algunas optimizaciones problemáticas en Windows
            os.environ['TORCH_COMPILE_DISABLE'] = '0'  # Permitir compile pero con fallback
            
        print("✅ Configuraciones de GPU optimizadas")
    else:
        print("⚠️ CUDA no disponible, usando CPU")

def find_optimal_batch_size(dataset_name='mit', max_batch_size=128):
    """Encuentra el batch size óptimo para el hardware disponible"""
    print("🔍 Buscando batch size óptimo...")
    
    from configs.data_configs import get_dataset_class
    from configs.hparams import get_hparams_class
    from models import ecgTransForm
    
    dataset_configs = get_dataset_class(dataset_name)()
    hparams_class = get_hparams_class('supervised')()
    hparams = hparams_class.train_params
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Probar diferentes batch sizes
    batch_sizes = [16, 32, 48, 64, 80, 96, 112, 128]
    optimal_batch_size = 32  # Valor por defecto
    
    for batch_size in batch_sizes:
        if batch_size > max_batch_size:
            break
            
        try:
            print(f"   Probando batch_size = {batch_size}...")
            
            # Limpiar memoria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Crear modelo de prueba
            temp_hparams = hparams.copy()
            temp_hparams['batch_size'] = batch_size
            
            model = ecgTransForm(configs=dataset_configs, hparams=temp_hparams)
            model.to(device)
            
            # Crear datos de prueba
            test_data = torch.randn(batch_size, dataset_configs.input_channels, dataset_configs.sequence_len).to(device)
            
            # Forward pass de prueba
            model.eval()
            with torch.no_grad():
                output = model(test_data)
            
            optimal_batch_size = batch_size
            print(f"   ✅ Batch size {batch_size} funciona")
            
            # Limpiar memoria
            del model, test_data, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   ❌ Batch size {batch_size} causa error de memoria")
                break
            else:
                print(f"   ❌ Error con batch size {batch_size}: {e}")
                break
        except Exception as e:
            print(f"   ❌ Error inesperado: {e}")
            break
    
    print(f"🎯 Batch size óptimo encontrado: {optimal_batch_size}")
    return optimal_batch_size

def start_gpu_monitoring():
    """Inicia el monitoreo de GPU en segundo plano"""
    try:
        from monitor_gpu import GPUMonitor
        monitor = GPUMonitor(interval=5.0)
        monitor.start_monitoring()
        print("📊 Monitor de GPU iniciado")
        return monitor
    except ImportError:
        print("⚠️ No se pudo importar GPUMonitor. Monitoreo deshabilitado.")
        return None

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento optimizado de ECGTransForm")
    print("=" * 50)
    
    # Optimizar configuraciones de GPU
    optimize_gpu_settings()
    
    # Encontrar batch size óptimo si está habilitado
    if args.auto_batch_size:
        optimal_batch_size = find_optimal_batch_size(args.dataset)
        print(f"💡 Recomendación: Actualizar batch_size en configs/hparams.py a {optimal_batch_size}")
    
    # Iniciar monitoreo si está habilitado
    monitor = None
    if args.enable_monitoring:
        monitor = start_gpu_monitoring()
    
    try:
        # Crear y ejecutar entrenador
        trainer_instance = trainer(args)
        trainer_instance.train()
        
        print("✅ Entrenamiento completado exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        raise
    
    finally:
        # Detener monitoreo si está activo
        if monitor:
            monitor.stop_monitoring()
            monitor.generate_report()
            
        print("🏁 Proceso finalizado")
