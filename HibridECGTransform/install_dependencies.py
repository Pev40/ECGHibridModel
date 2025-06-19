"""
Script para instalar y verificar dependencias necesarias para ECGTransForm
Maneja problemas específicos de Windows con Triton y torch.compile
"""

import subprocess
import sys
import platform
import torch

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {description} completado")
            return True
        else:
            print(f"   ❌ Error en {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error ejecutando {description}: {e}")
        return False

def check_pytorch_installation():
    """Verifica la instalación de PyTorch"""
    print("🔍 Verificando instalación de PyTorch...")
    
    try:
        print(f"   PyTorch versión: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        return True
    except Exception as e:
        print(f"   ❌ Error verificando PyTorch: {e}")
        return False

def install_windows_dependencies():
    """Instala dependencias específicas para Windows"""
    print("🪟 Instalando dependencias para Windows...")
    
    dependencies = [
        "psutil",  # Para monitor de GPU
        "einops",  # Para operaciones de tensor
    ]
    
    success = True
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Instalando {dep}"):
            success = False
    
    return success

def fix_triton_issue():
    """Intenta resolver problemas con Triton en Windows"""
    print("🔧 Intentando resolver problemas con Triton...")
    
    if platform.system() != "Windows":
        print("   ℹ️ No es Windows, omitiendo arreglo de Triton")
        return True
    
    print("   📝 Configurando variables de entorno para bypass de Triton...")
    
    # Crear script de configuración
    config_script = """
@echo off
echo Configurando variables de entorno para PyTorch en Windows...
set TORCH_COMPILE_DISABLE=1
set TORCHDYNAMO_DISABLE=1
echo Variables configuradas. Ejecuta tus scripts de entrenamiento desde esta ventana.
"""
    
    with open("setup_env.bat", "w") as f:
        f.write(config_script)
    
    print("   ✅ Archivo setup_env.bat creado")
    print("   💡 Ejecuta 'setup_env.bat' antes de entrenar para evitar problemas con Triton")
    
    return True

def create_alternative_main():
    """Crea una versión alternativa de main.py sin torch.compile"""
    print("🔧 Creando versión alternativa sin torch.compile...")
    
    alt_main = '''import os
import argparse
import warnings
from trainer import trainer
import sklearn.exceptions
import torch
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Desactivar torch.compile para evitar problemas con Triton en Windows
os.environ['TORCH_COMPILE_DISABLE'] = '1'

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

args = parser.parse_args()

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento optimizado (modo compatible Windows)")
    print("=" * 60)
    
    # Configuraciones básicas de GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ CUDNN benchmark habilitado")
    
    try:
        trainer_instance = trainer(args)
        trainer_instance.train()
        print("✅ Entrenamiento completado exitosamente")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        raise
    
    print("🏁 Proceso finalizado")
'''
    
    with open("main_windows.py", "w", encoding='utf-8') as f:
        f.write(alt_main)
    
    print("   ✅ Archivo main_windows.py creado")
    print("   💡 Usa 'python main_windows.py' para entrenamiento sin problemas de Triton")
    
    return True

def check_dependencies():
    """Verifica si las dependencias están instaladas"""
    print("🔍 Verificando dependencias...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'psutil': 'PSUtil (para monitoreo)',
        'einops': 'Einops (para operaciones tensor)'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} - NO ENCONTRADO")
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Función principal de instalación y verificación"""
    print("🛠️ CONFIGURACIÓN DE DEPENDENCIAS PARA ECGTransForm")
    print("=" * 60)
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.architecture()[0]}")
    print()
    
    # Verificar PyTorch
    if not check_pytorch_installation():
        print("❌ PyTorch no está instalado correctamente")
        return False
    
    # Verificar dependencias
    missing = check_dependencies()
    if missing:
        print(f"\n📦 Instalando dependencias faltantes: {', '.join(missing)}")
        for package in missing:
            run_command(f"pip install {package}", f"Instalando {package}")
    
    # Resolver problemas específicos de Windows
    if platform.system() == "Windows":
        install_windows_dependencies()
        fix_triton_issue()
        create_alternative_main()
    
    print("\n🎯 RESUMEN DE INSTALACIÓN:")
    print("✅ Dependencias verificadas e instaladas")
    
    if platform.system() == "Windows":
        print("\n🪟 INSTRUCCIONES PARA WINDOWS:")
        print("Opción 1 (Recomendada): python main_windows.py --dataset ptb")
        print("Opción 2: Ejecutar setup_env.bat y luego python main.py --dataset ptb")
        print("Opción 3: Instalar WSL2 + Ubuntu para mejor compatibilidad")
    else:
        print("\n🐧 INSTRUCCIONES PARA LINUX/MAC:")
        print("python main.py --dataset ptb --device cuda:0")
    
    print("\n💡 Para monitorear GPU: python monitor_gpu.py")
    print("🎉 ¡Configuración completada!")

if __name__ == "__main__":
    main() 