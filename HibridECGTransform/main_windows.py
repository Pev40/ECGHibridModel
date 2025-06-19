import os
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
