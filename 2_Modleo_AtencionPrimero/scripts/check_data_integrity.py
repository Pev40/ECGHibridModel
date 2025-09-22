import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

try:
    from scipy.io import loadmat
except ImportError:
    print("Por favor, instala scipy: pip install scipy")
    loadmat = None

def _load_signal_mat(mat_path):
    """Carga la señal de un archivo .mat, buscando en claves comunes."""
    if loadmat is None:
        raise RuntimeError('scipy.io.loadmat no disponible. Instala scipy.')
    m = loadmat(mat_path)
    for key in ('val', 'data', 'signal'):
        if key in m:
            arr = np.asarray(m[key])
            return arr
    raise RuntimeError(f'No se encontró variable de señal en {mat_path}')

def check_file_integrity(mat_path):
    """
    Comprueba la integridad de un único archivo .mat.
    Devuelve None si es válido, o un string con el motivo del error si es inválido.
    """
    try:
        signal = _load_signal_mat(mat_path)

        # 1. Comprobar valores no finitos (NaN, inf)
        if not np.isfinite(signal).all():
            return "contiene valores no finitos (NaN/inf)"

        # 2. Comprobar si la señal es plana (std dev muy baja)
        # Se calcula por canal/derivación para manejar casos donde solo una derivación es plana
        stds = np.std(signal, axis=1)
        if np.any(stds < 1e-6):
            return "contiene al menos una derivación plana (sin señal)"

        return None
    except Exception as e:
        return f"error al cargar o procesar: {e}"

def main():
    """Función principal para escanear y reportar archivos problemáticos."""
    print("Iniciando la verificación de integridad de los datos...")
    data_root = os.path.join('datos', 'WFDBRecords')
    
    # Usamos glob para encontrar todos los archivos .mat de forma recursiva
    all_mat_files = glob(os.path.join(data_root, '**', '*.mat'), recursive=True)
    
    if not all_mat_files:
        print(f"No se encontraron archivos .mat en el directorio: {data_root}")
        return

    print(f"Se encontraron {len(all_mat_files)} archivos .mat para analizar.")

    problematic_files = []
    
    # Usamos ProcessPoolExecutor para paralelizar la verificación de archivos, es mucho más rápido
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(check_file_integrity, path): path for path in all_mat_files}
        
        with tqdm(total=len(all_mat_files), desc="Analizando archivos") as pbar:
            for future in as_completed(future_to_file):
                path = future_to_file[future]
                reason = future.result()
                if reason:
                    problematic_files.append((path, reason))
                pbar.update(1)

    # Guardar el informe
    report_path = os.path.join('datos', 'problematic_files.txt')
    if problematic_files:
        print(f"\n¡Verificación completada! Se encontraron {len(problematic_files)} archivos problemáticos.")
        print(f"Se ha generado un informe en: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Informe de integridad de datos\n")
            f.write(f"Total de archivos problemáticos: {len(problematic_files)}\n\n")
            for path, reason in sorted(problematic_files):
                f.write(f"{path}: {reason}\n")
    else:
        print("\n¡Verificación completada! No se encontraron archivos problemáticos.")
        # Si no hay problemas, podemos borrar un informe antiguo si existe
        if os.path.exists(report_path):
            os.remove(report_path)

if __name__ == '__main__':
    main()
