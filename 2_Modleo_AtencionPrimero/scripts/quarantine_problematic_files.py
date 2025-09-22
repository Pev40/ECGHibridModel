import os
import shutil
from glob import glob
from tqdm import tqdm

def quarantine_files():
    """
    Lee la lista de archivos problemáticos y los mueve a un directorio de cuarentena.
    """
    problematic_files_path = os.path.join('datos', 'problematic_files.txt')
    if not os.path.exists(problematic_files_path):
        print(f"No se encontró el archivo de informe: {problematic_files_path}")
        print("Por favor, ejecuta primero 'scripts/check_data_integrity.py' para generar el informe.")
        return

    source_root = os.path.join('datos', 'WFDBRecords')
    quarantine_root = os.path.join('datos', 'WFDBRecords_problematicos')
    
    print(f"Creando directorio de cuarentena en: {quarantine_root}")
    os.makedirs(quarantine_root, exist_ok=True)

    with open(problematic_files_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filtrar cabecera y líneas vacías
    mat_paths_to_move = [
        line.split(':')[0].strip() for line in lines if not line.startswith('#') and line.strip()
    ]

    if not mat_paths_to_move:
        print("El informe de archivos problemáticos está vacío. No hay nada que mover.")
        return

    print(f"Se moverán {len(mat_paths_to_move)} registros (.mat y .hea) a cuarentena.")

    moved_count = 0
    failed_moves = []

    with tqdm(total=len(mat_paths_to_move), desc="Moviendo archivos a cuarentena") as pbar:
        for mat_path in mat_paths_to_move:
            mat_path = os.path.normpath(mat_path)
            hea_path = os.path.splitext(mat_path)[0] + '.hea'

            # Construir la ruta de destino manteniendo la estructura de carpetas
            relative_path = os.path.relpath(mat_path, source_root)
            dest_path_mat = os.path.join(quarantine_root, relative_path)
            
            # Crear el subdirectorio de destino si no existe
            os.makedirs(os.path.dirname(dest_path_mat), exist_ok=True)

            try:
                # Mover el archivo .mat
                if os.path.exists(mat_path):
                    shutil.move(mat_path, dest_path_mat)
                
                # Mover el archivo .hea
                dest_path_hea = os.path.splitext(dest_path_mat)[0] + '.hea'
                if os.path.exists(hea_path):
                    shutil.move(hea_path, dest_path_hea)

                moved_count += 1
            except Exception as e:
                failed_moves.append((mat_path, str(e)))
            
            pbar.update(1)

    print(f"\nProceso completado. Se movieron exitosamente {moved_count} registros.")
    if failed_moves:
        print(f"Hubo errores al mover {len(failed_moves)} registros:")
        for path, error in failed_moves:
            print(f"  - {path}: {error}")

if __name__ == '__main__':
    quarantine_files()
