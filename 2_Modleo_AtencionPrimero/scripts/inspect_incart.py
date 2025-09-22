import os
import argparse
from glob import glob
from collections import Counter

import numpy as np


def try_import_wfdb():
    try:
        import wfdb  # type: ignore
        return wfdb
    except Exception:
        print("wfdb no está instalado. Instálalo con: pip install wfdb")
        return None


def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {
            'count': 0,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan,
        }
    return {
        'count': int(arr.size),
        'min': float(np.nanmin(arr)),
        'median': float(np.nanmedian(arr)),
        'max': float(np.nanmax(arr)),
    }


def inspect_incart(root_dir, sample_n=25):
    files_dir = os.path.join(root_dir, 'files')
    print(f"Root: {root_dir}")
    print(f"Carpeta esperada de archivos: {files_dir}")

    if not os.path.isdir(files_dir):
        print("[ERROR] No existe la carpeta 'files' en el root especificado.")
        return 1

    hea_files = sorted(glob(os.path.join(files_dir, '*.hea')))
    dat_files = sorted(glob(os.path.join(files_dir, '*.dat')))
    hea_stems = {os.path.splitext(os.path.basename(p))[0] for p in hea_files}
    dat_stems = {os.path.splitext(os.path.basename(p))[0] for p in dat_files}
    missing_dat = sorted(list(hea_stems - dat_stems))
    missing_hea = sorted(list(dat_stems - hea_stems))
    print(f".hea encontrados: {len(hea_files)} | .dat encontrados: {len(dat_files)}")
    if missing_dat:
        print(f"[WARN] Registros sin .dat: {len(missing_dat)} (p.ej. {missing_dat[:5]})")
    if missing_hea:
        print(f"[WARN] Registros sin .hea: {len(missing_hea)} (p.ej. {missing_hea[:5]})")

    fpd_path = os.path.join(files_dir, 'files-patients-diagnoses.txt')
    if os.path.exists(fpd_path):
        try:
            with open(fpd_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [ln.strip() for ln in f]
            patient_count = sum(1 for ln in lines if ln.lower().startswith('patient'))
            print(f"files-patients-diagnoses.txt presente. Pacientes estimados: {patient_count}")
        except Exception:
            print("[WARN] No se pudo leer files-patients-diagnoses.txt")
    else:
        print("[WARN] No se encontró files-patients-diagnoses.txt")

    hier_path = os.path.join(root_dir, 'labels_hierarchy.json')
    if os.path.exists(hier_path):
        import json
        try:
            with open(hier_path, 'r', encoding='utf-8') as f:
                hier = json.load(f)
            fine = hier.get('fine_codes', [])
            coarse = hier.get('coarse_groups', {})
            print(f"Jerarquía encontrada: fine={len(fine)} | coarse_grupos={len(coarse)}")
        except Exception:
            print("[WARN] No se pudo leer labels_hierarchy.json")
    else:
        print("[WARN] No se encontró labels_hierarchy.json (se puede copiar uno genérico de 12Large)")

    wfdb = try_import_wfdb()
    if wfdb is None:
        return 2

    # Muestreo de registros para estadísticas
    stems = sorted(list(hea_stems & dat_stems))
    if not stems:
        print("[ERROR] No hay pares .hea/.dat válidos para inspección.")
        return 3
    sample = stems[:min(sample_n, len(stems))]
    fs_list = []
    leads_list = []
    length_list = []
    nan_records = []
    for s in sample:
        try:
            r = wfdb.rdrecord(os.path.join(files_dir, s))
            x = np.asarray(r.p_signal)
            fs = float(getattr(r, 'fs', np.nan))
            fs_list.append(fs)
            leads_list.append(x.shape[1])
            length_list.append(x.shape[0])
            if not np.isfinite(x).all():
                nan_records.append(s)
        except Exception as e:
            print(f"[ERROR] Falló lectura {s}: {e}")

    # Resumen
    print("\nResumen de muestreo:")
    fs_counter = Counter([round(f, 3) for f in fs_list if np.isfinite(f)])
    print(f"- Frecuencias de muestreo (conteo): {dict(fs_counter)}")
    print(f"- Derivaciones (min/med/max): {summarize(leads_list)}")
    print(f"- Longitud T (muestras) (min/med/max): {summarize(length_list)}")
    if nan_records:
        print(f"[WARN] Registros con NaN/inf: {len(nan_records)} (p.ej. {nan_records[:5]})")

    print("\nSugerencias:")
    print("- Verifica que haya pares .hea/.dat completos y el archivo files-patients-diagnoses.txt.")
    print("- Si no hay etiquetas SNOMED, copia una jerarquía base: 'cp datos/12Large/labels_hierarchy.json datos/StPetersburgIncart12LeadArrhythmiaDatabase/labels_hierarchy.json'.")
    print("- Entrena con --no_sampler en INCART y considera desactivar pérdidas que requieran etiquetas inexistentes.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=os.path.join('datos', 'StPetersburgIncart12LeadArrhythmiaDatabase'))
    ap.add_argument('--sample_n', type=int, default=25)
    args = ap.parse_args()
    code = inspect_incart(args.root, sample_n=args.sample_n)
    raise SystemExit(code)


if __name__ == '__main__':
    main()


