import os
import re
import json
import math
from glob import glob

import numpy as np

try:
    import wfdb  # opcional
except Exception:
    wfdb = None

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


DATOS_DIR = os.path.join('datos', 'WFDBRecords')


def parse_header_basic(hea_path):
    info = {
        'record': None,
        'n_sig': None,
        'fs': None,
        'sig_len': None,
        'dx': []
    }
    with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f.readlines()]
    if not lines:
        return info
    # primera línea WFDB: <record> <n_sig> <fs> <sig_len> ...
    first = lines[0].split()
    if len(first) >= 4:
        info['record'] = first[0]
        try:
            info['n_sig'] = int(first[1])
        except Exception:
            info['n_sig'] = None
        try:
            info['fs'] = float(first[2])
        except Exception:
            info['fs'] = None
        try:
            info['sig_len'] = int(first[3])
        except Exception:
            info['sig_len'] = None
    # etiquetas en líneas con '#Dx:' como en Challenge 2020
    for ln in lines:
        if ln.startswith('#Dx:'):
            codes = re.split(r'[,:;\s]+', ln.replace('#Dx:', '').strip())
            info['dx'] = [c for c in codes if c]
            break
    return info


def load_signal_mat(mat_path):
    if loadmat is None:
        return None
    try:
        m = loadmat(mat_path)
        # Común en WFDB .mat (Challenge 2020): variable 'val' con shape [n_leads, T]
        for key in ('val', 'data', 'signal'):
            if key in m:
                arr = np.asarray(m[key])
                return arr
    except Exception:
        return None
    return None


def summarize(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        'count': int(arr.size),
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr))
    }


def main(max_files=300):
    hea_files = glob(os.path.join(DATOS_DIR, '**', '*.hea'), recursive=True)
    hea_files = hea_files[:max_files]
    if not hea_files:
        print('No se encontraron .hea en', DATOS_DIR)
        return

    fs_list, n_sig_list, siglen_list = [], [], []
    n_leads_from_mat = []
    labels_counter = {}

    for hea in hea_files:
        info = parse_header_basic(hea)
        if info['fs'] is not None:
            fs_list.append(info['fs'])
        if info['n_sig'] is not None:
            n_sig_list.append(info['n_sig'])
        if info['sig_len'] is not None:
            siglen_list.append(info['sig_len'])
        for c in info['dx']:
            labels_counter[c] = labels_counter.get(c, 0) + 1

        # inspeccionar .mat asociado
        mat_path = os.path.splitext(hea)[0] + '.mat'
        if os.path.exists(mat_path):
            arr = load_signal_mat(mat_path)
            if arr is not None and arr.ndim == 2:
                n_leads_from_mat.append(arr.shape[0])

    summary = {
        'fs': summarize(fs_list),
        'n_signals_header': summarize(n_sig_list),
        'signal_length_samples': summarize(siglen_list),
        'n_leads_from_mat': summarize(n_leads_from_mat),
        'top_labels': sorted(labels_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    }

    print(json.dumps(summary, indent=2))

    # Recomendaciones preliminares de alimentación del modelo
    # - F (num_leads): usar el valor modal de n_leads_from_mat o n_signals_header
    # - T (sequence_len): recorte/segmentación a cuantiles (p.ej. p25-p75) y/o padding
    # - fs: si heterogéneo, considerar resample a valor mayoritario


if __name__ == '__main__':
    main()


