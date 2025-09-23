import os
import sys
from glob import glob
from tqdm import tqdm
import torch
import numpy as np

# Ensure repo root is on PYTHONPATH when running directly
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.ecg12large import _load_signal_mat
from datasets.ecg12large import _parse_header_labels


def main():
    root = os.path.join('datos', 'ECG12Large', 'WFDBRecords')
    cache_dir = os.path.join('datos', 'pt_cache')
    os.makedirs(cache_dir, exist_ok=True)

    hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    for hea in tqdm(hea_files, desc='Caching'):
        mat = os.path.splitext(hea)[0] + '.mat'
        if not os.path.exists(mat):
            continue
        rec_id = os.path.relpath(os.path.splitext(mat)[0], root).replace(os.sep, '_')
        pt_path = os.path.join(cache_dir, f"{rec_id}.pt")
        if os.path.exists(pt_path):
            continue
        try:
            x = _load_signal_mat(mat)
            # padding/normalize se hará en dataset; aquí guardamos raw para flexibilidad
            torch.save(torch.from_numpy(np.asarray(x)).float(), pt_path)
        except Exception:
            continue

    print('Done caching at', cache_dir)


if __name__ == '__main__':
    main()


