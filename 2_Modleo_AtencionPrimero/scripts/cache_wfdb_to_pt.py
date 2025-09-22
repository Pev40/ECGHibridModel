import os
from glob import glob
from tqdm import tqdm
from datasets.wfdb_dataset import _load_signal_mat
from datasets.wfdb_dataset import _parse_header_labels
import torch
import numpy as np


def main():
    root = os.path.join('datos', 'WFDBRecords')
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


