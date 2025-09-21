import os
import re
import json
from glob import glob
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


def _parse_header_labels(hea_path):
    labels = []
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                if ln.startswith('#Dx:'):
                    txt = ln.replace('#Dx:', '').strip()
                    parts = re.split(r'[,:;\s]+', txt)
                    labels = [p for p in parts if p]
                    break
    except Exception:
        pass
    return labels


def _load_signal_mat(mat_path):
    if loadmat is None:
        raise RuntimeError('scipy.io.loadmat no disponible. Instala scipy.')
    m = loadmat(mat_path)
    for key in ('val', 'data', 'signal'):
        if key in m:
            arr = np.asarray(m[key])  # [leads, T]
            return arr
    raise RuntimeError(f'No se encontró variable de señal en {mat_path}')


def build_code_map(hea_files, top_k=10, save_path=None):
    counter = Counter()
    for hea in hea_files:
        labels = _parse_header_labels(hea)
        counter.update(labels)
    most_common = [c for c, _ in counter.most_common(top_k)]
    code_to_idx = {code: idx for idx, code in enumerate(most_common)}
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(code_to_idx, f, ensure_ascii=False, indent=2)
    return code_to_idx


class WFDBECGDataset(Dataset):
    def __init__(self, root_dir, sequence_len=5000, code_to_idx=None, files=None, normalize='zscore'):
        super().__init__()
        self.root_dir = root_dir
        if files is None:
            self.hea_files = glob(os.path.join(root_dir, '**', '*.hea'), recursive=True)
        else:
            self.hea_files = files
        self.sequence_len = sequence_len
        self.code_to_idx = code_to_idx or {}
        self.normalize = normalize

        # filtrar a ejemplos que tengan .mat
        self.pairs = []
        for hea in self.hea_files:
            mat = os.path.splitext(hea)[0] + '.mat'
            if os.path.exists(mat):
                self.pairs.append((hea, mat))

        # si no hay code_map, derivar de los archivos presentes (top 10)
        if not self.code_to_idx:
            self.code_to_idx = build_code_map([p[0] for p in self.pairs], top_k=10)

        # filtrar pares sin label mapeable
        self.samples = []
        for hea, mat in self.pairs:
            labels = _parse_header_labels(hea)
            idx = None
            for code in labels:
                if code in self.code_to_idx:
                    idx = self.code_to_idx[code]
                    break
            if idx is not None:
                self.samples.append((hea, mat, idx))

    def __len__(self):
        return len(self.samples)

    def _pad_or_trim(self, x):
        # x: [C, T]
        c, t = x.shape
        if t == self.sequence_len:
            return x
        if t > self.sequence_len:
            return x[:, :self.sequence_len]
        # pad at the end
        pad = np.zeros((c, self.sequence_len - t), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)

    def _normalize(self, x):
        if self.normalize == 'zscore':
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        hea, mat, y = self.samples[idx]
        x = _load_signal_mat(mat)  # [C, T]
        x = self._pad_or_trim(x)
        x = self._normalize(x)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()
        return {'samples': x, 'labels': y, 'hea': hea}


