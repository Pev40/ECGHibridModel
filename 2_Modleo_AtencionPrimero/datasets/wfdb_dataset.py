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
    def __init__(self, root_dir, sequence_len=5000, code_to_idx=None, files=None, normalize='zscore',
                 multilabel=False, hierarchy_path=None, cache_dir=None):
        super().__init__()
        self.root_dir = root_dir
        if files is None:
            self.hea_files = glob(os.path.join(root_dir, '**', '*.hea'), recursive=True)
        else:
            self.hea_files = files
        self.sequence_len = sequence_len
        self.code_to_idx = code_to_idx or {}
        self.normalize = normalize
        self.multilabel = multilabel
        self.cache_dir = cache_dir

        # Jerarquía (opcional)
        self.hierarchy = None
        if hierarchy_path and os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                self.hierarchy = json.load(f)
            self.fine_codes = self.hierarchy.get('fine_codes', [])
            self.fine_code_to_idx = {c: i for i, c in enumerate(self.fine_codes)}
            self.coarse_groups = self.hierarchy.get('coarse_groups', {})
            self.coarse_names = list(self.coarse_groups.keys())
            self.coarse_name_to_idx = {g: i for i, g in enumerate(self.coarse_names)}

        # filtrar a ejemplos que tengan .mat
        self.pairs = []
        for hea in self.hea_files:
            mat = os.path.splitext(hea)[0] + '.mat'
            if os.path.exists(mat):
                self.pairs.append((hea, mat))

        # si no hay code_map, derivar de los archivos presentes (top 10)
        if not self.code_to_idx and not self.multilabel:
            self.code_to_idx = build_code_map([p[0] for p in self.pairs], top_k=10)

        # filtrar pares sin label mapeable
        self.samples = []
        for hea, mat in self.pairs:
            labels = _parse_header_labels(hea)
            if self.multilabel and self.hierarchy:
                # vector fine
                y_fine = np.zeros(len(self.fine_codes), dtype=np.float32)
                for code in labels:
                    if code in self.fine_code_to_idx:
                        y_fine[self.fine_code_to_idx[code]] = 1.0
                if y_fine.sum() == 0:
                    continue
                # vector coarse derivado
                y_coarse = np.zeros(len(self.coarse_names), dtype=np.float32)
                for g_idx, g in enumerate(self.coarse_names):
                    group_codes = set(self.coarse_groups[g])
                    # activar coarse si alguna fine del grupo está activa
                    for code in labels:
                        if code in group_codes:
                            y_coarse[g_idx] = 1.0
                            break
                self.samples.append((hea, mat, y_fine, y_coarse))
            else:
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
        rec = self.samples[idx]
        hea, mat = rec[0], rec[1]
        # Cache: .pt por registro (opcional)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            rec_id = os.path.relpath(os.path.splitext(mat)[0], self.root_dir).replace(os.sep, '_')
            pt_path = os.path.join(self.cache_dir, f"{rec_id}.pt")
            if os.path.exists(pt_path):
                x = torch.load(pt_path, map_location='cpu').numpy()
            else:
                x = _load_signal_mat(mat)
                x = self._pad_or_trim(x)
                x = self._normalize(x)
                torch.save(torch.from_numpy(x).float(), pt_path)
        else:
            x = _load_signal_mat(mat)  # [C, T]
            x = self._pad_or_trim(x)
            x = self._normalize(x)
        x = torch.from_numpy(x).float()
        if self.multilabel and self.hierarchy:
            y_fine = torch.from_numpy(rec[2]).float()
            y_coarse = torch.from_numpy(rec[3]).float()
            return {'samples': x, 'labels_fine': y_fine, 'labels_coarse': y_coarse, 'hea': hea}
        else:
            y = torch.tensor(rec[2]).long()
            return {'samples': x, 'labels': y, 'hea': hea}


