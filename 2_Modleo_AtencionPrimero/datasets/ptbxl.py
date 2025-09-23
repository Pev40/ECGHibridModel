import os
import ast
import json
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .ecg12large import _apply_filters, _resample_if_needed, extract_patient_id as _extract_from_header


def _ptbxl_extract_patient_id(row):
    pid = row.get('patient_id', None)
    if pid is None:
        return None
    try:
        return str(int(pid))
    except Exception:
        return str(pid)


def _parse_scp_codes(s):
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}
    return {}


class PTBXL(Dataset):
    def __init__(self, root_dir, split='train', sequence_len=5000, target_fs=500.0,
                 hierarchy_path=None, normalize='zscore', random_crop=True, eval_mode=False,
                 cache_dir=None, bandpass_hz=(0.5, 45.0), notch_hz=None,
                 aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                 aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
                 aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0,
                 folds=(0,1,2,3,4), use_high_res=True):
        super().__init__()
        self.root_dir = root_dir
        self.sequence_len = int(sequence_len)
        self.normalize = normalize
        self.random_crop = bool(random_crop)
        self.eval_mode = bool(eval_mode)
        self.target_fs = float(target_fs) if target_fs is not None else None
        self.bandpass_hz = bandpass_hz
        self.notch_hz = notch_hz
        self.cache_dir = cache_dir
        self.aug_jitter_std = float(aug_jitter_std)
        self.aug_shift_max = int(aug_shift_max)
        self.aug_lead_drop_prob = float(aug_lead_drop_prob)
        self.aug_amp_scale_min = float(aug_amp_scale_min)
        self.aug_amp_scale_max = float(aug_amp_scale_max)
        self.aug_lead_noise_scale_max = float(aug_lead_noise_scale_max)
        self.aug_time_warp_max = float(aug_time_warp_max)
        self.aug_time_warp_p = float(aug_time_warp_p)
        self.use_high_res = bool(use_high_res)

        # jerarquía
        self.hierarchy = None
        if hierarchy_path and os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                self.hierarchy = json.load(f)
            self.fine_codes = self.hierarchy.get('fine_codes', [])
            self.fine_code_to_idx = {c: i for i, c in enumerate(self.fine_codes)}
            self.coarse_groups = self.hierarchy.get('coarse_groups', {})
            self.coarse_names = list(self.coarse_groups.keys())
            self.coarse_name_to_idx = {g: i for i, g in enumerate(self.coarse_names)}

        # CSV principal
        csv_path = os.path.join(root_dir, 'ptbxl_database.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'No se encontró ptbxl_database.csv en {root_dir}')
        df = pd.read_csv(csv_path)
        # seleccionar archivos por split fold
        # PTBXL tiene columna strat_fold (1..10), usaremos 1..8 train, 9 val, 10 test por defecto
        if split == 'train':
            mask = df['strat_fold'].isin([1,2,3,4,5,6,7,8])
        elif split == 'val':
            mask = df['strat_fold'] == 9
        else:
            mask = df['strat_fold'] == 10
        df = df[mask].copy()

        # preparar lista de muestras
        self.samples = []
        base_subdir = 'records500' if self.use_high_res else 'records100'
        for _, row in df.iterrows():
            scp = _parse_scp_codes(row.get('scp_codes', '{}'))
            # Convertir scp (diagnósticos PTB-XL) a códigos fine si hay mapeo SNOMED
            # Asumimos que jerarquía usa SNOMED; si se requiere un mapeo scp->SNOMED, se puede añadir aquí
            y_fine = np.zeros(len(getattr(self, 'fine_codes', [])), dtype=np.float32)
            for code in getattr(self, 'fine_codes', []):
                # si el fine code es literal igual a alguna clave scp, activamos
                if code in scp:
                    y_fine[self.fine_code_to_idx[code]] = 1.0
            # si no hay jerarquía o no se activa nada, saltamos en modo multilabel
            if getattr(self, 'fine_codes', None) is not None and y_fine.sum() == 0:
                continue
            y_coarse = None
            if getattr(self, 'coarse_groups', None) is not None:
                y_coarse = np.zeros(len(self.coarse_names), dtype=np.float32)
                for g_idx, g in enumerate(self.coarse_names):
                    codes = set(self.coarse_groups[g])
                    # activar si coincide alguna clave de scp
                    for k in scp.keys():
                        if k in codes:
                            y_coarse[g_idx] = 1.0
                            break

            # ruta WFDB
            rel = row['filename_hr'] if self.use_high_res else row['filename_lr']
            hea = os.path.join(root_dir, rel + '.hea')
            dat_or_mat = os.path.join(root_dir, rel + '.dat')
            if not os.path.exists(hea):
                continue
            self.samples.append((hea, dat_or_mat, y_fine, y_coarse, row))

        # pre-cálculo de patient_ids
        self.sample_patient_ids = [ _ptbxl_extract_patient_id(s[-1]) for s in self.samples ]

    def __len__(self):
        return len(self.samples)

    def _normalize(self, x):
        if self.normalize == 'zscore':
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std
        return x

    def _pad_or_trim(self, x):
        c, t = x.shape
        if t == self.sequence_len:
            return x
        if t > self.sequence_len:
            return x[:, :self.sequence_len]
        pad = np.zeros((c, self.sequence_len - t), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)

    def __getitem__(self, idx):
        hea, data_path, y_fine, y_coarse, row = self.samples[idx]
        # leer con wfdb
        import wfdb
        try:
            record_dir = os.path.dirname(hea)
            record_basename = os.path.splitext(os.path.basename(hea))[0]
            r = wfdb.rdrecord(os.path.join(record_dir, record_basename))
            x = np.asarray(r.p_signal).T  # [C, T]
            fs = float(r.fs)
        except Exception:
            # fallback a 500 Hz
            x = None
            fs = 500.0
        if x is None:
            # último recurso: fallo duro
            raise RuntimeError(f'No se pudo leer {hea} con wfdb')
        # resample/filtros
        x, fs_eff = _resample_if_needed(x, fs, self.target_fs)
        x = _apply_filters(x, fs_eff or (self.target_fs or 500.0), band=self.bandpass_hz, notch_hz=self.notch_hz)
        x = self._normalize(x)

        # crop/pad
        c, t = x.shape
        if t >= self.sequence_len:
            if self.eval_mode or not self.random_crop:
                start = max(0, (t - self.sequence_len)//2)
            else:
                start = int(np.random.randint(0, max(1, t - self.sequence_len + 1)))
            x = x[:, start:start+self.sequence_len]
        else:
            x = self._pad_or_trim(x)

        x = torch.from_numpy(x).float()

        # augmentaciones solo entrenamiento
        if not self.eval_mode:
            # time-warp (ligero)
            if self.aug_time_warp_max > 0 and np.random.rand() < self.aug_time_warp_p:
                L = x.shape[1]
                wf = float(np.random.uniform(max(0.5, 1.0 - self.aug_time_warp_max), 1.0 + self.aug_time_warp_max))
                xw = F.interpolate(x.unsqueeze(0), scale_factor=wf, mode='linear', align_corners=False).squeeze(0)
                if xw.shape[1] >= L:
                    x = xw[:, :L]
                else:
                    pad = torch.zeros(x.shape[0], L - xw.shape[1], dtype=x.dtype, device=x.device)
                    x = torch.cat([xw, pad], dim=1)
            if self.aug_shift_max > 0:
                shift = int(np.random.randint(-self.aug_shift_max, self.aug_shift_max + 1))
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=1)
            if self.aug_jitter_std > 0:
                if self.aug_lead_noise_scale_max and self.aug_lead_noise_scale_max > 1.0:
                    c = x.shape[0]
                    scales = torch.empty(c, device=x.device).uniform_(1.0/self.aug_lead_noise_scale_max, self.aug_lead_noise_scale_max).view(-1,1)
                    noise = torch.randn_like(x) * float(self.aug_jitter_std) * scales
                    x = x + noise
                else:
                    x = x + torch.randn_like(x) * float(self.aug_jitter_std)
            if self.aug_lead_drop_prob > 0:
                c = x.shape[0]
                keep_mask = (torch.rand(c) > self.aug_lead_drop_prob).float().to(x.device)
                x = x * keep_mask.view(-1, 1)
            if self.aug_amp_scale_max != 1.0 or self.aug_amp_scale_min != 1.0:
                scale = float(np.random.uniform(self.aug_amp_scale_min, self.aug_amp_scale_max))
                x = x * scale

        y_fine_t = torch.from_numpy(y_fine).float() if y_fine is not None else None
        y_coarse_t = torch.from_numpy(y_coarse).float() if y_coarse is not None else None
        return {
            'samples': x,
            'labels_fine': y_fine_t if y_fine_t is not None else torch.zeros(0),
            'labels_coarse': y_coarse_t if y_coarse_t is not None else torch.zeros(0),
            'hea': hea,
            'patient_id': _ptbxl_extract_patient_id(row)
        }


