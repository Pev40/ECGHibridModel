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

try:
    from scipy.signal import butter, filtfilt, iirnotch, resample_poly
except Exception:
    butter = None
    filtfilt = None
    iirnotch = None
    resample_poly = None


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


def _parse_header_fs(hea_path):
    fs = None
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            first = f.readline()
            m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", first, re.IGNORECASE)
            if m:
                fs = float(m.group(1))
            else:
                m2 = re.search(r"(\d+(?:\.\d+)?)/(?:mV|\w+)", first)
                if m2:
                    try:
                        fs = float(m2.group(1))
                    except Exception:
                        pass
        if fs is None:
            with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    m = re.search(r"(?:Sampling frequency|fs)\s*:?\s*(\d+(?:\.\d+)?)", ln, re.IGNORECASE)
                    if m:
                        fs = float(m.group(1))
                        break
    except Exception:
        pass
    return fs


def extract_patient_id(hea_path, root_dir=None):
    pid = None
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                if ln.startswith('#'):
                    m = re.search(r"(Patient ID|Patient|Subject ID|Subject|Record ID|PID)\s*:?\s*([\w\-]+)", ln, re.IGNORECASE)
                    if m:
                        pid = m.group(2)
                        break
    except Exception:
        pass
    if pid is None and root_dir is not None:
        try:
            rel = os.path.relpath(hea_path, root_dir)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                pid = parts[0]
        except Exception:
            pid = None
    if pid is None:
        stem = os.path.splitext(os.path.basename(hea_path))[0]
        pid = re.split(r"[_\-]", stem)[0]
    return pid


def _butter_bandpass(lowcut_hz, highcut_hz, fs, order=2):
    if butter is None or filtfilt is None:
        return None, None
    nyq = 0.5 * fs
    low = max(1e-6, lowcut_hz / nyq)
    high = min(0.999, highcut_hz / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _apply_filters(x_np, fs, band=(0.5, 45.0), notch_hz=None):
    if butter is None or filtfilt is None:
        return x_np
    x = x_np
    try:
        if band is not None:
            b, a = _butter_bandpass(band[0], band[1], fs, order=2)
            if b is not None:
                x = filtfilt(b, a, x, axis=1)
        if notch_hz in (50, 60) and iirnotch is not None:
            q = 30.0
            b_notch, a_notch = iirnotch(w0=notch_hz/(fs/2), Q=q)
            x = filtfilt(b_notch, a_notch, x, axis=1)
    except Exception:
        return x_np
    return x


def _resample_if_needed(x_np, fs, target_fs):
    if target_fs is None or fs is None or resample_poly is None:
        return x_np, fs
    if abs(fs - target_fs) < 1e-3:
        return x_np, fs
    try:
        from fractions import Fraction
        frac = Fraction(str(target_fs / fs)).limit_denominator(100)
        up, down = frac.numerator, frac.denominator
        x_rs = resample_poly(x_np, up, down, axis=1)
        return x_rs, float(target_fs)
    except Exception:
        return x_np, fs
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
                 multilabel=False, hierarchy_path=None, cache_dir=None, random_crop=True,
                 target_fs=500.0, bandpass_hz=(0.5, 45.0), notch_hz=None, eval_mode=False,
                 aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                 aug_amp_scale_min=1.0, aug_amp_scale_max=1.0):
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
        self.random_crop = random_crop
        self.eval_mode = eval_mode
        self.target_fs = target_fs
        self.bandpass_hz = bandpass_hz
        self.notch_hz = notch_hz
        # Augmentaciones (solo en entrenamiento)
        self.aug_jitter_std = float(aug_jitter_std)
        self.aug_shift_max = int(aug_shift_max)
        self.aug_lead_drop_prob = float(aug_lead_drop_prob)
        self.aug_amp_scale_min = float(aug_amp_scale_min)
        self.aug_amp_scale_max = float(aug_amp_scale_max)

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
        self.sample_patient_ids = []
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
            self.sample_patient_ids.append(extract_patient_id(hea, self.root_dir))

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
        fs = _parse_header_fs(hea)
        # Cache: .pt por registro (opcional) con preprocesado
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            rec_id = os.path.relpath(os.path.splitext(mat)[0], self.root_dir).replace(os.sep, '_')
            pt_path = os.path.join(self.cache_dir, f"{rec_id}.pt")
            if os.path.exists(pt_path):
                cached = torch.load(pt_path, map_location='cpu')
                if isinstance(cached, dict) and 'signal' in cached:
                    x = cached['signal'].numpy()
                else:
                    x = cached.numpy()
            else:
                x = _load_signal_mat(mat)
                x, fs_eff = _resample_if_needed(x, fs, self.target_fs)
                x = _apply_filters(x, fs_eff or (self.target_fs or 500.0), band=self.bandpass_hz, notch_hz=self.notch_hz)
                x = self._normalize(x)
                torch.save({'signal': torch.from_numpy(x).float(), 'fs': fs_eff or fs}, pt_path)
        else:
            x = _load_signal_mat(mat)
            x, fs_eff = _resample_if_needed(x, fs, self.target_fs)
            x = _apply_filters(x, fs_eff or (self.target_fs or 500.0), band=self.bandpass_hz, notch_hz=self.notch_hz)
            x = self._normalize(x)
        # seleccionar crop
        c, t = x.shape
        if t >= self.sequence_len:
            if self.eval_mode or not self.random_crop:
                start = max(0, (t - self.sequence_len)//2)
            else:
                rng = np.random.default_rng()
                start = int(rng.integers(0, max(1, t - self.sequence_len + 1)))
            x = x[:, start:start+self.sequence_len]
        else:
            x = self._pad_or_trim(x)

        x = torch.from_numpy(x).float()
        # Augmentaciones en modo entrenamiento
        if not self.eval_mode:
            # Shift temporal (circular)
            if self.aug_shift_max > 0:
                shift = int(np.random.randint(-self.aug_shift_max, self.aug_shift_max + 1))
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=1)
            # Jitter gaussiano
            if self.aug_jitter_std > 0:
                x = x + torch.randn_like(x) * float(self.aug_jitter_std)
            # Lead dropout
            if self.aug_lead_drop_prob > 0:
                c = x.shape[0]
                keep_mask = (torch.rand(c) > self.aug_lead_drop_prob).float().to(x.device)
                x = x * keep_mask.view(-1, 1)
            # Amplitude scaling
            if self.aug_amp_scale_max != 1.0 or self.aug_amp_scale_min != 1.0:
                scale = float(np.random.uniform(self.aug_amp_scale_min, self.aug_amp_scale_max))
                x = x * scale
        if self.multilabel and self.hierarchy:
            y_fine = torch.from_numpy(rec[2]).float()
            y_coarse = torch.from_numpy(rec[3]).float()
            return {'samples': x, 'labels_fine': y_fine, 'labels_coarse': y_coarse, 'hea': hea, 'patient_id': extract_patient_id(hea, self.root_dir)}
        else:
            y = torch.tensor(rec[2]).long()
            return {'samples': x, 'labels': y, 'hea': hea, 'patient_id': extract_patient_id(hea, self.root_dir)}


