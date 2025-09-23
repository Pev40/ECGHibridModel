import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .ecg12large import _apply_filters, _resample_if_needed


def _incart_patient_id_from_txt(files_patients_path, record_basename):
    try:
        with open(files_patients_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f.readlines()]
        patient = None
        for i, ln in enumerate(lines):
            if ln.lower().startswith('patient'):
                patient = ln.split()[-1]
                recs_line = lines[i+1] if i+1 < len(lines) else ''
                recs = recs_line.split()
                if record_basename in recs:
                    return str(patient)
        return None
    except Exception:
        return None


class INCART12Lead(Dataset):
    def __init__(self, root_dir, split='train', sequence_len=5000, target_fs=257.0,
                 hierarchy_path=None, normalize='zscore', random_crop=True, eval_mode=False,
                 cache_dir=None, bandpass_hz=(0.5, 45.0), notch_hz=None,
                 aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                 aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
                 aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0):
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

        # listar archivos
        files_dir = os.path.join(root_dir, 'files')
        records = []
        for name in os.listdir(files_dir):
            if name.lower().endswith('.hea'):
                records.append(os.path.splitext(name)[0])
        records = sorted(records)

        # split simple por pacientes (a partir de files-patients-diagnoses.txt)
        fpd = os.path.join(files_dir, 'files-patients-diagnoses.txt')
        patient_to_records = {}
        if os.path.exists(fpd):
            with open(fpd, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [ln.strip() for ln in f.readlines()]
            current_patient = None
            for ln in lines:
                if ln.lower().startswith('patient'):
                    try:
                        current_patient = ln.split()[-1]
                    except Exception:
                        current_patient = None
                elif ln and ln[0].upper() == 'I':
                    recs = ln.split()
                    if current_patient is not None:
                        for r in recs:
                            patient_to_records.setdefault(current_patient, []).append(r)

        patients = sorted(patient_to_records.keys()) if patient_to_records else None
        def assign_split(rec_basename):
            if not patients:
                # fallback por índice
                idx = records.index(rec_basename)
                if idx % 10 in (8,):
                    return 'val'
                if idx % 10 in (9,):
                    return 'test'
                return 'train'
            # 70/15/15 por pacientes
            pidx = None
            for i, p in enumerate(patients):
                if rec_basename in patient_to_records.get(p, []):
                    pidx = i
                    break
            if pidx is None:
                return 'train'
            n = len(patients)
            n_tr = int(0.7*n)
            n_va = int(0.15*n)
            if pidx < n_tr:
                return 'train'
            elif pidx < n_tr + n_va:
                return 'val'
            else:
                return 'test'

        wanted_split = split
        self.samples = []
        for r in records:
            if assign_split(r) != wanted_split:
                continue
            hea = os.path.join(files_dir, r + '.hea')
            dat = os.path.join(files_dir, r + '.dat')
            # INCART no trae etiquetas SNOMED en #Dx; almacenamiento vacío
            y_fine = np.zeros(len(getattr(self, 'fine_codes', [])), dtype=np.float32) if getattr(self, 'fine_codes', None) is not None else None
            y_coarse = np.zeros(len(getattr(self, 'coarse_names', [])), dtype=np.float32) if getattr(self, 'coarse_names', None) is not None else None
            self.samples.append((hea, dat, y_fine, y_coarse, r))

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
        hea, dat, y_fine, y_coarse, rec = self.samples[idx]
        # leer con wfdb
        import wfdb
        record_dir = os.path.dirname(hea)
        record_basename = os.path.splitext(os.path.basename(hea))[0]
        r = wfdb.rdrecord(os.path.join(record_dir, record_basename))
        x = np.asarray(r.p_signal).T  # [C, T]
        fs = float(r.fs)

        x, fs_eff = _resample_if_needed(x, fs, self.target_fs)
        x = _apply_filters(x, fs_eff or (self.target_fs or fs), band=self.bandpass_hz, notch_hz=self.notch_hz)
        x = self._normalize(x)

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

        if not self.eval_mode:
            if self.aug_time_warp_max > 0 and np.random.rand() < self.aug_time_warp_p:
                L = x.shape[1]
                wf = float(np.random.uniform(max(0.5, 1.0 - self.aug_time_warp_max), 1.0 + self.aug_time_warp_max))
                xw = F.interpolate(x.unsqueeze(0), scale_factor=(1.0, wf), mode='bilinear', align_corners=False).squeeze(0)
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

        # patient id
        pid = _incart_patient_id_from_txt(os.path.join(self.root_dir, 'files', 'files-patients-diagnoses.txt'), rec)
        return {
            'samples': x,
            'labels_fine': torch.from_numpy(y_fine).float() if y_fine is not None else torch.zeros(0),
            'labels_coarse': torch.from_numpy(y_coarse).float() if y_coarse is not None else torch.zeros(0),
            'hea': hea,
            'patient_id': pid or rec
        }


