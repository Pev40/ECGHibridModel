# Añade a imports
from imblearn.over_sampling import SMOTENC
from scipy.signal import find_peaks
import requests  # Para BioPortal API
from .ecg12large import ECG12Large
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import pandas as pd
import torch.nn as nn

class HMSTPreprocessor(ECG12Large):  # Extiende
    def __init__(self, *args, wide_feats=True, balance_rare=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.wide_feats = wide_feats
        self.balance_rare = balance_rare
        self.snomed_api_key = "80697352-209b-4287-802b-c6a6f73fe545"  # Opcional para queries
        self.snomed_hier = self._load_snomed_hierarchy_from_csv()  
        if self.balance_rare:
            self._balance_labels()  # SMOTE en init

    def _load_snomed_hierarchy(self):
        hier = {}  # Ejemplo: {'AFIB': {'coarse': 'Rhythm', 'snomed_coarse': '426783006', 'is_a': True}}
        # Query BioPortal (simplificado; full con API key)
        url = "https://data.bioontology.org/ontologies/SNOMEDCT/classes"
        params = {'q': 'atrial fibrillation', 'apikey': self.snomed_api_key or 'free'}
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            for item in data.get('collection', []):
                code = item['@id'].split('/')[-1]
                hier[code] = {'coarse': '426783006', 'is_a': True}  # Hardcode para demo; expandir
        except:
            hier = {'AFIB': {'coarse': 'Rhythm', 'snomed_coarse': '426783006', 'is_a': True}}  # Fallback
        # Merge con self.hierarchy
        for fine, coarse in self.hierarchy.get('fine_to_coarse', {}).items():
            if fine not in hier:
                hier[fine] = {'coarse': coarse, 'snomed_coarse': 'default', 'is_a': True}
        return hier

    def _load_snomed_hierarchy_from_csv(self, csv_path='datos/12Large/ConditionNames_SNOMED-CT.csv'):
            # Lee CSV local (ajusta columnas si diferente, e.g., 'code', 'parent', 'name')
            df = pd.read_csv(csv_path)
            hier = {}  # {fine_code: {'coarse_code': str, 'snomed_coarse': int, 'is_a': bool}}
            for _, row in df.iterrows():
                fine = row.get('fine_code', row.get('code', 'UNKNOWN'))  # Ajusta columna
                coarse = row.get('coarse_code', row.get('parent', 'UNKNOWN'))
                snomed_coarse = row.get('snomed_coarse', row.get('snomed_parent', 0))
                is_a = row.get('is_a', True)  # Default True si implica
                hier[fine] = {'coarse': coarse, 'snomed_coarse': snomed_coarse, 'is_a': bool(is_a)}
            # Merge con self.hierarchy si existe
            for fine, coarse in self.hierarchy.get('fine_to_coarse', {}).items():
                if fine not in hier:
                    hier[fine] = {'coarse': coarse, 'snomed_coarse': 0, 'is_a': True}
            self.snomed_hier = hier
            # Construye matriz learnable [num_fine, num_coarse] para loss (1 si is_a)
            num_fine = len(self.fine_codes) if self.fine_codes else 71
            num_coarse = len(self.coarse_names) if self.coarse_names else 10
            hier_matrix = torch.zeros(num_fine, num_coarse)
            for i, fine in enumerate(self.fine_codes):
                if fine in hier and hier[fine]['is_a']:
                    coarse_idx = self.coarse_name_to_idx.get(hier[fine]['coarse'], 0)
                    hier_matrix[i, coarse_idx] = 1.0
            self.hier_matrix = nn.Parameter(hier_matrix)  # Learnable en model
            print(f"SNOMED cargado de CSV: {len(hier)} mappings, matriz {hier_matrix.shape}")
            return hier


    def _balance_labels(self):
        # SMOTE multi-label approx: Trata coarse como cat, fine como num
        if not self.multilabel:
            return
        # Stack y_fine + y_coarse para SMOTE
        stacked = np.hstack([self._get_all_y_fine(), self._get_all_y_coarse()[:, :10]])  # Top 10 coarse cat
        smote = SMOTENC(categorical_features=range(71, 81), random_state=42)  # Fine num, coarse cat
        balanced = smote.fit_resample(stacked)[0]
        self.balanced_samples = balanced[:71]  # Solo fine; reasigna a samples
        # Rebalance self.samples indices (simplificado; full: reordena)
        print(f"Balanceado: {len(self.samples)} → {len(balanced)}")

    def _get_all_y_fine(self):
        return np.array([rec[2] for rec in self.samples if len(rec)>2])

    def _get_all_y_coarse(self):
        return np.array([rec[3] for rec in self.samples if len(rec)>3])

    def extract_wide_features(self, x):
        if not self.wide_feats:
            return np.zeros(3)
        # HR, RMSSD, entropy en lead II (x[1])
        lead_ii = x[1].numpy() if torch.is_tensor(x) else x[1]
        peaks, _ = find_peaks(lead_ii, distance=FS_TARGET//2, height=np.mean(lead_ii)*0.5)
        if len(peaks) < 2:
            return np.array([60.0, 0.0, 0.0])  # Default
        rr = np.diff(peaks) / self.target_fs * 60  # ms to BPM
        hr = 60 / np.mean(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr)**2))
        hist, _ = np.histogram(lead_ii, bins=50, density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        return np.array([hr, rmssd, entropy])

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        x = item['samples']  # [12, 7500] ahora
        self.sequence_len = 7500  # Override
        # Wide feats
        wide = torch.tensor(self.extract_wide_features(x)).float()
        item['wide_feats'] = wide
        # SNOMED embed (one-hot coarse para CLS)
        if self.snomed_hier:
            fine_labels = item.get('labels_fine', torch.zeros(71))
            active_snomed = []
            for i, fine in enumerate(self.fine_codes):
                if fine_labels[i] > 0 and fine in self.snomed_hier:
                    snomed_c = self.snomed_hier[fine]['snomed_coarse']
                    active_snomed.append(snomed_c % 10)  # Mod 10 para dim=10; ajusta si full codes
            if active_snomed:
                snomed_embed = F.one_hot(torch.tensor(active_snomed), 10).float().mean(0)  # Avg one-hot
            else:
                snomed_embed = torch.zeros(10)
            item['snomed_embed'] = snomed_embed
        return item

# Uso extendido para PhysioNet/KCL (adaptar glob para /physionet/challenge-2020)
ds = HMSTPreprocessor(root_dir='datos/12Large', sequence_len=7500, multilabel=True, hierarchy_path='hierarchy.json', target_fs=500.0)
dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=16)  # RTX maneja fácil
for batch in dl:
    print(batch['samples'].shape)  # [128,12,7500]
    break
wandb.log({"dataset_size": len(ds), "wide_feats_mean_hr": batch['wide_feats'][:,0].mean()})