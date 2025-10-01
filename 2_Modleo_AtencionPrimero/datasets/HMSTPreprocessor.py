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
import os

class HMSTPreprocessor(ECG12Large):  # Extiende
    def __init__(self, *args, wide_feats=True, balance_rare=False, **kwargs):  # Desactivar balance_rare por defecto
        super().__init__(*args, **kwargs)
        self.wide_feats = wide_feats
        self.balance_rare = balance_rare
        self.snomed_api_key = "80697352-209b-4287-802b-c6a6f73fe545"  # Opcional para queries
        
        # Verificar que self.hierarchy esté disponible
        if not hasattr(self, 'hierarchy') or self.hierarchy is None:
            raise ValueError("Hierarchy not loaded. Make sure hierarchy_path is valid and file exists.")
        
        # Usar solo los datos reales de la jerarquía (sin CSV)
        self.snomed_hier = self._create_hierarchy_based_fallback()
        
        if self.balance_rare:
            self._balance_labels()  # SMOTE en init

    def _create_hierarchy_based_fallback(self):
        """Crear jerarquía basada en self.hierarchy existente (datos reales)"""
        hier = {}
        
        # Usar self.hierarchy como base (datos reales)
        if hasattr(self, 'hierarchy') and self.hierarchy is not None:
            # Crear mapeo fine_to_coarse desde coarse_groups
            for coarse_name, fine_codes_list in self.hierarchy.get('coarse_groups', {}).items():
                for fine_code in fine_codes_list:
                    hier[fine_code] = {
                        'coarse': coarse_name, 
                        'snomed_coarse': fine_code,  # Usar el mismo código
                        'is_a': True
                    }
        
        # Construye matriz learnable [num_fine, num_coarse] para loss (1 si is_a)
        num_fine = len(self.fine_codes) if self.fine_codes else 71
        num_coarse = len(self.coarse_names) if self.coarse_names else 10
        hier_matrix = torch.zeros(num_fine, num_coarse)
        
        if hasattr(self, 'fine_codes') and hasattr(self, 'coarse_name_to_idx'):
            for i, fine in enumerate(self.fine_codes):
                if fine in hier and hier[fine]['is_a']:
                    coarse_idx = self.coarse_name_to_idx.get(hier[fine]['coarse'], 0)
                    hier_matrix[i, coarse_idx] = 1.0
        
        self.hier_matrix = nn.Parameter(hier_matrix)  # Learnable en model
        print(f"SNOMED basado en hierarchy: {len(hier)} mappings, matriz {hier_matrix.shape}")
        return hier

    def _balance_labels(self):
        """Balancear labels usando SMOTE - solo si hay suficientes datos"""
        if not self.multilabel:
            print("No es multilabel, saltando balanceo")
            return
            
        try:
            # Obtener datos de labels
            y_fine = self._get_all_y_fine()
            y_coarse = self._get_all_y_coarse()
            
            if len(y_fine) == 0 or len(y_coarse) == 0:
                print("No hay datos de labels para balancear")
                return
                
            print(f"Datos para balancear: {len(y_fine)} muestras, {y_fine.shape[1]} fine labels, {y_coarse.shape[1]} coarse labels")
            
            # Verificar que hay suficientes muestras para SMOTE
            if len(y_fine) < 10:
                print("Muy pocas muestras para SMOTE, saltando balanceo")
                return
                
            # Combinar fine y coarse para SMOTE
            stacked = np.hstack([y_fine, y_coarse[:, :10]])  # Top 10 coarse categories
            
            # SMOTENC necesita X e y por separado
            X = stacked  # Usar los labels como "features" para SMOTE
            y = np.ones(len(X))  # Dummy target para SMOTE
            
            # Aplicar SMOTE
            smote = SMOTENC(categorical_features=range(y_fine.shape[1], y_fine.shape[1] + 10), random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Extraer los labels balanceados
            balanced_fine = X_balanced[:, :y_fine.shape[1]]
            balanced_coarse = X_balanced[:, y_fine.shape[1]:y_fine.shape[1] + 10]
            
            # Actualizar samples con datos balanceados
            self._update_samples_with_balanced_labels(balanced_fine, balanced_coarse)
            
            print(f"Balanceado exitoso: {len(y_fine)} → {len(balanced_fine)} muestras")
            
        except Exception as e:
            print(f"Error en SMOTE: {e}, continuando sin balanceo")

    def _update_samples_with_balanced_labels(self, balanced_fine, balanced_coarse):
        """Actualizar samples con labels balanceados"""
        # Crear nuevos samples con labels balanceados
        new_samples = []
        for i, sample in enumerate(self.samples):
            if i < len(balanced_fine):
                # Mantener la estructura original pero con labels balanceados
                if len(sample) >= 4:
                    hea, mat, _, _ = sample
                    new_samples.append((hea, mat, balanced_fine[i], balanced_coarse[i]))
                else:
                    new_samples.append(sample)
            else:
                new_samples.append(sample)
        
        self.samples = new_samples

    def _get_all_y_fine(self):
        return np.array([rec[2] for rec in self.samples if len(rec)>2])

    def _get_all_y_coarse(self):
        return np.array([rec[3] for rec in self.samples if len(rec)>3])

    def extract_wide_features(self, x):
        if not self.wide_feats:
            return np.zeros(3)
        # HR, RMSSD, entropy en lead II (x[1])
        lead_ii = x[1].numpy() if torch.is_tensor(x) else x[1]
        peaks, _ = find_peaks(lead_ii, distance=self.target_fs//2, height=np.mean(lead_ii)*0.5)
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