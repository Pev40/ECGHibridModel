import os
import json
import math
import argparse
from glob import glob
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
try:
    from torch.amp import autocast as _autocast_new, GradScaler  # PyTorch >=2.0
    def autocast_cuda(enabled):
        return _autocast_new('cuda', enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler  # Fallback
    def autocast_cuda(enabled):
        return _autocast_old(enabled=enabled)
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
try:
    from ModeloNuevo.v2 import ECGHybridVariableBeforeBiTransV2
    _use_v2 = True
except Exception:
    from ModeloNuevo.v1 import ECGHybridVariableBeforeBiTrans
    _use_v2 = False
from datasets.ecg12large import ECG12Large, extract_patient_id
from datasets import PTBXL, INCART12Lead
from losses.asymmetric_loss import AsymmetricLossMultiLabel

# Utilidades para verificación y generación de jerarquía
try:
    from scripts.check_data_integrity import scan_and_report as _scan_mat_integrity
except Exception:
    _scan_mat_integrity = None
try:
    from scripts.generate_labels_hierarchy import generate_labels_hierarchy as _gen_wfdb_hierarchy
except Exception:
    _gen_wfdb_hierarchy = None
try:
    from scripts.generate_ptbxl_labels_hierarchy import generate_ptbxl_hierarchy as _gen_ptbxl_hierarchy
except Exception:
    _gen_ptbxl_hierarchy = None


def set_seed(seed=42, deterministic=False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def build_patient_splits(root, seed=42, train_frac=0.7, val_frac=0.15):
    hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    pairs = [(hea, os.path.splitext(hea)[0] + '.mat') for hea in hea_files]
    files = [hea for hea, mat in pairs if os.path.exists(mat)]
    # agrupar por paciente
    pid_to_files = {}
    for hea in files:
        pid = extract_patient_id(hea, root)
        pid_to_files.setdefault(pid, []).append(hea)
    rng = np.random.default_rng(seed)
    pids = list(pid_to_files.keys())
    rng.shuffle(pids)
    n = len(pids)
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)
    tr_pids = set(pids[:n_tr])
    va_pids = set(pids[n_tr:n_tr+n_va])
    te_pids = set(pids[n_tr+n_va:])
    tr_files, va_files, te_files = [], [], []
    for pid, flist in pid_to_files.items():
        if pid in tr_pids:
            tr_files.extend(flist)
        elif pid in va_pids:
            va_files.extend(flist)
        else:
            te_files.extend(flist)
    return tr_files, va_files, te_files


def mixup_collate_fn(batch, alpha=0.2):
    if alpha == 0 or len(batch) < 2:
        # Stack normal si no mixup o batch pequeño
        samples = torch.stack([b['samples'] for b in batch])
        labels_coarse = torch.stack([b['labels_coarse'] for b in batch])
        labels_fine = torch.stack([b['labels_fine'] for b in batch])
        return {'samples': samples, 'labels_coarse': labels_coarse, 'labels_fine': labels_fine}
    
    # Mixup: permuta y mezcla
    samples = torch.stack([b['samples'] for b in batch])
    labels_coarse = torch.stack([b['labels_coarse'] for b in batch])
    labels_fine = torch.stack([b['labels_fine'] for b in batch])
    
    idx = torch.randperm(len(samples))
    lam = np.random.beta(alpha, alpha)
    
    samples_mix = lam * samples + (1 - lam) * samples[idx]
    labels_coarse_mix = lam * labels_coarse + (1 - lam) * labels_coarse[idx]
    labels_fine_mix = lam * labels_fine + (1 - lam) * labels_fine[idx]
    
    return {
        'samples': samples_mix,
        'labels_coarse': labels_coarse_mix,
        'labels_fine': labels_fine_mix
    }


def build_loaders(dataset_name, sequence_len, hierarchy_path, batch_size, workers, cache_dir=None,
                 target_fs=500.0, bandpass_hz=(0.5, 45.0), notch_hz=None, seed=42,
                 use_sampler=True, sampler_power=1.0,
                 sampler_power_rare=1.0, rare_class_thresh=0.01,
                 aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                 aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
                 aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0,
                 smoke_test=False, smoke_n=256, mixup_alpha=0.0):
    dataset_name = str(dataset_name).lower()
    if dataset_name in ('12large', 'ecg12large', 'wfdbrecords'):
        root = os.path.join('datos', '12Large', 'WFDBRecords')
        tr_files, va_files, te_files = build_patient_splits(root, seed=seed)
        tr_ds = ECG12Large(root, sequence_len=sequence_len, files=tr_files[:smoke_n] if smoke_test else tr_files,
                              multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                              random_crop=True, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=False,
                              aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max,
                              aug_lead_drop_prob=aug_lead_drop_prob, aug_amp_scale_min=aug_amp_scale_min,
                              aug_amp_scale_max=aug_amp_scale_max,
                              aug_lead_noise_scale_max=aug_lead_noise_scale_max, aug_time_warp_max=aug_time_warp_max, aug_time_warp_p=aug_time_warp_p)
        va_ds = ECG12Large(root, sequence_len=sequence_len, files=va_files[:max(1, smoke_n//4)] if smoke_test else va_files,
                              multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                              random_crop=False, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=True)
        te_ds = ECG12Large(root, sequence_len=sequence_len, files=te_files[:max(1, smoke_n//4)] if smoke_test else te_files,
                              multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                              random_crop=False, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=True)
    elif dataset_name in ('ptbxl', 'ptb-xl') and PTBXL is not None:
        root = os.path.join('datos', 'PTBXL')
        tr_ds = PTBXL(root, split='train', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                      target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=True, eval_mode=False,
                      aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max, aug_lead_drop_prob=aug_lead_drop_prob,
                      aug_amp_scale_min=aug_amp_scale_min, aug_amp_scale_max=aug_amp_scale_max,
                      aug_lead_noise_scale_max=aug_lead_noise_scale_max, aug_time_warp_max=aug_time_warp_max, aug_time_warp_p=aug_time_warp_p)
        if smoke_test and hasattr(tr_ds, 'samples'):
            tr_ds.samples = tr_ds.samples[:smoke_n]
        va_ds = PTBXL(root, split='val', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                      target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=False, eval_mode=True)
        if smoke_test and hasattr(va_ds, 'samples'):
            va_ds.samples = va_ds.samples[:max(1, smoke_n//4)]
        te_ds = PTBXL(root, split='test', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                      target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=False, eval_mode=True)
        if smoke_test and hasattr(te_ds, 'samples'):
            te_ds.samples = te_ds.samples[:max(1, smoke_n//4)]
    elif dataset_name in ('incart', 'stpetersburg', 'stpetersburgincart12leadarrhythmiadatabase') and INCART12Lead is not None:
        root = os.path.join('datos', 'StPetersburgIncart12LeadArrhythmiaDatabase')
        tr_ds = INCART12Lead(root, split='train', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                             target_fs=257.0, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=True, eval_mode=False,
                             aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max, aug_lead_drop_prob=aug_lead_drop_prob,
                             aug_amp_scale_min=aug_amp_scale_min, aug_amp_scale_max=aug_amp_scale_max,
                             aug_lead_noise_scale_max=aug_lead_noise_scale_max, aug_time_warp_max=aug_time_warp_max, aug_time_warp_p=aug_time_warp_p)
        if smoke_test and hasattr(tr_ds, 'samples'):
            tr_ds.samples = tr_ds.samples[:smoke_n]
        va_ds = INCART12Lead(root, split='val', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                             target_fs=257.0, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=False, eval_mode=True)
        if smoke_test and hasattr(va_ds, 'samples'):
            va_ds.samples = va_ds.samples[:max(1, smoke_n//4)]
        te_ds = INCART12Lead(root, split='test', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                             target_fs=257.0, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=False, eval_mode=True)
        if smoke_test and hasattr(te_ds, 'samples'):
            te_ds.samples = te_ds.samples[:max(1, smoke_n//4)]
    elif dataset_name in ('georgia', 'georgia12leadecgdatabase'):
        root = os.path.join('datos', 'Georgia12LeadECGDatabase')
        hea_files = glob(os.path.join(root, '*.hea'))
        # split por paciente usando utilidad existente
        tr_files, va_files, te_files = build_patient_splits(root, seed=seed)
        tr_ds = ECG12Large(root, sequence_len=sequence_len, files=tr_files[:smoke_n] if smoke_test else tr_files,
                              multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                              random_crop=True, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=False,
                              aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max,
                              aug_lead_drop_prob=aug_lead_drop_prob, aug_amp_scale_min=aug_amp_scale_min,
                              aug_amp_scale_max=aug_amp_scale_max)
        va_ds = ECG12Large(root, sequence_len=sequence_len, files=va_files[:max(1, smoke_n//4)] if smoke_test else va_files,
                              multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                              random_crop=False, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=True)
        te_ds = ECG12Large(root, sequence_len=sequence_len, files=te_files[:max(1, smoke_n//4)] if smoke_test else te_files,
                              multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                              random_crop=False, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=True)
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = device.type == 'cuda'

    # WeightedRandomSampler según frecuencia de etiquetas fine en el dataset de entrenamiento
    y_fine_list = [s[2] for s in tr_ds.samples]
    sampler = None
    shuffle_train = True
    # Desactivar sampler para datasets sin etiquetas (p. ej., INCART) o si no hay datos suficientes
    if dataset_name in ('incart', 'stpetersburg', 'stpetersburgincart12leadarrhythmiadatabase'):
        sampler = None
        shuffle_train = True
    elif use_sampler and len(y_fine_list) > 0 and isinstance(y_fine_list[0], np.ndarray):
        y_mat = np.stack(y_fine_list, axis=0)
        # si todas las etiquetas son cero, no usar sampler
        if np.sum(y_mat) <= 0:
            sampler = None
            shuffle_train = True
        else:
            class_freq = y_mat.mean(axis=0) + 1e-6
            # Potenciar clases raras con mayor exponente
            base_power = float(max(0.0, sampler_power))
            extra_power = float(max(1.0, sampler_power_rare))
            is_rare = (class_freq < float(max(1e-6, rare_class_thresh)))
            per_class_power = np.where(is_rare, base_power * extra_power, base_power)
            inv_freq = (1.0 / class_freq) ** per_class_power
            inv_freq = inv_freq / max(1e-12, inv_freq.sum())
            sample_w = (y_mat * inv_freq).sum(axis=1)
            # Validar pesos
            if not np.isfinite(sample_w).all() or sample_w.sum() <= 0:
                sampler = None
                shuffle_train = True
            else:
                sample_w = sample_w / (sample_w.mean() + 1e-8)
                sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_w).float(), num_samples=len(tr_ds), replacement=True)
                shuffle_train = False
    collate_fn_tr = mixup_collate_fn if mixup_alpha > 0 else None
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle_train, sampler=sampler, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None,
                       **({'collate_fn': collate_fn_tr} if collate_fn_tr else {}))
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    return tr_dl, va_dl, te_dl


def evaluate(model, dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
             use_dice_on_fine=False, dice_weight=0.5, save_attn=False, attn_dir=None, attn_viz_max_batches=0):
    device = next(model.parameters()).device
    model.eval()
    loss_sum = 0.0
    count = 0
    all_y_fine = []
    all_p_fine = []
    saved_batches = 0
    with torch.no_grad():
        for batch in dl:
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            if save_attn and saved_batches < int(attn_viz_max_batches or 0):
                try:
                    logits_coarse, logits_fine, attn = model.forward_with_attn(x)
                    saved_batches += 1
                    if attn is not None and attn_dir is not None:
                        os.makedirs(attn_dir, exist_ok=True)
                        attn_cpu = attn.detach().cpu().mean(dim=1).numpy()
                        for i in range(min(attn_cpu.shape[0], 4)):
                            np.save(os.path.join(attn_dir, f'attn_b{i}_batch{saved_batches}.npy'), attn_cpu[i])
                except Exception:
                    logits_coarse, logits_fine = model(x)
            else:
                logits_coarse, logits_fine = model(x)
            loss_coarse = base_loss_fn(logits_coarse, y_coarse)
            if use_dice_on_fine:
                base_l = base_loss_fn(logits_fine, y_fine)
                dice_l = dice_loss_multilabel(logits_fine, y_fine)
                loss_fine = (1.0 - float(dice_weight)) * base_l + float(dice_weight) * dice_l
            else:
                loss_fine = base_loss_fn(logits_fine, y_fine)
            # consistencia basada en predicciones
            p_fine = torch.sigmoid(logits_fine)
            max_per_group_pred = []
            for g, codes in coarse_groups.items():
                idxs = [fine_code_to_idx.get(c, None) for c in codes]
                idxs = [i for i in idxs if i is not None]
                if len(idxs) == 0:
                    max_per_group_pred.append(torch.zeros(y_fine.shape[0], device=device))
                else:
                    max_per_group_pred.append(p_fine[:, idxs].max(dim=1).values)
            max_per_group_pred = torch.stack(max_per_group_pred, dim=1)
            cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group_pred.detach())
            loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons
            loss_sum += loss.item() * x.size(0)
            count += x.size(0)
            if compute_stats:
                all_y_fine.append(y_fine.cpu())
                all_p_fine.append(torch.sigmoid(logits_fine).cpu())
    mean_loss = loss_sum / max(1, count)
    if not compute_stats or len(all_y_fine) == 0:
        return mean_loss, {}
    y_true = torch.cat(all_y_fine, dim=0).numpy()
    y_prob = torch.cat(all_p_fine, dim=0).numpy()
    c = y_true.shape[1]
    aurocs = []
    auprcs = []
    f1s = []
    for i in range(c):
        yi = y_true[:, i]
        pi = y_prob[:, i]
        pos = yi.sum() > 0
        neg = (1 - yi).sum() > 0
        if pos and neg:
            try:
                aurocs.append(roc_auc_score(yi, pi))
            except Exception:
                aurocs.append(np.nan)
        else:
            aurocs.append(np.nan)
        if pos:
            try:
                auprcs.append(average_precision_score(yi, pi))
            except Exception:
                auprcs.append(np.nan)
        else:
            auprcs.append(np.nan)
        if pos:
            yi_pred = (pi >= 0.5).astype(np.float32)
            try:
                f1s.append(f1_score(yi, yi_pred, zero_division=0))
            except Exception:
                f1s.append(np.nan)
        else:
            f1s.append(np.nan)
    def nanmean_safe(arr):
        return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float('nan')
    metrics = {}
    try:
        auroc_micro = roc_auc_score(y_true, y_prob, average='micro')
    except Exception:
        auroc_micro = float('nan')
    try:
        auprc_micro = average_precision_score(y_true, y_prob, average='micro')
    except Exception:
        auprc_micro = float('nan')
    metrics.update({
        'auroc_macro': nanmean_safe(np.array(aurocs, dtype=float)),
        'auprc_macro': nanmean_safe(np.array(auprcs, dtype=float)),
        'f1_macro': nanmean_safe(np.array(f1s, dtype=float)),
        'auroc_micro': float(auroc_micro),
        'auprc_micro': float(auprc_micro),
    })
    try:
        prevalences = y_true.mean(axis=0)
        order = np.argsort(prevalences)
        topk = order[:min(10, len(order))]
        metrics['rare_topk_indices'] = topk.tolist()
        metrics['rare_topk_prevalences'] = prevalences[topk].tolist()
        metrics['rare_topk_f1'] = [float(f1s[i]) for i in topk]
    except Exception:
        pass
    metrics['y_true'] = y_true
    metrics['y_prob'] = y_prob
    return mean_loss, metrics


def focal_loss_multi_label(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    # logits, targets: (N, C)
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def dice_loss_multilabel(logits, targets, eps=1e-7, reduction='mean'):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=0)
    cardinality = probs.sum(dim=0) + targets.sum(dim=0)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    loss = 1.0 - dice
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def get_kfold_loaders(dataset_name, sequence_len, hierarchy_path, batch_size, workers, cache_dir=None,
                      target_fs=500.0, bandpass_hz=(0.5, 45.0), notch_hz=None, seed=42,
                      use_sampler=True, sampler_power=1.0, sampler_power_rare=1.0, rare_class_thresh=0.01,
                      aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                      aug_amp_scale_min=1.0, aug_amp_scale_max=1.0, aug_lead_noise_scale_max=0.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0,
                      smoke_test=False, smoke_n=256, n_folds=1, kfold_seed=42, mixup_alpha=0.0):
    if n_folds == 1:
        return [build_loaders(dataset_name, sequence_len, hierarchy_path, batch_size, workers, cache_dir,
                              target_fs, bandpass_hz, notch_hz, seed, use_sampler, sampler_power, sampler_power_rare, rare_class_thresh,
                              aug_jitter_std, aug_shift_max, aug_lead_drop_prob, aug_amp_scale_min, aug_amp_scale_max, aug_lead_noise_scale_max, aug_time_warp_max, aug_time_warp_p,
                              smoke_test, smoke_n, mixup_alpha)]
    
    # Carga full_ds
    if dataset_name in ('12large', 'georgia'):
        root = os.path.join('datos', '12Large' if '12large' in dataset_name else 'Georgia12LeadECGDatabase', 'WFDBRecords' if '12large' in dataset_name else '')
        all_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
        full_ds = ECG12Large(root, sequence_len=sequence_len, files=all_files[:smoke_n * n_folds] if smoke_test else all_files,
                             multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                             random_crop=not smoke_test, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=False,
                             aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max, aug_lead_drop_prob=aug_lead_drop_prob,
                             aug_amp_scale_min=aug_amp_scale_min, aug_amp_scale_max=aug_amp_scale_max, aug_lead_noise_scale_max=aug_lead_noise_scale_max, aug_time_warp_max=aug_time_warp_max, aug_time_warp_p=aug_time_warp_p)
        y_strat = np.array([s[2][:, 0].item() if s[2].sum() > 0 else 0 for s in full_ds.samples])  # Primer coarse
    elif dataset_name == 'ptbxl':
        root = os.path.join('datos', 'PTBXL')
        # Custom full load: combina train/val/test samples
        full_samples = PTBXL(root, split='train').samples + PTBXL(root, split='val').samples + PTBXL(root, split='test').samples
        full_ds = PTBXL(root, sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                        target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=True, eval_mode=False,
                        aug_jitter_std=aug_jitter_std )  # Set samples=full_samples
        y_strat = full_ds.labels_coarse[:, 0].numpy() if hasattr(full_ds, 'labels_coarse') else np.zeros(len(full_ds))
    elif dataset_name == 'incart':
        root = os.path.join('datos', 'StPetersburgIncart12LeadArrhythmiaDatabase')
        # INCART no soporta 'all'; combinamos splits manualmente
        tr_all = INCART12Lead(root, split='train', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                               target_fs=257.0, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=True, eval_mode=False,
                               aug_jitter_std=aug_jitter_std)
        va_all = INCART12Lead(root, split='val', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                               target_fs=257.0, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=True, eval_mode=False,
                               aug_jitter_std=aug_jitter_std)
        te_all = INCART12Lead(root, split='test', sequence_len=sequence_len, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                               target_fs=257.0, bandpass_hz=bandpass_hz, notch_hz=notch_hz, random_crop=True, eval_mode=False,
                               aug_jitter_std=aug_jitter_std)
        from torch.utils.data import ConcatDataset
        full_ds = ConcatDataset([tr_all, va_all, te_all])
        y_strat = full_ds.labels_coarse[:, 0].numpy() if hasattr(full_ds, 'labels_coarse') else np.zeros(len(full_ds))
    else:
        raise ValueError(f"Dataset no soportado para k-fold: {dataset_name}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=kfold_seed)
    indices = list(range(len(full_ds)))
    folds = list(skf.split(indices, y_strat))

    fold_loaders = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        tr_ds_fold = Subset(full_ds, train_idx)
        va_ds_fold = Subset(full_ds, val_idx)
        # Test: Rota 20% de train como test
        test_size = len(train_idx) // 5
        test_idx = train_idx[:test_size]
        te_ds_fold = Subset(full_ds, test_idx)

        # DLs con params compartidos
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pin = device.type == 'cuda'
        collate_fn_tr = mixup_collate_fn if mixup_alpha > 0 else None
        tr_dl_fold = DataLoader(tr_ds_fold, batch_size=batch_size, shuffle=True, num_workers=workers,
                                pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None,
                                **({'collate_fn': collate_fn_tr} if collate_fn_tr else {}))
        va_dl_fold = DataLoader(va_ds_fold, batch_size=batch_size, shuffle=False, num_workers=workers,
                                pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
        te_dl_fold = DataLoader(te_ds_fold, batch_size=batch_size, shuffle=False, num_workers=workers,
                                pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
        fold_loaders.append((tr_dl_fold, va_dl_fold, te_dl_fold))

    return fold_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Épocas sin mejora en val_loss para detener el entrenamiento. 0 para desactivar.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4, help='Mejora mínima en val_loss para resetear la paciencia.')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=os.path.join('datos','pt_cache'))
    parser.add_argument('--exp_dir', type=str, default=os.path.join('experiments_logs','full_run'))
    parser.add_argument('--target_fs', type=float, default=500.0)
    parser.add_argument('--bandpass_low', type=float, default=0.5)
    parser.add_argument('--bandpass_high', type=float, default=45.0)
    parser.add_argument('--notch_hz', type=int, default=0, help='50 o 60; 0 para desactivar')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attn_dropout', type=float, default=None, help='Si se especifica, dropout específico para atención variable')
    parser.add_argument('--trans_dropout', type=float, default=0.1, help='Dropout en TransformerEncoderLayer')
    parser.add_argument('--gamma_pos', type=float, default=0.0)
    parser.add_argument('--gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_clip', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Smoothing para etiquetas multilabel (aplicado antes de la pérdida)')
    parser.add_argument('--loss_type', type=str, default='asl', choices=['asl','focal'], help='Tipo de pérdida para multilabel')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--sampler_power', type=float, default=1.0)
    parser.add_argument('--sampler_power_rare', type=float, default=1.0, help='Potencia extra para clases raras (<1%)')
    parser.add_argument('--rare_class_thresh', type=float, default=0.01, help='Umbral de rareza por frecuencia')
    parser.add_argument('--no_sampler', action='store_true')
    # Augmentaciones
    parser.add_argument('--aug_jitter_std', type=float, default=0.0)
    parser.add_argument('--aug_shift_max', type=int, default=0)
    parser.add_argument('--aug_lead_drop_prob', type=float, default=0.0)
    parser.add_argument('--aug_amp_scale_min', type=float, default=1.0)
    parser.add_argument('--aug_amp_scale_max', type=float, default=1.0)
    parser.add_argument('--aug_lead_noise_scale_max', type=float, default=1.0, help='Escala max aleatoria por derivación para jitter')
    parser.add_argument('--aug_time_warp_max', type=float, default=0.0, help='Factor máx de warping temporal (e.g., 0.05)')
    parser.add_argument('--aug_time_warp_p', type=float, default=0.0, help='Probabilidad de aplicar time-warp')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Alpha de Beta para mixup temporal (0 desactiva)')
    parser.add_argument('--mixup_p', type=float, default=0.0, help='Probabilidad de aplicar mixup por batch')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--scheduler_metric', type=str, default='auroc_macro', choices=['val_loss','auroc_macro'], help='Métrica para ReduceLROnPlateau')
    parser.add_argument('--cosine_after_plateau', action='store_true', help='Activar CosineAnnealingWarmRestarts tras ReduceLROnPlateau')
    parser.add_argument('--cosine_t0', type=int, default=10)
    parser.add_argument('--cosine_tmult', type=int, default=1)
    parser.add_argument('--cosine_eta_min_scale', type=float, default=0.01, help='eta_min = lr * scale')
    # Stockwell/Swin controles
    parser.add_argument('--freq_bins_low', type=int, default=1, help='Bin inicial (>=1) para S-Transform')
    parser.add_argument('--freq_bins_high', type=int, default=65, help='Bin final (exclusivo) para S-Transform')
    parser.add_argument('--swin_freeze_stages', type=int, default=0, help='Número de stages iniciales del Swin a congelar (0=no congelar)')
    parser.add_argument('--dataset', type=str, default='12large', choices=['12large','ptbxl','georgia','incart'], help='Dataset a usar')
    parser.add_argument('--no_data_check', action='store_true', help='Desactivar verificación de integridad .mat al inicio')
    parser.add_argument('--no_auto_hierarchy', action='store_true', help='Desactivar generación automática de labels_hierarchy.json si falta')
    parser.add_argument('--smoke_test', action='store_true', help='Activar modo rápido con pocos ejemplos')
    parser.add_argument('--smoke_n', type=int, default=256, help='Máx ejemplos train en smoke test')
    parser.add_argument('--use_dice_on_fine', action='store_true', help='Añadir Dice a la pérdida de fine')
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Peso de Dice en la combinación híbrida')
    parser.add_argument('--save_attn_viz', action='store_true', help='Guardar mapas de atención durante validación')
    parser.add_argument('--attn_viz_max_batches', type=int, default=2)
    parser.add_argument('--n_folds', type=int, default=1, help='Número de folds para k-fold estratificado (1=single split)')
    parser.add_argument('--kfold_seed', type=int, default=42, help='Seed para k-fold')
    args = parser.parse_args()





    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    print('Device:', device)
    if device.type == 'cuda':
        try:
            print('GPU:', torch.cuda.get_device_name(0))
        except Exception:
            pass

    # Definir roots por dataset
    if args.dataset == '12large':
        data_root = os.path.join('datos', '12Large', 'WFDBRecords')
        hierarchy_path = os.path.join('datos', '12Large', 'labels_hierarchy.json')
    elif args.dataset == 'georgia':
        data_root = os.path.join('datos', 'Georgia12LeadECGDatabase')
        hierarchy_path = os.path.join('datos', 'Georgia12LeadECGDatabase', 'labels_hierarchy.json')
    elif args.dataset == 'ptbxl':
        data_root = os.path.join('datos', 'PTBXL')
        hierarchy_path = os.path.join('datos', 'PTBXL', 'labels_hierarchy.json')
    else:  # incart
        data_root = os.path.join('datos', 'StPetersburgIncart12LeadArrhythmiaDatabase')
        hierarchy_path = os.path.join('datos', 'StPetersburgIncart12LeadArrhythmiaDatabase', 'labels_hierarchy.json')

    # Verificación de datos (opcional por flag)
    if not args.no_data_check:
        if _scan_mat_integrity is None:
            print('Aviso: verificador de integridad no disponible (scripts/check_data_integrity.py).')
        else:
            print(f'Iniciando verificación de integridad en {data_root} ...')
            try:
                _ = _scan_mat_integrity(data_root, report_path=os.path.join('datos', f'problematic_files_{args.dataset}.txt'))
            except Exception as e:
                print(f'Aviso: la verificación de integridad falló: {e}')

    # Asegurar jerarquía por dataset
    if not os.path.exists(hierarchy_path) and not args.no_auto_hierarchy:
        print(f'No se encontró jerarquía en {hierarchy_path}. Intentando generarla automáticamente...')
        try:
            if args.dataset == 'ptbxl' and _gen_ptbxl_hierarchy is not None:
                _gen_ptbxl_hierarchy(data_root)
            elif _gen_wfdb_hierarchy is not None:
                _gen_wfdb_hierarchy(data_root, hierarchy_path, top_k=30)
        except Exception as e:
            print(f'Aviso: no se pudo generar jerarquía automáticamente: {e}')

    # fallback si no existe la jerarquía específica tras intentar generar
    if not os.path.exists(hierarchy_path):
        fallback_h = os.path.join('datos', '12Large', 'labels_hierarchy.json')
        print(f"No se encontró jerarquía en {hierarchy_path}. Usando fallback: {fallback_h}")
        hierarchy_path = fallback_h
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    coarse_groups = hier['coarse_groups']
    fine_code_to_idx = {c:i for i,c in enumerate(fine_codes)}

    # K-FOLD branch
    if args.n_folds > 1:
        fold_loaders = get_kfold_loaders(
            args.dataset, args.sequence_len, hierarchy_path, args.batch_size, args.workers,
            cache_dir=args.cache_dir,
            target_fs=args.target_fs,
            bandpass_hz=(args.bandpass_low, args.bandpass_high),
            notch_hz=(args.notch_hz if args.notch_hz in (50,60) else None),
            seed=args.seed,
            use_sampler=(not args.no_sampler),
            sampler_power=args.sampler_power,
            sampler_power_rare=args.sampler_power_rare,
            rare_class_thresh=args.rare_class_thresh,
            aug_jitter_std=args.aug_jitter_std,
            aug_shift_max=args.aug_shift_max,
            aug_lead_drop_prob=args.aug_lead_drop_prob,
            aug_amp_scale_min=args.aug_amp_scale_min,
            aug_amp_scale_max=args.aug_amp_scale_max,
            aug_lead_noise_scale_max=args.aug_lead_noise_scale_max,
            aug_time_warp_max=args.aug_time_warp_max,
            aug_time_warp_p=args.aug_time_warp_p,
            smoke_test=args.smoke_test,
            smoke_n=args.smoke_n,
            n_folds=args.n_folds,
            kfold_seed=args.kfold_seed,
            mixup_alpha=args.mixup_alpha
        )

        exp_root = os.path.join(args.exp_dir, args.dataset)
        os.makedirs(exp_root, exist_ok=True)
        all_test_metrics = []

        for fold_i, (tr_dl, va_dl, te_dl) in enumerate(fold_loaders):
            print(f"\n=== Fold {fold_i+1}/{args.n_folds} ===")

            # modelo por fold
            configs = type('Cfg', (), dict(
                input_channels=12, sequence_len=args.sequence_len, kernel_size=8, stride=1, dropout=args.dropout,
                attn_dropout=(args.attn_dropout if args.attn_dropout is not None else args.dropout),
                trans_dropout=args.trans_dropout,
                mid_channels=64, final_out_channels=128, trans_dim=64, num_heads=4, num_leads=12,
                num_fine=len(fine_codes), num_coarse=len(coarse_groups),
                stockwell_freq_low=max(1, int(args.freq_bins_low)), stockwell_freq_high=max(2, int(args.freq_bins_high)),
                swin_freeze_stages=max(0, int(args.swin_freeze_stages))
            ))
            if _use_v2:
                model = ECGHybridVariableBeforeBiTransV2(configs, {"feature_dim": 128}).to(device)
            else:
                model = ECGHybridVariableBeforeBiTrans(configs, {"feature_dim": 128}).to(device)

            # optim / schedulers por fold
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sched_mode = 'min' if args.scheduler_metric == 'val_loss' else 'max'
            scheduler = ReduceLROnPlateau(opt, mode=sched_mode, factor=0.5, patience=5, threshold=1e-4, cooldown=0, min_lr=1e-6)
            cosine_sched = CosineAnnealingWarmRestarts(opt, T_0=max(1, args.cosine_t0), T_mult=max(1, args.cosine_tmult), eta_min=args.lr * float(max(0.0, args.cosine_eta_min_scale))) if args.cosine_after_plateau else None

            # criterio
            if args.loss_type == 'asl':
                base_loss_fn = AsymmetricLossMultiLabel(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, clip=args.asl_clip)
            else:
                def base_loss_fn(logits, targets):
                    return focal_loss_multi_label(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma)
            scaler = GradScaler(enabled=args.mixed_precision and device.type=='cuda')

            # logs por fold
            fold_dir = os.path.join(exp_root, f'fold_{fold_i}')
            os.makedirs(fold_dir, exist_ok=True)
            fold_log_path = os.path.join(fold_dir, 'train_log.csv')
            with open(fold_log_path, 'w', encoding='utf-8') as f:
                f.write('epoch,train_loss,val_loss,val_auroc_macro,val_auprc_macro,val_f1_macro,lr\n')

            best_val = math.inf
            epochs_no_improve = 0
            max_epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
            for epoch in range(1, max_epochs+1):
                model.train()
                pbar = tqdm(tr_dl, desc=f'Fold {fold_i+1} - Epoch {epoch}/{args.epochs}')
                opt.zero_grad(set_to_none=True)
                train_sum = 0.0
                train_n = 0
                for step, batch in enumerate(pbar, start=1):
                    x = batch['samples'].to(device, non_blocking=True)
                    y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
                    y_fine = batch['labels_fine'].to(device, non_blocking=True)

                    # Mixup del loop desactivado si collate mixup activo
                    use_collate_mixup = getattr(args, 'mixup_alpha', 0.0) > 0.0
                    if use_collate_mixup:
                        y_coarse_mix = y_coarse
                        y_fine_mix = y_fine
                    else:
                        use_mixup = (getattr(args, 'mixup_alpha', 0.0) > 0.0 and np.random.rand() < getattr(args, 'mixup_p', 0.0))
                        if use_mixup:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                            perm = torch.randperm(x.size(0), device=x.device)
                            x = lam * x + (1.0 - lam) * x[perm]
                            y_coarse_mix = lam * y_coarse + (1.0 - lam) * y_coarse[perm]
                            y_fine_mix = lam * y_fine + (1.0 - lam) * y_fine[perm]
                        else:
                            y_coarse_mix = y_coarse
                            y_fine_mix = y_fine

                    # Label smoothing opcional
                    if float(getattr(args, 'label_smoothing', 0.0)) > 0.0:
                        eps = float(args.label_smoothing)
                        y_coarse_mix = (1.0 - eps) * y_coarse_mix + 0.5 * eps
                        y_fine_mix = (1.0 - eps) * y_fine_mix + 0.5 * eps

                    with autocast_cuda(args.mixed_precision and device.type=='cuda'):
                        # Permitir pérdidas auxiliares si el modelo lo soporta
                        if hasattr(model, 'forward_with_aux'):
                            logits_coarse, logits_fine, x_coarse_vec, x_fine_vec = model.forward_with_aux(x)
                        else:
                            logits_coarse, logits_fine = model(x)
                        loss_coarse = base_loss_fn(logits_coarse, y_coarse_mix)
                        if getattr(args, 'use_dice_on_fine', False):
                            base_l = base_loss_fn(logits_fine, y_fine_mix)
                            dice_l = dice_loss_multilabel(logits_fine, y_fine_mix)
                            loss_fine = (1.0 - float(getattr(args, 'dice_weight', 0.5))) * base_l + float(getattr(args, 'dice_weight', 0.5)) * dice_l
                        else:
                            loss_fine = base_loss_fn(logits_fine, y_fine_mix)
                        # consistencia
                        p_fine = torch.sigmoid(logits_fine)
                        max_per_group_pred = []
                        for g, codes in coarse_groups.items():
                            idxs = [fine_code_to_idx.get(c, None) for c in codes]
                            idxs = [i for i in idxs if i is not None]
                            if len(idxs) == 0:
                                max_per_group_pred.append(torch.zeros(y_fine.shape[0], device=device))
                            else:
                                max_per_group_pred.append(p_fine[:, idxs].max(dim=1).values)
                        max_per_group_pred = torch.stack(max_per_group_pred, dim=1)
                        cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group_pred.detach())
                        loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons
                        # Aux loss: alinear media de embeddings coarse con labels_coarse
                        if 'x_coarse_vec' in locals():
                            try:
                                coarse_pred_aux = x_coarse_vec.mean(dim=1)  # [B, D]
                                # Mapear D->num_coarse temporalmente con head_coarse weights si existen
                                if hasattr(model, 'head_coarse') and isinstance(model.head_coarse, nn.Sequential):
                                    head_last = model.head_coarse[-1]
                                    if isinstance(head_last, nn.Linear) and head_last.in_features == coarse_pred_aux.shape[-1]:
                                        coarse_logits_aux = head_last(coarse_pred_aux)
                                    else:
                                        coarse_logits_aux = nn.Linear(coarse_pred_aux.shape[-1], y_coarse.shape[-1]).to(device)(coarse_pred_aux)
                                else:
                                    coarse_logits_aux = nn.Linear(coarse_pred_aux.shape[-1], y_coarse.shape[-1]).to(device)(coarse_pred_aux)
                                aux_loss = nn.functional.binary_cross_entropy_with_logits(coarse_logits_aux, y_coarse)
                                loss = loss + 0.5 * aux_loss
                            except Exception:
                                pass

                    scaler.scale(loss).backward()
                    if step % args.accum_steps == 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        if epoch <= args.warmup_epochs:
                            for pg in opt.param_groups:
                                base_lr = args.lr
                                pg['lr'] = base_lr * float(epoch) / float(max(1, args.warmup_epochs))
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
                    pbar.set_postfix({'loss': f'{loss.item():.3f}'})
                    train_sum += loss.item() * x.size(0)
                    train_n += x.size(0)

                # validación
                attn_dir = None
                if getattr(args, 'save_attn_viz', False):
                    attn_dir = os.path.join(fold_dir, 'attn_maps')
                val_loss, val_metrics = evaluate(
                    model, va_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
                    use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5),
                    save_attn=getattr(args, 'save_attn_viz', False), attn_dir=attn_dir, attn_viz_max_batches=getattr(args, 'attn_viz_max_batches', 0)
                )
                train_mean = train_sum / max(1, train_n)
                current_lr = opt.param_groups[0]['lr']
                with open(fold_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch},{train_mean:.6f},{val_loss:.6f},{val_metrics.get('auroc_macro', float('nan')):.6f},{val_metrics.get('auprc_macro', float('nan')):.6f},{val_metrics.get('f1_macro', float('nan')):.6f},{current_lr:.8f}\n")
                if val_loss < best_val - args.early_stopping_min_delta:
                    best_val = val_loss
                    epochs_no_improve = 0
                    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, os.path.join(fold_dir, 'ckpt_best.pt'))
                else:
                    epochs_no_improve += 1
                print(f"Fold {fold_i+1} Epoch {epoch} done. Train: {train_mean:.4f} | Val: {val_loss:.4f} (best {best_val:.4f}) | AUROC_macro: {val_metrics.get('auroc_macro', float('nan')):.4f}")

                if epoch > args.warmup_epochs:
                    if args.scheduler_metric == 'val_loss':
                        scheduler.step(val_loss)
                    else:
                        scheduler.step(val_metrics.get('auroc_macro', float('nan')))
                    if cosine_sched is not None:
                        cosine_sched.step(epoch)

                if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                    print(f"[Fold {fold_i+1}] Early stopping tras {args.early_stopping_patience} épocas sin mejora")
                    break

            # Test del fold con mejor ckpt
            best_ckpt = os.path.join(fold_dir, 'ckpt_best.pt')
            if os.path.exists(best_ckpt):
                state = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(state['model'])
            test_loss, test_metrics = evaluate(
                model, te_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
                use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5)
            )
            print(f"Fold {fold_i+1} Test: AUROC_macro {test_metrics.get('auroc_macro', float('nan')):.4f} | AUPRC_macro {test_metrics.get('auprc_macro', float('nan')):.4f} | F1_macro {test_metrics.get('f1_macro', float('nan')):.4f}")
            all_test_metrics.append(test_metrics)

        # Promedio K-Fold
        keys = ['auroc_macro', 'auprc_macro', 'f1_macro', 'auroc_micro', 'auprc_micro']
        mean_metrics = {k: float(np.nanmean([m.get(k, np.nan) for m in all_test_metrics])) for k in keys}
        std_metrics = {k: float(np.nanstd([m.get(k, np.nan) for m in all_test_metrics])) for k in keys}
        print("\n=== K-Fold Average ===")
        for k in keys:
            print(f"{k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
        with open(os.path.join(exp_root, 'kfold_test_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({'mean': mean_metrics, 'std': std_metrics}, f, ensure_ascii=False, indent=2)
        print('Entrenamiento completo.')
        return

    # loaders (single split)
    tr_dl, va_dl, te_dl = build_loaders(
        args.dataset, args.sequence_len, hierarchy_path, args.batch_size, args.workers,
        cache_dir=args.cache_dir,
        target_fs=args.target_fs,
        bandpass_hz=(args.bandpass_low, args.bandpass_high),
        notch_hz=(args.notch_hz if args.notch_hz in (50,60) else None),
        seed=args.seed,
        use_sampler=(not args.no_sampler),
        sampler_power=args.sampler_power,
        sampler_power_rare=args.sampler_power_rare,
        rare_class_thresh=args.rare_class_thresh,
        aug_jitter_std=args.aug_jitter_std,
        aug_shift_max=args.aug_shift_max,
        aug_lead_drop_prob=args.aug_lead_drop_prob,
        aug_amp_scale_min=args.aug_amp_scale_min,
        aug_amp_scale_max=args.aug_amp_scale_max,
        aug_lead_noise_scale_max=args.aug_lead_noise_scale_max,
        aug_time_warp_max=args.aug_time_warp_max,
        aug_time_warp_p=args.aug_time_warp_p,
        smoke_test=args.smoke_test,
        smoke_n=args.smoke_n,
        mixup_alpha=args.mixup_alpha
    )

    # modelo
    configs = type('Cfg', (), dict(
        input_channels=12, sequence_len=args.sequence_len, kernel_size=8, stride=1, dropout=args.dropout,
        attn_dropout=(args.attn_dropout if args.attn_dropout is not None else args.dropout),
        trans_dropout=args.trans_dropout,
        mid_channels=64, final_out_channels=128, trans_dim=64, num_heads=4, num_leads=12,
        num_fine=len(fine_codes), num_coarse=len(coarse_groups),
        stockwell_freq_low=max(1, int(args.freq_bins_low)), stockwell_freq_high=max(2, int(args.freq_bins_high)),
        swin_freeze_stages=max(0, int(args.swin_freeze_stages))
    ))
    if _use_v2:
        model = ECGHybridVariableBeforeBiTransV2(configs, {"feature_dim": 128}).to(device)
    else:
        model = ECGHybridVariableBeforeBiTrans(configs, {"feature_dim": 128}).to(device)

    # optim / loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler según métrica
    sched_mode = 'min' if args.scheduler_metric == 'val_loss' else 'max'
    scheduler = ReduceLROnPlateau(opt, mode=sched_mode, factor=0.5, patience=5, threshold=1e-4, cooldown=0, min_lr=1e-6)
    cosine_sched = None
    if getattr(args, 'cosine_after_plateau', False):
        cosine_sched = CosineAnnealingWarmRestarts(
            opt,
            T_0=max(1, args.cosine_t0),
            T_mult=max(1, args.cosine_tmult),
            eta_min=args.lr * float(max(0.0, args.cosine_eta_min_scale))
        )
    # Criterio
    if args.loss_type == 'asl':
        base_loss_fn = AsymmetricLossMultiLabel(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, clip=args.asl_clip)
    else:
        def base_loss_fn(logits, targets):
            return focal_loss_multi_label(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma)
    scaler = GradScaler(enabled=args.mixed_precision and device.type=='cuda')

    # carpeta de experimento por dataset
    exp_root = os.path.join(args.exp_dir, args.dataset)
    os.makedirs(exp_root, exist_ok=True)
    log_path = os.path.join(exp_root, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,val_loss,val_auroc_macro,val_auprc_macro,val_f1_macro,lr\n')

    best_val = math.inf
    epochs_no_improve = 0
    max_epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
    for epoch in range(1, max_epochs+1):
        model.train()
        pbar = tqdm(tr_dl, desc=f'Epoch {epoch}/{args.epochs}')
        opt.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_n = 0
        for step, batch in enumerate(pbar, start=1):
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            # Mixup en loop sólo si NO se usa collate mixup
            use_collate_mixup = getattr(args, 'mixup_alpha', 0.0) > 0.0
            if use_collate_mixup:
                y_coarse_mix = y_coarse
                y_fine_mix = y_fine
            else:
                use_mixup = (getattr(args, 'mixup_alpha', 0.0) > 0.0 and np.random.rand() < getattr(args, 'mixup_p', 0.0))
                if use_mixup:
                    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                    perm = torch.randperm(x.size(0), device=x.device)
                    x = lam * x + (1.0 - lam) * x[perm]
                    y_coarse_mix = lam * y_coarse + (1.0 - lam) * y_coarse[perm]
                    y_fine_mix = lam * y_fine + (1.0 - lam) * y_fine[perm]
                else:
                    y_coarse_mix = y_coarse
                    y_fine_mix = y_fine
            # Label smoothing opcional
            if float(getattr(args, 'label_smoothing', 0.0)) > 0.0:
                eps = float(args.label_smoothing)
                y_coarse_mix = (1.0 - eps) * y_coarse_mix + 0.5 * eps
                y_fine_mix = (1.0 - eps) * y_fine_mix + 0.5 * eps
            with autocast_cuda(args.mixed_precision and device.type=='cuda'):
                if hasattr(model, 'forward_with_aux'):
                    logits_coarse, logits_fine, x_coarse_vec, x_fine_vec = model.forward_with_aux(x)
                else:
                    logits_coarse, logits_fine = model(x)
                loss_coarse = base_loss_fn(logits_coarse, y_coarse_mix)
                if getattr(args, 'use_dice_on_fine', False):
                    base_l = base_loss_fn(logits_fine, y_fine_mix)
                    dice_l = dice_loss_multilabel(logits_fine, y_fine_mix)
                    loss_fine = (1.0 - float(getattr(args, 'dice_weight', 0.5))) * base_l + float(getattr(args, 'dice_weight', 0.5)) * dice_l
                else:
                    loss_fine = base_loss_fn(logits_fine, y_fine_mix)
                # consistencia basada en predicciones
                p_fine = torch.sigmoid(logits_fine)
                max_per_group_pred = []
                for g, codes in coarse_groups.items():
                    idxs = [fine_code_to_idx.get(c, None) for c in codes]
                    idxs = [i for i in idxs if i is not None]
                    if len(idxs) == 0:
                        max_per_group_pred.append(torch.zeros(y_fine.shape[0], device=device))
                    else:
                        max_per_group_pred.append(p_fine[:, idxs].max(dim=1).values)
                max_per_group_pred = torch.stack(max_per_group_pred, dim=1)
                cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group_pred.detach())
                loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons
                if 'x_coarse_vec' in locals():
                    try:
                        coarse_pred_aux = x_coarse_vec.mean(dim=1)
                        if hasattr(model, 'head_coarse') and isinstance(model.head_coarse, nn.Sequential):
                            head_last = model.head_coarse[-1]
                            if isinstance(head_last, nn.Linear) and head_last.in_features == coarse_pred_aux.shape[-1]:
                                coarse_logits_aux = head_last(coarse_pred_aux)
                            else:
                                coarse_logits_aux = nn.Linear(coarse_pred_aux.shape[-1], y_coarse.shape[-1]).to(device)(coarse_pred_aux)
                        else:
                            coarse_logits_aux = nn.Linear(coarse_pred_aux.shape[-1], y_coarse.shape[-1]).to(device)(coarse_pred_aux)
                        aux_loss = nn.functional.binary_cross_entropy_with_logits(coarse_logits_aux, y_coarse)
                        loss = loss + 0.5 * aux_loss
                    except Exception:
                        pass

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5
                
                
                
                
                
                
                
                )
                # Warmup: escalar el LR durante las primeras warmup_epochs
                if epoch <= args.warmup_epochs:
                    for pg in opt.param_groups:
                        base_lr = args.lr
                        pg['lr'] = base_lr * float(epoch) / float(max(1, args.warmup_epochs))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            train_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        # validación
        # Visualización de atención opcional durante validación
        attn_dir = None
        if getattr(args, 'save_attn_viz', False):
            attn_dir = os.path.join(exp_root, 'attn_maps')
        val_loss, val_metrics = evaluate(
            model, va_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
            use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5),
            save_attn=getattr(args, 'save_attn_viz', False), attn_dir=attn_dir, attn_viz_max_batches=getattr(args, 'attn_viz_max_batches', 0)
        )
        train_mean = train_sum / max(1, train_n)
        current_lr = opt.param_groups[0]['lr']
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_mean:.6f},{val_loss:.6f},{val_metrics.get('auroc_macro', float('nan')):.6f},{val_metrics.get('auprc_macro', float('nan')):.6f},{val_metrics.get('f1_macro', float('nan')):.6f},{current_lr:.8f}\n")
        # checkpoint
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                   os.path.join(exp_root, f'ckpt_epoch_{epoch}.pt'))
        if val_loss < best_val - args.early_stopping_min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                       os.path.join(exp_root, 'ckpt_best.pt'))
        else:
            epochs_no_improve += 1
        
        print(f"Epoch {epoch} done. Train: {train_mean:.4f} | Val: {val_loss:.4f} (best {best_val:.4f}) | AUROC_macro: {val_metrics.get('auroc_macro', float('nan')):.4f}")

        # Actualizar el learning rate con ReduceLROnPlateau según val_loss (después del warmup)
        if epoch > args.warmup_epochs:
            if args.scheduler_metric == 'val_loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_metrics.get('auroc_macro', float('nan')))
            if cosine_sched is not None:
                cosine_sched.step(epoch)

        # Comprobar Early Stopping
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping activado tras {args.early_stopping_patience} épocas sin mejora ≥ {args.early_stopping_min_delta} en val_loss.")
            break

    # Evaluación en test con mejor checkpoint
    best_ckpt = os.path.join(exp_root, 'ckpt_best.pt')
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state['model'])
        print('Evaluando en test con mejor checkpoint...')
        # Optimización de umbrales por clase usando validación si está disponible
        val_loss_best, val_metrics_best = evaluate(
            model, va_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
            use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5)
        )
        y_true_val = val_metrics_best.get('y_true', None)
        y_prob_val = val_metrics_best.get('y_prob', None)
        class_thresholds = None
        if y_true_val is not None and y_prob_val is not None:
            try:
                from sklearn.metrics import f1_score
                c = y_true_val.shape[1]
                class_thresholds = np.zeros(c, dtype=np.float32)
                for i in range(c):
                    yi = y_true_val[:, i]
                    pi = y_prob_val[:, i]
                    if yi.sum() == 0:
                        class_thresholds[i] = 0.5
                        continue
                    # búsqueda de umbral óptimo por F1
                    best_f1 = -1.0
                    best_t = 0.5
                    for t in np.linspace(0.05, 0.95, 19):
                        f1i = f1_score(yi, (pi >= t).astype(np.float32), zero_division=0)
                        if f1i > best_f1:
                            best_f1 = f1i
                            best_t = t
                    class_thresholds[i] = best_t
            except Exception:
                class_thresholds = None

        # TTA/Multi-crop en test: promediar probabilidades de k ventanas si la señal es > sequence_len
        test_loss, test_metrics = evaluate(
            model, te_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
            use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5)
        )
        print(f"Test loss: {test_loss:.4f} | AUROC_macro: {test_metrics.get('auroc_macro', float('nan')):.4f} | AUPRC_macro: {test_metrics.get('auprc_macro', float('nan')):.4f} | F1_macro: {test_metrics.get('f1_macro', float('nan')):.4f}")
        # Recalcular F1 usando umbrales por-clase si los tenemos
        if class_thresholds is not None:
            try:
                from sklearn.metrics import f1_score
                y_true_test = test_metrics.get('y_true')
                y_prob_test = test_metrics.get('y_prob')
                if y_true_test is not None and y_prob_test is not None:
                    y_pred_thr = (y_prob_test >= class_thresholds[None, :]).astype(np.float32)
                    f1_macro_thr = float(np.nanmean([
                        f1_score(y_true_test[:, i], y_pred_thr[:, i], zero_division=0) if y_true_test[:, i].sum() > 0 else np.nan
                        for i in range(y_true_test.shape[1])
                    ]))
                    print(f"F1_macro (umbrales por clase): {f1_macro_thr:.4f}")
            except Exception:
                pass
        # Guardar métricas serializables a JSON y arrays por separado en .npz
        tm = dict(test_metrics)
        y_true_test = tm.pop('y_true', None)
        y_prob_test = tm.pop('y_prob', None)
        with open(os.path.join(exp_root, 'test_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({'loss': test_loss, **tm}, f, ensure_ascii=False, indent=2)
        if y_true_test is not None and y_prob_test is not None:
            np.savez(os.path.join(exp_root, 'test_predictions.npz'), y_true=y_true_test, y_prob=y_prob_test, thresholds=class_thresholds)
    print('Entrenamiento completo.')


if __name__ == '__main__':
    main()


