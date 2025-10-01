import os
import json
import math
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.amp import autocast as _autocast_new, GradScaler  # PyTorch >=2.0
    def autocast_cuda(enabled):
        return _autocast_new('cuda', enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler  # Fallback
    def autocast_cuda(enabled):
        return _autocast_old(enabled=enabled)
from tqdm import tqdm

from losses.asymmetric_loss import AsymmetricLossMultiLabel
from datasets.ecg12large import ECG12Large
from scripts.generate_labels_hierarchy import generate_labels_hierarchy as _gen_wfdb_hierarchy

from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST


def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return None, None, None, None
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank, torch.device(f'cuda:{local_rank}')


def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def build_loaders_12large(sequence_len, hierarchy_path, batch_size, workers, cache_dir=None,
                         target_fs=500.0, bandpass_hz=(0.5, 45.0), notch_hz=None, seed=42,
                         smoke_test=False, smoke_n=256, is_distributed=False,
                         aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                         aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
                         aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0,
                         sampler_weighted=False, max_weight=10.0):
    from glob import glob
    from datasets.ecg12large import extract_patient_id

    root = os.path.join('datos', '12Large', 'WFDBRecords')
    all_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    
    if smoke_test:
        all_files = all_files[:smoke_n]
    
    # Usar HMSTPreprocessor que maneja 7500 puntos
    from datasets.HMSTPreprocessor import HMSTPreprocessor
    full_ds = HMSTPreprocessor(
        root, sequence_len=sequence_len, files=all_files,
        multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
        random_crop=not smoke_test, target_fs=target_fs, 
        bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=False,
        aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max, 
        aug_lead_drop_prob=aug_lead_drop_prob, aug_amp_scale_min=aug_amp_scale_min,
        aug_amp_scale_max=aug_amp_scale_max, aug_lead_noise_scale_max=aug_lead_noise_scale_max,
        aug_time_warp_max=aug_time_warp_max, aug_time_warp_p=aug_time_warp_p
    )
    
    # Split train/val/test (80/10/10)
    n = len(full_ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_ds = torch.utils.data.Subset(full_ds, range(n_train))
    val_ds = torch.utils.data.Subset(full_ds, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(full_ds, range(n_train + n_val, n))
    
    # Distributed samplers
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # Sampler ponderado (solo si no hay sampler distribuido)
    if sampler_weighted and (train_sampler is None) and not smoke_test:
        try:
            import numpy as _np
            from torch.utils.data import WeightedRandomSampler as _WRS
            # train_ds es Subset de HMSTPreprocessor → acceder a .dataset.samples y .indices
            base_ds = train_ds.dataset
            indices = train_ds.indices if hasattr(train_ds, 'indices') else list(range(len(train_ds)))
            y_fines = [_np.asarray(base_ds.samples[i][2], dtype=_np.float32) for i in indices]
            Y = _np.stack(y_fines, axis=0)
            prev = Y.mean(axis=0)
            inv = 1.0 / _np.clip(prev, 1e-6, None)
            inv = _np.minimum(inv, float(max_weight))
            weights = []
            for yi in Y:
                if yi.sum() > 0:
                    w = float(inv[yi > 0].mean())
                else:
                    w = 1.0
                weights.append(w)
            wrs = _WRS(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                 sampler=wrs, num_workers=workers, pin_memory=True)
        except Exception:
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                                 sampler=train_sampler, num_workers=workers, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       sampler=val_sampler, num_workers=workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_dl, val_dl, test_dl


def dice_loss_multilabel(logits, targets, smooth=1e-6):
    """Dice loss for multi-label classification"""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def focal_loss_multi_label(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for multi-label classification"""
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    return focal_loss.mean()


def evaluate_v3(model, dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
                use_dice_on_fine=False, dice_weight=0.5, device=None, is_distributed=False):
    """Evaluate model with distributed support"""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    all_coarse_preds = []
    all_fine_preds = []
    all_coarse_targets = []
    all_fine_targets = []
    
    with torch.no_grad():
        for batch in dl:
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            
            b = x.size(0)
            wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
            snomed_embed = y_coarse
            
            with autocast_cuda(device.type=='cuda'):
                model_output = model(x, wide_feats, snomed_embed)
                if len(model_output) == 3:
                    logits_coarse, logits_fine, _ = model_output
                else:
                    logits_coarse, logits_fine = model_output
                
                loss_coarse = base_loss_fn(logits_coarse, y_coarse)
                if use_dice_on_fine:
                    base_l = base_loss_fn(logits_fine, y_fine)
                    dice_l = dice_loss_multilabel(logits_fine, y_fine)
                    loss_fine = (1.0 - dice_weight) * base_l + dice_weight * dice_l
                else:
                    loss_fine = base_loss_fn(logits_fine, y_fine)
                
                # Consistency loss
                p_fine = torch.sigmoid(logits_fine)
                max_per_group_pred = []
                for g, codes in coarse_groups.items():
                    idxs = [fine_code_to_idx.get(c, None) for c in codes]
                    idxs = [i for i in idxs if i is not None]
                    if len(idxs) == 0:
                        max_per_group_pred.append(torch.zeros(b, device=device))
                    else:
                        max_per_group_pred.append(p_fine[:, idxs].max(dim=1).values)
                max_per_group_pred = torch.stack(max_per_group_pred, dim=1)
                cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group_pred.detach())
                loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons
            
            total_loss += loss.item() * b
            n_samples += b
            
            if compute_stats:
                # Mantener en CUDA para NCCL; no pasar a CPU antes del all_gather
                all_coarse_preds.append(torch.sigmoid(logits_coarse))
                all_fine_preds.append(torch.sigmoid(logits_fine))
                all_coarse_targets.append(y_coarse)
                all_fine_targets.append(y_fine)
    
    mean_loss = total_loss / n_samples
    
    if compute_stats and len(all_coarse_preds) > 0:
        # Gather predictions from all processes
        if is_distributed:
            # Concatenar locales (CUDA)
            all_coarse_preds = torch.cat(all_coarse_preds, dim=0)
            all_fine_preds = torch.cat(all_fine_preds, dim=0)
            all_coarse_targets = torch.cat(all_coarse_targets, dim=0)
            all_fine_targets = torch.cat(all_fine_targets, dim=0)

            ws = dist.get_world_size()
            # Recolectar longitudes (CUDA int64)
            n_cp = torch.tensor([all_coarse_preds.size(0)], device=device, dtype=torch.int64)
            n_fp = torch.tensor([all_fine_preds.size(0)], device=device, dtype=torch.int64)
            n_ct = torch.tensor([all_coarse_targets.size(0)], device=device, dtype=torch.int64)
            n_ft = torch.tensor([all_fine_targets.size(0)], device=device, dtype=torch.int64)
            g_n_cp = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(ws)]
            g_n_fp = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(ws)]
            g_n_ct = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(ws)]
            g_n_ft = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(ws)]
            dist.all_gather(g_n_cp, n_cp)
            dist.all_gather(g_n_fp, n_fp)
            dist.all_gather(g_n_ct, n_ct)
            dist.all_gather(g_n_ft, n_ft)
            max_cp = int(torch.stack([t[0] for t in g_n_cp]).max().item())
            max_fp = int(torch.stack([t[0] for t in g_n_fp]).max().item())
            max_ct = int(torch.stack([t[0] for t in g_n_ct]).max().item())
            max_ft = int(torch.stack([t[0] for t in g_n_ft]).max().item())

            # Acolchado a máximos
            if all_coarse_preds.size(0) < max_cp:
                pad = max_cp - all_coarse_preds.size(0)
                all_coarse_preds = F.pad(all_coarse_preds, (0, 0, 0, pad))
            if all_fine_preds.size(0) < max_fp:
                pad = max_fp - all_fine_preds.size(0)
                all_fine_preds = F.pad(all_fine_preds, (0, 0, 0, pad))
            if all_coarse_targets.size(0) < max_ct:
                pad = max_ct - all_coarse_targets.size(0)
                all_coarse_targets = F.pad(all_coarse_targets, (0, 0, 0, pad))
            if all_fine_targets.size(0) < max_ft:
                pad = max_ft - all_fine_targets.size(0)
                all_fine_targets = F.pad(all_fine_targets, (0, 0, 0, pad))

            # Gather en CUDA
            g_cp = [torch.empty_like(all_coarse_preds, device=device) for _ in range(ws)]
            g_fp = [torch.empty_like(all_fine_preds, device=device) for _ in range(ws)]
            g_ct = [torch.empty_like(all_coarse_targets, device=device) for _ in range(ws)]
            g_ft = [torch.empty_like(all_fine_targets, device=device) for _ in range(ws)]
            dist.all_gather(g_cp, all_coarse_preds)
            dist.all_gather(g_fp, all_fine_preds)
            dist.all_gather(g_ct, all_coarse_targets)
            dist.all_gather(g_ft, all_fine_targets)

            parts_cp, parts_fp, parts_ct, parts_ft = [], [], [], []
            for r in range(ws):
                n_r_cp = int(g_n_cp[r][0].item())
                n_r_fp = int(g_n_fp[r][0].item())
                n_r_ct = int(g_n_ct[r][0].item())
                n_r_ft = int(g_n_ft[r][0].item())
                if n_r_cp > 0:
                    parts_cp.append(g_cp[r][:n_r_cp])
                if n_r_fp > 0:
                    parts_fp.append(g_fp[r][:n_r_fp])
                if n_r_ct > 0:
                    parts_ct.append(g_ct[r][:n_r_ct])
                if n_r_ft > 0:
                    parts_ft.append(g_ft[r][:n_r_ft])
            all_coarse_preds = torch.cat(parts_cp, dim=0) if parts_cp else all_coarse_preds[:0]
            all_fine_preds = torch.cat(parts_fp, dim=0) if parts_fp else all_fine_preds[:0]
            all_coarse_targets = torch.cat(parts_ct, dim=0) if parts_ct else all_coarse_targets[:0]
            all_fine_targets = torch.cat(parts_ft, dim=0) if parts_ft else all_fine_targets[:0]
        else:
            all_coarse_preds = torch.cat(all_coarse_preds, dim=0)
            all_fine_preds = torch.cat(all_fine_preds, dim=0)
            all_coarse_targets = torch.cat(all_coarse_targets, dim=0)
            all_fine_targets = torch.cat(all_fine_targets, dim=0)
        
        # Compute metrics (robust to classes with only one label value)
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

        # Mover a CPU para sklearn
        cp = all_coarse_preds.detach().cpu().numpy()
        fp = all_fine_preds.detach().cpu().numpy()
        ct = all_coarse_targets.detach().cpu().numpy()
        ft = all_fine_targets.detach().cpu().numpy()

        def _per_class_metrics(y_true_np, y_prob_np):
            c = y_true_np.shape[1]
            aurocs, auprcs, f1s = [], [], []
            pos_counts = y_true_np.sum(axis=0)
            neg_counts = (1.0 - y_true_np).sum(axis=0)
            for i in range(c):
                yi = y_true_np[:, i]
                pi = y_prob_np[:, i]
                has_pos = pos_counts[i] > 0
                has_neg = neg_counts[i] > 0
                if has_pos and has_neg:
                    try:
                        aurocs.append(roc_auc_score(yi, pi))
                    except Exception:
                        aurocs.append(np.nan)
                else:
                    aurocs.append(np.nan)
                if has_pos:
                    try:
                        auprcs.append(average_precision_score(yi, pi))
                    except Exception:
                        auprcs.append(np.nan)
                    try:
                        f1s.append(f1_score(yi, (pi >= 0.5).astype(np.int32), zero_division=0))
                    except Exception:
                        f1s.append(np.nan)
                else:
                    auprcs.append(np.nan)
                    f1s.append(np.nan)
            def _nanmean_safe(arr):
                return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float('nan')
            metrics_local = {
                'auroc_macro': _nanmean_safe(np.array(aurocs, dtype=float)),
                'auprc_macro': _nanmean_safe(np.array(auprcs, dtype=float)),
                'f1_macro': _nanmean_safe(np.array(f1s, dtype=float)),
                'num_valid_auroc': int(np.sum((pos_counts > 0) & (neg_counts > 0))),
                'num_valid_auprc': int(np.sum(pos_counts > 0)),
                'num_valid_f1': int(np.sum(pos_counts > 0)),
                'num_allzero': int(np.sum(pos_counts == 0)),
                'num_allone': int(np.sum(neg_counts == 0)),
                'prevalence_mean': float(np.mean(y_true_np)),
            }
            # Micro (probar, si falla devolver NaN)
            try:
                metrics_local['auroc_micro'] = float(roc_auc_score(y_true_np, y_prob_np, average='micro'))
            except Exception:
                metrics_local['auroc_micro'] = float('nan')
            try:
                metrics_local['auprc_micro'] = float(average_precision_score(y_true_np, y_prob_np, average='micro'))
            except Exception:
                metrics_local['auprc_micro'] = float('nan')
            return metrics_local

        coarse_m = _per_class_metrics(ct, cp)
        fine_m = _per_class_metrics(ft, fp)

        def _nm(x):
            return x if (isinstance(x, float) and np.isfinite(x)) else np.nan

        metrics = {
            'coarse_auroc': coarse_m['auroc_macro'],
            'coarse_auprc': coarse_m['auprc_macro'],
            'coarse_f1': coarse_m['f1_macro'],
            'coarse_auroc_micro': coarse_m['auroc_micro'],
            'coarse_auprc_micro': coarse_m['auprc_micro'],
            'coarse_num_valid_auroc': coarse_m['num_valid_auroc'],
            'coarse_num_valid_auprc': coarse_m['num_valid_auprc'],
            'coarse_num_valid_f1': coarse_m['num_valid_f1'],
            'coarse_num_allzero': coarse_m['num_allzero'],
            'coarse_num_allone': coarse_m['num_allone'],
            'coarse_prevalence_mean': coarse_m['prevalence_mean'],

            'fine_auroc': fine_m['auroc_macro'],
            'fine_auprc': fine_m['auprc_macro'],
            'fine_f1': fine_m['f1_macro'],
            'fine_auroc_micro': fine_m['auroc_micro'],
            'fine_auprc_micro': fine_m['auprc_micro'],
            'fine_num_valid_auroc': fine_m['num_valid_auroc'],
            'fine_num_valid_auprc': fine_m['num_valid_auprc'],
            'fine_num_valid_f1': fine_m['num_valid_f1'],
            'fine_num_allzero': fine_m['num_allzero'],
            'fine_num_allone': fine_m['num_allone'],
            'fine_prevalence_mean': fine_m['prevalence_mean'],

            # Combinar coarse y fine con promedio ignorando NaN
            'macro_auroc': float(np.nanmean([_nm(coarse_m['auroc_macro']), _nm(fine_m['auroc_macro'])])),
            'macro_auprc': float(np.nanmean([_nm(coarse_m['auprc_macro']), _nm(fine_m['auprc_macro'])])),
            'macro_f1': float(np.nanmean([_nm(coarse_m['f1_macro']), _nm(fine_m['f1_macro'])])),
        }
        # Alias compatibles con evaluate_results.py (usa sufijo _macro y micro global en fine)
        metrics['auroc_macro'] = metrics['macro_auroc']
        metrics['auprc_macro'] = metrics['macro_auprc']
        metrics['f1_macro'] = metrics['macro_f1']
        metrics['auroc_micro'] = fine_m['auroc_micro']
        metrics['auprc_micro'] = fine_m['auprc_micro']
        # Incluir arrays de test (fine) para graficado externo
        metrics['y_true'] = ft
        metrics['y_prob'] = fp
    else:
        metrics = {}
    
    return mean_loss, metrics


def main():
    # Setup distributed training
    rank, world_size, local_rank, device = setup_ddp()
    is_distributed = rank is not None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_len', type=int, default=7500)  # Usar 7500 por HMSTPreprocessor
    parser.add_argument('--batch_size', type=int, default=32)  # Reducido para memoria
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--accum_steps', type=int, default=8)  # Aumentado para mantener effective batch size
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=os.path.join('datos','pt_cache'))
    parser.add_argument('--exp_dir', type=str, default=os.path.join('experiments_logs','full_run_v3_ddp'))
    parser.add_argument('--target_fs', type=float, default=500.0)
    parser.add_argument('--bandpass_low', type=float, default=0.5)
    parser.add_argument('--bandpass_high', type=float, default=45.0)
    parser.add_argument('--notch_hz', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--loss_type', type=str, default='asl', choices=['asl','focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_dice_on_fine', action='store_true')
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='12large', choices=['12large'])
    parser.add_argument('--no_auto_hierarchy', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--smoke_n', type=int, default=256)
    # Augmentaciones y oversampling
    parser.add_argument('--aug_jitter_std', type=float, default=0.0)
    parser.add_argument('--aug_shift_max', type=int, default=0)
    parser.add_argument('--aug_lead_drop_prob', type=float, default=0.0)
    parser.add_argument('--aug_amp_scale_min', type=float, default=1.0)
    parser.add_argument('--aug_amp_scale_max', type=float, default=1.0)
    parser.add_argument('--aug_lead_noise_scale_max', type=float, default=1.0)
    parser.add_argument('--aug_time_warp_max', type=float, default=0.0)
    parser.add_argument('--aug_time_warp_p', type=float, default=0.0)
    parser.add_argument('--oversample_minority', action='store_true')
    parser.add_argument('--oversample_max_weight', type=float, default=10.0)
    # HMST params optimizados
    parser.add_argument('--hmst_d_model', type=int, default=128)  # Reducido
    parser.add_argument('--hmst_heads', type=int, default=4)  # Reducido
    parser.add_argument('--hmst_layers', type=int, default=4)  # Reducido
    parser.add_argument('--hmst_stages', type=int, default=2)  # Reducido
    # Eval-only
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='')
    args = parser.parse_args()

    # Solo el proceso 0 imprime logs
    if not is_distributed or rank == 0:
        print(f"=== CONFIGURACIÓN DDP ===")
        print(f"Distributed: {is_distributed}")
        if is_distributed:
            print(f"World size: {world_size}")
            print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Batch size: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * args.accum_steps * (world_size if is_distributed else 1)}")
        print(f"d_model: {args.hmst_d_model}")
        print(f"Sequence len: {args.sequence_len}")
        print("========================")

    set_seed(args.seed, deterministic=args.deterministic)
    
    try:
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # Rutas de datos / jerarquía
    data_root = os.path.join('datos', '12Large', 'WFDBRecords')
    hierarchy_path = os.path.join('datos', '12Large', 'labels_hierarchy.json')
    if not os.path.exists(hierarchy_path) and not args.no_auto_hierarchy:
        try:
            from scripts.generate_labels_hierarchy import generate_labels_hierarchy as _gen
            _gen(os.path.join('datos','12Large'), hierarchy_path, top_k=30)
        except Exception as e:
            if not is_distributed or rank == 0:
                print(f'Aviso: no se pudo generar jerarquía automáticamente: {e}')
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f'No se encontró jerarquía en {hierarchy_path}')

    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    coarse_groups = hier['coarse_groups']
    fine_code_to_idx = {c:i for i,c in enumerate(fine_codes)}
    num_fine = len(fine_codes)
    num_coarse = len(coarse_groups)

    # DataLoaders
    tr_dl, va_dl, te_dl = build_loaders_12large(
        args.sequence_len, hierarchy_path, args.batch_size, args.workers,
        cache_dir=args.cache_dir, target_fs=args.target_fs,
        bandpass_hz=(args.bandpass_low, args.bandpass_high),
        notch_hz=(args.notch_hz if args.notch_hz in (50,60) else None),
        seed=args.seed, smoke_test=args.smoke_test, smoke_n=args.smoke_n,
        is_distributed=is_distributed,
        aug_jitter_std=float(args.aug_jitter_std), aug_shift_max=int(args.aug_shift_max),
        aug_lead_drop_prob=float(args.aug_lead_drop_prob),
        aug_amp_scale_min=float(args.aug_amp_scale_min), aug_amp_scale_max=float(args.aug_amp_scale_max),
        aug_lead_noise_scale_max=float(args.aug_lead_noise_scale_max),
        aug_time_warp_max=float(args.aug_time_warp_max), aug_time_warp_p=float(args.aug_time_warp_p),
        sampler_weighted=bool(args.oversample_minority), max_weight=float(args.oversample_max_weight)
    )

    # Modelo HMST
    model = HMST(
        input_channels=12+3,
        d_model=int(args.hmst_d_model),
        nhead=int(args.hmst_heads),
        num_layers=int(args.hmst_layers),
        num_stages=int(args.hmst_stages),
        num_coarse=num_coarse,
        num_fine=num_fine,
        dropout=float(args.dropout),
        snomed_dim=num_coarse,
    ).to(device)
    
    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizador y pérdidas
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, threshold=1e-4, cooldown=0, min_lr=1e-6)
    if args.loss_type == 'asl':
        base_loss_fn = AsymmetricLossMultiLabel(gamma_pos=3.0, gamma_neg=4.0, clip=0.05)
    else:
        def base_loss_fn(logits, targets):
            return focal_loss_multi_label(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma)
    scaler = GradScaler(enabled=args.mixed_precision and device.type=='cuda')

    # Logs (solo proceso 0)
    if not is_distributed or rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        log_path = os.path.join(args.exp_dir, 'train_log.csv')
        if not os.path.exists(log_path):
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('epoch,train_loss,val_loss,val_auroc_macro,val_auprc_macro,val_f1_macro,lr\n')

    best_val = math.inf
    epochs_no_improve = 0

    # Ruta por defecto de checkpoint
    default_ckpt_path = os.path.join(args.exp_dir, 'ckpt_best.pt')

    # Modo evaluación únicamente (no entrena)
    if args.eval_only:
        # Cargar checkpoint
        ckpt_to_load = args.ckpt_path if (args.ckpt_path and os.path.exists(args.ckpt_path)) else default_ckpt_path
        if not os.path.exists(ckpt_to_load):
            raise FileNotFoundError(f'No se encontró checkpoint para evaluar: {ckpt_to_load}')
        state = torch.load(ckpt_to_load, map_location=device)
        if is_distributed:
            model.module.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state['model'])
        else:
            model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state['model'])
        if is_distributed:
            dist.barrier()
        # Evaluación en test
        if not is_distributed or rank == 0:
            print('Evaluando en test set (eval_only)...')
        test_loss, test_metrics = evaluate_v3(model, te_dl, fine_code_to_idx, coarse_groups,
                                              base_loss_fn, compute_stats=True,
                                              use_dice_on_fine=getattr(args, 'use_dice_on_fine', False),
                                              dice_weight=getattr(args, 'dice_weight', 0.5),
                                              device=device, is_distributed=is_distributed)
        if is_distributed:
            dist.barrier()
        if not is_distributed or rank == 0:
            print(f'Test results: {test_metrics}')
            os.makedirs(args.exp_dir, exist_ok=True)
            # Guardar JSON sin arrays y NPZ con arrays
            tm = dict(test_metrics)
            y_true_np = tm.pop('y_true', None)
            y_prob_np = tm.pop('y_prob', None)
            with open(os.path.join(args.exp_dir, 'test_metrics.json'), 'w') as f:
                json.dump(tm, f, indent=2)
            if y_true_np is not None and y_prob_np is not None:
                np.savez(os.path.join(args.exp_dir, 'test_predictions.npz'), y_true=y_true_np, y_prob=y_prob_np)
        cleanup_ddp()
        return
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        if is_distributed:
            tr_dl.sampler.set_epoch(epoch)
        
        model.train()
        pbar = tqdm(tr_dl, desc=f'Epoch {epoch}/{args.epochs}', disable=(is_distributed and rank != 0))
        opt.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_n = 0
        
        for step, batch in enumerate(pbar, start=1):
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)

            b = x.size(0)
            wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
            snomed_embed = y_coarse

            with autocast_cuda(args.mixed_precision and device.type=='cuda'):
                model_output = model(x, wide_feats, snomed_embed)
                if len(model_output) == 3:
                    logits_coarse, logits_fine, _ = model_output
                else:
                    logits_coarse, logits_fine = model_output
                loss_coarse = base_loss_fn(logits_coarse, y_coarse)
                if getattr(args, 'use_dice_on_fine', False):
                    base_l = base_loss_fn(logits_fine, y_fine)
                    dice_l = dice_loss_multilabel(logits_fine, y_fine)
                    loss_fine = (1.0 - float(args.dice_weight)) * base_l + float(args.dice_weight) * dice_l
                else:
                    loss_fine = base_loss_fn(logits_fine, y_fine)
                
                # Consistency loss
                p_fine = torch.sigmoid(logits_fine)
                max_per_group_pred = []
                for g, codes in coarse_groups.items():
                    idxs = [fine_code_to_idx.get(c, None) for c in codes]
                    idxs = [i for i in idxs if i is not None]
                    if len(idxs) == 0:
                        max_per_group_pred.append(torch.zeros(b, device=device))
                    else:
                        max_per_group_pred.append(p_fine[:, idxs].max(dim=1).values)
                max_per_group_pred = torch.stack(max_per_group_pred, dim=1)
                cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group_pred.detach())
                loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            if not is_distributed or rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            train_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        # Validación
        val_loss, val_metrics = evaluate_v3(model, va_dl, fine_code_to_idx, coarse_groups, 
                                          base_loss_fn, compute_stats=True,
                                          use_dice_on_fine=getattr(args, 'use_dice_on_fine', False),
                                          dice_weight=getattr(args, 'dice_weight', 0.5),
                                          device=device, is_distributed=is_distributed)
        
        # Scheduler step
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]['lr']

        # Early stopping sincronizado entre ranks
        stop_now = torch.tensor([0], device=device)
        if not is_distributed or rank == 0:
            val_auroc = val_metrics.get('macro_auroc', 0.0)
            val_auprc = val_metrics.get('macro_auprc', 0.0)
            val_f1 = val_metrics.get('macro_f1', 0.0)

            print(f'Epoch {epoch}: train_loss={train_sum/train_n:.4f}, val_loss={val_loss:.4f}, '
                  f'val_auroc={val_auroc:.4f}, val_auprc={val_auprc:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.2e}')

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f'{epoch},{train_sum/train_n:.6f},{val_loss:.6f},{val_auroc:.6f},{val_auprc:.6f},{val_f1:.6f},{current_lr:.6e}\n')

            # Early stopping: decidir en rank 0
            if val_loss < best_val - args.early_stopping_min_delta:
                best_val = val_loss
                epochs_no_improve = 0
                # Guardar checkpoint (solo rank 0)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, os.path.join(args.exp_dir, 'ckpt_best.pt'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    stop_now = torch.tensor([1], device=device)

        # Broadcast de la decisión de parada a todos los ranks
        if is_distributed:
            dist.broadcast(stop_now, src=0)
        if int(stop_now.item()) == 1:
            break

    # Cargar mejor checkpoint antes de test (en todos los ranks para sincronía)
    if os.path.exists(default_ckpt_path):
        state = torch.load(default_ckpt_path, map_location=device)
        if is_distributed:
            model.module.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state['model'])
        else:
            model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state['model'])
    if is_distributed:
        dist.barrier()

    # Evaluación final (todos los ranks ejecutan para evitar deadlocks; guarda/imprime solo rank 0)
    if is_distributed:
        dist.barrier()
    print_prefix = (not is_distributed) or (rank == 0)
    if print_prefix:
        print('Evaluando en test set...')
    test_loss, test_metrics = evaluate_v3(model, te_dl, fine_code_to_idx, coarse_groups, 
                                        base_loss_fn, compute_stats=True,
                                        use_dice_on_fine=getattr(args, 'use_dice_on_fine', False),
                                        dice_weight=getattr(args, 'dice_weight', 0.5),
                                        device=device, is_distributed=is_distributed)
    if is_distributed:
        dist.barrier()
    if print_prefix:
        print(f'Test results: {test_metrics}')
        # Guardar JSON sin arrays y NPZ con arrays
        tm = dict(test_metrics)
        y_true_np = tm.pop('y_true', None)
        y_prob_np = tm.pop('y_prob', None)
        with open(os.path.join(args.exp_dir, 'test_metrics.json'), 'w') as f:
            json.dump(tm, f, indent=2)
        if y_true_np is not None and y_prob_np is not None:
            np.savez(os.path.join(args.exp_dir, 'test_predictions.npz'), y_true=y_true_np, y_prob=y_prob_np)

    cleanup_ddp()


if __name__ == '__main__':
    main()