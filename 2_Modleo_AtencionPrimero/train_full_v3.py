import os
import json
import math
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
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
                          smoke_test=False, smoke_n=256,
                          aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                          aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
                          aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0):
    from glob import glob
    from datasets.ecg12large import extract_patient_id

    root = os.path.join('datos', '12Large', 'WFDBRecords')
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
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = device.type == 'cuda'
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    return tr_dl, va_dl, te_dl


def focal_loss_multi_label(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
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


def evaluate_v3(model, dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
                use_dice_on_fine=False, dice_weight=0.5):
    device = next(model.parameters()).device
    model.eval()
    loss_sum = 0.0
    count = 0
    all_y_fine = []
    all_p_fine = []
    with torch.no_grad():
        for batch in dl:
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            b = x.size(0)
            wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
            snomed_embed = y_coarse  # asumir num_coarse == snomed_dim

            logits_coarse, logits_fine, _ = model(x, wide_feats, snomed_embed)
            loss_coarse = base_loss_fn(logits_coarse, y_coarse)
            if use_dice_on_fine:
                base_l = base_loss_fn(logits_fine, y_fine)
                dice_l = dice_loss_multilabel(logits_fine, y_fine)
                loss_fine = (1.0 - float(dice_weight)) * base_l + float(dice_weight) * dice_l
            else:
                loss_fine = base_loss_fn(logits_fine, y_fine)

            # consistencia basada en predicciones como en v2
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
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
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
        from sklearn.metrics import roc_auc_score as _roc
        auroc_micro = _roc(y_true, y_prob, average='micro')
    except Exception:
        auroc_micro = float('nan')
    try:
        from sklearn.metrics import average_precision_score as _apr
        auprc_micro = _apr(y_true, y_prob, average='micro')
    except Exception:
        auprc_micro = float('nan')
    metrics.update({
        'auroc_macro': nanmean_safe(np.array(aurocs, dtype=float)),
        'auprc_macro': nanmean_safe(np.array(auprcs, dtype=float)),
        'f1_macro': nanmean_safe(np.array(f1s, dtype=float)),
        'auroc_micro': float(auroc_micro),
        'auprc_micro': float(auprc_micro),
        'y_true': y_true,
        'y_prob': y_prob,
    })
    return mean_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=os.path.join('datos','pt_cache'))
    parser.add_argument('--exp_dir', type=str, default=os.path.join('experiments_logs','full_run_v3'))
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
    # HMST params
    parser.add_argument('--hmst_d_model', type=int, default=256)
    parser.add_argument('--hmst_heads', type=int, default=8)
    parser.add_argument('--hmst_layers', type=int, default=6)
    parser.add_argument('--hmst_stages', type=int, default=3)
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

    # Rutas de datos / jerarquía
    data_root = os.path.join('datos', '12Large', 'WFDBRecords')
    hierarchy_path = os.path.join('datos', '12Large', 'labels_hierarchy.json')
    if not os.path.exists(hierarchy_path) and not args.no_auto_hierarchy:
        try:
            from scripts.generate_labels_hierarchy import generate_labels_hierarchy as _gen
            _gen(os.path.join('datos','12Large'), hierarchy_path, top_k=30)
        except Exception as e:
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
        aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
        aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
        aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0,
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

    # Optimizador y pérdidas
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, threshold=1e-4, cooldown=0, min_lr=1e-6)
    if args.loss_type == 'asl':
        base_loss_fn = AsymmetricLossMultiLabel(gamma_pos=3.0, gamma_neg=4.0, clip=0.05)
    else:
        def base_loss_fn(logits, targets):
            return focal_loss_multi_label(logits, targets, alpha=args.focal_alpha, gamma=args.focal_gamma)
    scaler = GradScaler(enabled=args.mixed_precision and device.type=='cuda')

    # Logs
    os.makedirs(args.exp_dir, exist_ok=True)
    log_path = os.path.join(args.exp_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,val_loss,val_auroc_macro,val_auprc_macro,val_f1_macro,lr\n')

    best_val = math.inf
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr_dl, desc=f'Epoch {epoch}/{args.epochs}')
        opt.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_n = 0
        for step, batch in enumerate(pbar, start=1):
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)

            b = x.size(0)
            wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
            snomed_embed = y_coarse  # usar coarse como embedding SNOMED simple

            with autocast_cuda(args.mixed_precision and device.type=='cuda'):
                logits_coarse, logits_fine, _ = model(x, wide_feats, snomed_embed)
                loss_coarse = base_loss_fn(logits_coarse, y_coarse)
                if getattr(args, 'use_dice_on_fine', False):
                    base_l = base_loss_fn(logits_fine, y_fine)
                    dice_l = dice_loss_multilabel(logits_fine, y_fine)
                    loss_fine = (1.0 - float(args.dice_weight)) * base_l + float(args.dice_weight) * dice_l
                else:
                    loss_fine = base_loss_fn(logits_fine, y_fine)
                # consistencia como en v2
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

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            train_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        # validación
        val_loss, val_metrics = evaluate_v3(
            model, va_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
            use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5)
        )
        train_mean = train_sum / max(1, train_n)
        current_lr = opt.param_groups[0]['lr']
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_mean:.6f},{val_loss:.6f},{val_metrics.get('auroc_macro', float('nan')):.6f},{val_metrics.get('auprc_macro', float('nan')):.6f},{val_metrics.get('f1_macro', float('nan')):.6f},{current_lr:.8f}\n")
        if val_loss < best_val - args.early_stopping_min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, os.path.join(args.exp_dir, 'ckpt_best.pt'))
        else:
            epochs_no_improve += 1
        print(f"Epoch {epoch} done. Train: {train_mean:.4f} | Val: {val_loss:.4f} (best {best_val:.4f}) | AUROC_macro: {val_metrics.get('auroc_macro', float('nan')):.4f}")

        if epoch > 5:
            scheduler.step(val_loss)
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping tras {args.early_stopping_patience} épocas sin mejora")
            break

    # Evaluación en test
    best_ckpt = os.path.join(args.exp_dir, 'ckpt_best.pt')
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state['model'])
    test_loss, test_metrics = evaluate_v3(
        model, te_dl, fine_code_to_idx, coarse_groups, base_loss_fn, compute_stats=True,
        use_dice_on_fine=getattr(args, 'use_dice_on_fine', False), dice_weight=getattr(args, 'dice_weight', 0.5)
    )
    print(f"Test loss: {test_loss:.4f} | AUROC_macro: {test_metrics.get('auroc_macro', float('nan')):.4f} | AUPRC_macro: {test_metrics.get('auprc_macro', float('nan')):.4f} | F1_macro: {test_metrics.get('f1_macro', float('nan')):.4f}")
    tm = dict(test_metrics)
    y_true_test = tm.pop('y_true', None)
    y_prob_test = tm.pop('y_prob', None)
    with open(os.path.join(args.exp_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'loss': test_loss, **tm}, f, ensure_ascii=False, indent=2)
    if y_true_test is not None and y_prob_test is not None:
        np.savez(os.path.join(args.exp_dir, 'test_predictions.npz'), y_true=y_true_test, y_prob=y_prob_test)
    print('Entrenamiento completo.')


if __name__ == '__main__':
    main()


