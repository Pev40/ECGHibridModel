import os
import json
import math
import argparse
from glob import glob

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ModeloNuevo import ECGHybridVariableBeforeBiTrans
from datasets.wfdb_dataset import WFDBECGDataset, extract_patient_id
from losses.asymmetric_loss import AsymmetricLossMultiLabel


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


def build_loaders(sequence_len, hierarchy_path, batch_size, workers, cache_dir=None,
                 target_fs=500.0, bandpass_hz=(0.5, 45.0), notch_hz=None, seed=42,
                 use_sampler=True, sampler_power=1.0,
                 aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
                 aug_amp_scale_min=1.0, aug_amp_scale_max=1.0):
    root = os.path.join('datos', 'WFDBRecords')
    tr_files, va_files, te_files = build_patient_splits(root, seed=seed)

    tr_ds = WFDBECGDataset(root, sequence_len=sequence_len, files=tr_files,
                           multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                           random_crop=True, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=False,
                           aug_jitter_std=aug_jitter_std, aug_shift_max=aug_shift_max,
                           aug_lead_drop_prob=aug_lead_drop_prob, aug_amp_scale_min=aug_amp_scale_min,
                           aug_amp_scale_max=aug_amp_scale_max)
    va_ds = WFDBECGDataset(root, sequence_len=sequence_len, files=va_files,
                           multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                           random_crop=False, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=True)
    te_ds = WFDBECGDataset(root, sequence_len=sequence_len, files=te_files,
                           multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                           random_crop=False, target_fs=target_fs, bandpass_hz=bandpass_hz, notch_hz=notch_hz, eval_mode=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = device.type == 'cuda'

    # WeightedRandomSampler según frecuencia de etiquetas fine en el dataset de entrenamiento
    y_fine_list = [s[2] for s in tr_ds.samples]
    if use_sampler and len(y_fine_list) > 0 and isinstance(y_fine_list[0], np.ndarray):
        y_mat = np.stack(y_fine_list, axis=0)
        class_freq = y_mat.mean(axis=0) + 1e-6
        inv_freq = (1.0 / class_freq) ** float(max(0.0, sampler_power))
        inv_freq = inv_freq / inv_freq.sum()
        sample_w = (y_mat * inv_freq).sum(axis=1)
        sample_w = sample_w / (sample_w.mean() + 1e-8)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_w).float(), num_samples=len(tr_ds), replacement=True)
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle_train, sampler=sampler, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    return tr_dl, va_dl, te_dl


def evaluate(model, dl, fine_code_to_idx, coarse_groups, asl, compute_stats=True):
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
            logits_coarse, logits_fine = model(x)
            loss_coarse = asl(logits_coarse, y_coarse)
            loss_fine = asl(logits_fine, y_fine)
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
    # micro
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
    metrics['y_true'] = y_true
    metrics['y_prob'] = y_prob
    return mean_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Épocas sin mejora en val_loss para detener el entrenamiento. 0 para desactivar.')
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
    parser.add_argument('--gamma_pos', type=float, default=0.0)
    parser.add_argument('--gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_clip', type=float, default=0.05)
    parser.add_argument('--sampler_power', type=float, default=1.0)
    parser.add_argument('--no_sampler', action='store_true')
    # Augmentaciones
    parser.add_argument('--aug_jitter_std', type=float, default=0.0)
    parser.add_argument('--aug_shift_max', type=int, default=0)
    parser.add_argument('--aug_lead_drop_prob', type=float, default=0.0)
    parser.add_argument('--aug_amp_scale_min', type=float, default=1.0)
    parser.add_argument('--aug_amp_scale_max', type=float, default=1.0)
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

    # jerarquía
    hierarchy_path = os.path.join('datos', 'labels_hierarchy.json')
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    coarse_groups = hier['coarse_groups']
    fine_code_to_idx = {c:i for i,c in enumerate(fine_codes)}

    # loaders
    tr_dl, va_dl, te_dl = build_loaders(
        args.sequence_len, hierarchy_path, args.batch_size, args.workers,
        cache_dir=args.cache_dir,
        target_fs=args.target_fs,
        bandpass_hz=(args.bandpass_low, args.bandpass_high),
        notch_hz=(args.notch_hz if args.notch_hz in (50,60) else None),
        seed=args.seed,
        use_sampler=(not args.no_sampler),
        sampler_power=args.sampler_power,
        aug_jitter_std=args.aug_jitter_std,
        aug_shift_max=args.aug_shift_max,
        aug_lead_drop_prob=args.aug_lead_drop_prob,
        aug_amp_scale_min=args.aug_amp_scale_min,
        aug_amp_scale_max=args.aug_amp_scale_max
    )

    # modelo
    configs = type('Cfg', (), dict(
        input_channels=12, sequence_len=args.sequence_len, kernel_size=8, stride=1, dropout=args.dropout,
        mid_channels=32, final_out_channels=128, trans_dim=32, num_heads=4, num_leads=12,
        num_fine=len(fine_codes), num_coarse=len(coarse_groups)
    ))
    model = ECGHybridVariableBeforeBiTrans(configs, {"feature_dim": 128}).to(device)

    # optim / loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, threshold=1e-4, cooldown=0, min_lr=1e-6)
    asl = AsymmetricLossMultiLabel(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, clip=args.asl_clip)
    scaler = GradScaler(enabled=args.mixed_precision and device.type=='cuda')

    os.makedirs(args.exp_dir, exist_ok=True)
    log_path = os.path.join(args.exp_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,val_loss,val_auroc_macro,val_auprc_macro,val_f1_macro,lr\n')

    best_val = math.inf
    epochs_no_improve = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_dl, desc=f'Epoch {epoch}/{args.epochs}')
        opt.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_n = 0
        for step, batch in enumerate(pbar, start=1):
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            with autocast_cuda(args.mixed_precision and device.type=='cuda'):
                logits_coarse, logits_fine = model(x)
                loss_coarse = asl(logits_coarse, y_coarse)
                loss_fine = asl(logits_fine, y_fine)
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

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            train_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        # validación
        val_loss, val_metrics = evaluate(model, va_dl, fine_code_to_idx, coarse_groups, asl, compute_stats=True)
        train_mean = train_sum / max(1, train_n)
        current_lr = opt.param_groups[0]['lr']
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_mean:.6f},{val_loss:.6f},{val_metrics.get('auroc_macro', float('nan')):.6f},{val_metrics.get('auprc_macro', float('nan')):.6f},{val_metrics.get('f1_macro', float('nan')):.6f},{current_lr:.8f}\n")
        # checkpoint
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                   os.path.join(args.exp_dir, f'ckpt_epoch_{epoch}.pt'))
        if val_loss < best_val - args.early_stopping_min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                       os.path.join(args.exp_dir, 'ckpt_best.pt'))
        else:
            epochs_no_improve += 1
        
        print(f"Epoch {epoch} done. Train: {train_mean:.4f} | Val: {val_loss:.4f} (best {best_val:.4f}) | AUROC_macro: {val_metrics.get('auroc_macro', float('nan')):.4f}")

        # Actualizar el learning rate con ReduceLROnPlateau según val_loss
        scheduler.step(val_loss)

        # Comprobar Early Stopping
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping activado tras {args.early_stopping_patience} épocas sin mejora ≥ {args.early_stopping_min_delta} en val_loss.")
            break

    # Evaluación en test con mejor checkpoint
    best_ckpt = os.path.join(args.exp_dir, 'ckpt_best.pt')
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state['model'])
        print('Evaluando en test con mejor checkpoint...')
        # Optimización de umbrales por clase usando validación si está disponible
        val_loss_best, val_metrics_best = evaluate(model, va_dl, fine_code_to_idx, coarse_groups, asl, compute_stats=True)
        y_true_val = val_metrics_best.get('y_true', None)
        y_prob_val = val_metrics_best.get('y_prob', None)
        class_thresholds = None
        if y_true_val is not None and y_prob_val is not None:
            try:
                import numpy as np
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

        test_loss, test_metrics = evaluate(model, te_dl, fine_code_to_idx, coarse_groups, asl, compute_stats=True)
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
        with open(os.path.join(args.exp_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({'loss': test_loss, **tm}, f, ensure_ascii=False, indent=2)
        if y_true_test is not None and y_prob_test is not None:
            np.savez(os.path.join(args.exp_dir, 'test_predictions.npz'), y_true=y_true_test, y_prob=y_prob_test, thresholds=class_thresholds)
    print('Entrenamiento completo.')


if __name__ == '__main__':
    main()


