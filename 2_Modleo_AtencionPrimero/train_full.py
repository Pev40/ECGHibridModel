import os
import json
import math
import argparse
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ModeloNuevo import ECGHybridVariableBeforeBiTrans
from datasets.wfdb_dataset import WFDBECGDataset
from losses.asymmetric_loss import AsymmetricLossMultiLabel


def set_seed(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(sequence_len, hierarchy_path, batch_size, workers, cache_dir=None):
    root = os.path.join('datos', 'WFDBRecords')
    hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    # split por archivos (ideal: por paciente si hay ID)
    n = len(hea_files)
    n_tr = int(0.9 * n)
    tr_files = hea_files[:n_tr]
    va_files = hea_files[n_tr:]

    tr_ds = WFDBECGDataset(root, sequence_len=sequence_len, files=tr_files,
                           multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir)
    va_ds = WFDBECGDataset(root, sequence_len=sequence_len, files=va_files,
                           multilabel=True, hierarchy_path=hierarchy_path, cache_dir=cache_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = device.type == 'cuda'
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                       pin_memory=pin, persistent_workers=(workers>0), prefetch_factor=2 if workers>0 else None)
    return tr_dl, va_dl


def evaluate(model, dl, fine_code_to_idx, coarse_groups, asl):
    device = next(model.parameters()).device
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in dl:
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            logits_coarse, logits_fine = model(x)
            loss_coarse = asl(logits_coarse, y_coarse)
            loss_fine = asl(logits_fine, y_fine)
            # consistencia
            max_per_group = []
            for g, codes in coarse_groups.items():
                idxs = [fine_code_to_idx.get(c, None) for c in codes]
                idxs = [i for i in idxs if i is not None]
                if len(idxs) == 0:
                    max_per_group.append(torch.zeros(y_fine.shape[0], device=device))
                else:
                    max_per_group.append(y_fine[:, idxs].max(dim=1).values)
            max_per_group = torch.stack(max_per_group, dim=1)
            cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group)
            loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons
            loss_sum += loss.item() * x.size(0)
            count += x.size(0)
    return loss_sum / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=os.path.join('datos','pt_cache'))
    parser.add_argument('--exp_dir', type=str, default=os.path.join('experiments_logs','full_run'))
    args = parser.parse_args()

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
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
    tr_dl, va_dl = build_loaders(args.sequence_len, hierarchy_path, args.batch_size, args.workers, cache_dir=args.cache_dir)

    # modelo
    configs = type('Cfg', (), dict(
        input_channels=12, sequence_len=args.sequence_len, kernel_size=8, stride=1, dropout=0.2,
        mid_channels=32, final_out_channels=128, trans_dim=32, num_heads=4, num_leads=12,
        num_fine=len(fine_codes), num_coarse=len(coarse_groups)
    ))
    model = ECGHybridVariableBeforeBiTrans(configs, {"feature_dim": 128}).to(device)

    # optim / loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    asl = AsymmetricLossMultiLabel(gamma_pos=0, gamma_neg=4, clip=0.05)
    scaler = GradScaler(enabled=args.mixed_precision and device.type=='cuda')

    os.makedirs(args.exp_dir, exist_ok=True)
    log_path = os.path.join(args.exp_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,val_loss\n')

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(tr_dl, desc=f'Epoch {epoch}/{args.epochs}')
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            x = batch['samples'].to(device, non_blocking=True)
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            with autocast(enabled=args.mixed_precision and device.type=='cuda'):
                logits_coarse, logits_fine = model(x)
                loss_coarse = asl(logits_coarse, y_coarse)
                loss_fine = asl(logits_fine, y_fine)
                # consistencia
                max_per_group = []
                for g, codes in coarse_groups.items():
                    idxs = [fine_code_to_idx.get(c, None) for c in codes]
                    idxs = [i for i in idxs if i is not None]
                    if len(idxs) == 0:
                        max_per_group.append(torch.zeros(y_fine.shape[0], device=device))
                    else:
                        max_per_group.append(y_fine[:, idxs].max(dim=1).values)
                max_per_group = torch.stack(max_per_group, dim=1)
                cons = nn.functional.binary_cross_entropy_with_logits(logits_coarse, max_per_group)
                loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        # validación
        val_loss = evaluate(model, va_dl, fine_code_to_idx, coarse_groups, asl)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{loss.item():.6f},{val_loss:.6f}\n')
        # checkpoint
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                   os.path.join(args.exp_dir, f'ckpt_epoch_{epoch}.pt'))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch},
                       os.path.join(args.exp_dir, 'ckpt_best.pt'))
        print(f'Epoch {epoch} done. Val loss: {val_loss:.4f} (best {best_val:.4f})')

    print('Entrenamiento completo.')


if __name__ == '__main__':
    main()


