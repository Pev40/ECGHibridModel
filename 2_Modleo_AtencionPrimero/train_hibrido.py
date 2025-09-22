import os
import argparse
import random
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ModeloNuevo import ECGHybridVariableBeforeBiTrans
from datasets.wfdb_dataset import WFDBECGDataset, build_code_map


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleConfigs:
    def __init__(self, num_fine, num_coarse, input_channels=12, sequence_len=5000, trans_dim=32, num_heads=4):
        self.num_fine = num_fine
        self.num_coarse = num_coarse
        self.input_channels = input_channels
        self.sequence_len = sequence_len
        self.kernel_size = 8
        self.stride = 1
        self.dropout = 0.2
        self.mid_channels = 32
        self.final_out_channels = 128
        self.trans_dim = trans_dim
        self.num_heads = num_heads
        self.num_leads = input_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    print(f"Device: {device}")
    if device.type == 'cuda':
        try:
            print('GPU:', torch.cuda.get_device_name(0))
        except Exception:
            pass
    else:
        print('ADVERTENCIA: CUDA no disponible. Instala PyTorch con soporte CUDA (usa setup_env.ps1 -CUDA).')

    root = os.path.join('datos', 'WFDBRecords')
    hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    hea_files = hea_files[:5000]
    # cargar jerarquía
    import json
    with open(os.path.join('datos', 'labels_hierarchy.json'), 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    num_fine = len(fine_codes)
    coarse_names = list(hier['coarse_groups'].keys())
    num_coarse = len(coarse_names)

    configs = SimpleConfigs(num_fine=num_fine, num_coarse=num_coarse)
    hparams = {"feature_dim": 128}

    # split simple
    n = len(hea_files)
    n_tr = int(0.8 * n)
    tr_files = hea_files[:n_tr]
    va_files = hea_files[n_tr:]

    tr_ds = WFDBECGDataset(root, sequence_len=configs.sequence_len, files=tr_files,
                           multilabel=True, hierarchy_path=os.path.join('datos', 'labels_hierarchy.json'))
    va_ds = WFDBECGDataset(root, sequence_len=configs.sequence_len, files=va_files,
                           multilabel=True, hierarchy_path=os.path.join('datos', 'labels_hierarchy.json'))

    batch_size = 64
    num_workers = 8
    use_pin = device.type == 'cuda'
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                       pin_memory=use_pin, persistent_workers=(num_workers>0), prefetch_factor=2 if num_workers>0 else None)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=use_pin, persistent_workers=(num_workers>0), prefetch_factor=2 if num_workers>0 else None)

    model = ECGHybridVariableBeforeBiTrans(configs, hparams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    # entrenamiento corto: 2 épocas
    EPOCHS = 2
    with open(os.path.join('datos', 'labels_hierarchy.json'), 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    fine_code_to_idx = {c:i for i,c in enumerate(fine_codes)}
    coarse_groups = hier['coarse_groups']

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            x = batch['samples'].to(device, non_blocking=True)  # [B,C,T]
            y_coarse = batch['labels_coarse'].to(device, non_blocking=True)
            y_fine = batch['labels_fine'].to(device, non_blocking=True)
            opt.zero_grad()
            logits_coarse, logits_fine = model(x)
            # pérdidas
            loss_coarse = bce(logits_coarse, y_coarse)
            loss_fine = bce(logits_fine, y_fine)
            # consistencia jerárquica: coarse ≥ max(fines del grupo)
            max_per_group = []
            for g, codes in coarse_groups.items():
                idxs = [fine_code_to_idx.get(c, None) for c in codes]
                idxs = [i for i in idxs if i is not None]
                if len(idxs) == 0:
                    max_per_group.append(torch.zeros(y_fine.shape[0], device=device))
                else:
                    max_per_group.append(y_fine[:, idxs].max(dim=1).values)
            max_per_group = torch.stack(max_per_group, dim=1)  # [B, G]
            cons = bce(logits_coarse, max_per_group)
            loss = 0.5*loss_coarse + 1.0*loss_fine + 0.5*cons
            loss.backward()
            opt.step()
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'coarse': f'{loss_coarse.item():.3f}',
                'fine': f'{loss_fine.item():.3f}',
                'cons': f'{cons.item():.3f}'
            })

    print('Entrenamiento corto completado')


if __name__ == '__main__':
    main()


