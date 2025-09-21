import os
import random
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader

from ModeloNuevo import ECGHybridVariableBeforeBiTrans
from datasets.wfdb_dataset import WFDBECGDataset, build_code_map


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleConfigs:
    def __init__(self, num_classes, input_channels=12, sequence_len=5000, trans_dim=32, num_heads=4):
        self.num_classes = num_classes
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
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = os.path.join('datos', 'WFDBRecords')
    hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    hea_files = hea_files[:2000]
    code_to_idx = build_code_map(hea_files, top_k=10, save_path=os.path.join('datos', 'code_map.json'))

    num_classes = len(code_to_idx)
    configs = SimpleConfigs(num_classes=num_classes)
    hparams = {"feature_dim": 128}

    # split simple
    n = len(hea_files)
    n_tr = int(0.8 * n)
    tr_files = hea_files[:n_tr]
    va_files = hea_files[n_tr:]

    tr_ds = WFDBECGDataset(root, sequence_len=configs.sequence_len, code_to_idx=code_to_idx, files=tr_files)
    va_ds = WFDBECGDataset(root, sequence_len=configs.sequence_len, code_to_idx=code_to_idx, files=va_files)

    tr_dl = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=0)

    model = ECGHybridVariableBeforeBiTrans(configs, hparams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    # dry-run 1 epoch
    model.train()
    for step, batch in enumerate(tr_dl):
        x = batch['samples'].to(device)  # [B,C,T]
        y = batch['labels'].to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f'step {step} loss {loss.item():.4f}')
        if step >= 50:
            break

    print('Dry-run completado')


if __name__ == '__main__':
    main()


