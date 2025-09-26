import os
import sys
import re
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Asegura raíz del proyecto en sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.ecg12large import _parse_header_fs, _apply_filters, _resample_if_needed
from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST

try:
    from scipy.io import loadmat
except Exception as e:
    loadmat = None


def parse_header_labels(hea_path):
    labels = []
    with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if ln.startswith('#Dx:'):
                txt = ln.replace('#Dx:', '').strip()
                parts = re.split(r'[,:.;\s]+', txt)
                labels = [p for p in parts if p]
                break
    return labels


def load_hierarchy(hierarchy_path):
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    coarse_groups = hier['coarse_groups']
    coarse_names = list(coarse_groups.keys())
    return fine_codes, coarse_groups, coarse_names


def build_model(num_coarse, num_fine, d_model=256, heads=8, layers=6, stages=3, dropout=0.2, device='cpu'):
    model = HMST(
        input_channels=12+3,
        d_model=int(d_model),
        nhead=int(heads),
        num_layers=int(layers),
        num_stages=int(stages),
        num_coarse=int(num_coarse),
        num_fine=int(num_fine),
        dropout=float(dropout),
        snomed_dim=int(num_coarse),
    ).to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model


def normalize_zscore(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mean) / std


def plot_ecg(sig, save_path, title):
    # sig: [12, T]
    rows, cols = 4, 3
    fig, axs = plt.subplots(rows, cols, figsize=(15, 9), sharex=True)
    axs = axs.ravel()
    for i in range(12):
        axs[i].plot(sig[i], lw=0.8)
        axs[i].set_title(f'Lead {i+1}', fontsize=9)
        axs[i].grid(True, ls='--', lw=0.4)
    plt.suptitle(title)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Inferencia rápida v3 en un .hea/.mat')
    parser.add_argument('--hea', type=str, default='', help='Ruta al archivo .hea')
    parser.add_argument('--data_root', type=str, default=os.path.join('datos','12Large','WFDBRecords'))
    parser.add_argument('--hierarchy', type=str, default=os.path.join('datos','12Large','labels_hierarchy.json'))
    parser.add_argument('--ckpt', type=str, default=os.path.join('experiments_logs','full_run_v3','ckpt_best.pt'))
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--target_fs', type=float, default=500.0)
    parser.add_argument('--bandpass_low', type=float, default=0.5)
    parser.add_argument('--bandpass_high', type=float, default=45.0)
    parser.add_argument('--notch_hz', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='quick_infer_v3')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Seleccionar .hea
    hea_path = args.hea
    if not hea_path:
        # buscar el primero disponible en data_root
        files = [f for f in os.listdir(args.data_root) if f.endswith('.hea')]
        if len(files) == 0:
            raise FileNotFoundError('No se encontraron .hea en data_root')
        hea_path = os.path.join(args.data_root, files[0])
    mat_path = os.path.splitext(hea_path)[0] + '.mat'
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f'No existe .mat para {hea_path}')

    # Cargar jerarquía y modelo
    fine_codes, coarse_groups, coarse_names = load_hierarchy(args.hierarchy)
    model = build_model(len(coarse_names), len(fine_codes), device=device)
    model = load_checkpoint(model, args.ckpt, device)
    model.eval()

    # Cargar señal
    if loadmat is None:
        raise RuntimeError('scipy no disponible: instala scipy para leer .mat')
    m = loadmat(mat_path)
    x = None
    for key in ('val', 'data', 'signal'):
        if key in m:
            x = np.asarray(m[key])
            break
    if x is None:
        raise RuntimeError(f'No se encontró variable de señal en {mat_path}')
    if x.shape[0] < x.shape[1]:
        # comúnmente [leads, T]
        sig = x
    else:
        sig = x.T
    fs = _parse_header_fs(hea_path)
    sig, fs_eff = _resample_if_needed(sig, fs, args.target_fs)
    sig = _apply_filters(sig, fs_eff or args.target_fs, band=(args.bandpass_low, args.bandpass_high), notch_hz=(args.notch_hz if args.notch_hz in (50,60) else None))
    sig = normalize_zscore(sig)
    # recorte/pad
    if sig.shape[1] >= args.sequence_len:
        start = max(0, (sig.shape[1] - args.sequence_len)//2)
        sig = sig[:, start:start+args.sequence_len]
    else:
        pad = np.zeros((sig.shape[0], args.sequence_len - sig.shape[1]), dtype=sig.dtype)
        sig = np.concatenate([sig, pad], axis=1)

    # Forward
    x_t = torch.from_numpy(sig).float().unsqueeze(0).to(device)
    wide_feats = torch.zeros(1, 3, device=device, dtype=x_t.dtype)
    # para inferencia aislada, snomed_embed=0 (o podríamos derivar coarse GT si se quiere)
    snomed_embed = torch.zeros(1, len(coarse_names), device=device, dtype=x_t.dtype)
    with torch.no_grad():
        logits_coarse, logits_fine, _ = model(x_t, wide_feats, snomed_embed, use_attn=False)
        p_coarse = torch.sigmoid(logits_coarse).cpu().numpy().squeeze(0)
        p_fine = torch.sigmoid(logits_fine).cpu().numpy().squeeze(0)

    # Etiquetas verdaderas del header
    gt_labels = parse_header_labels(hea_path)

    # Binarios a umbral
    thr = float(args.threshold)
    y_fine_bin = (p_fine >= thr).astype(np.int32)
    # Mapeo de GT a índices fine (cuando coincidan con códigos SNOMED de la jerarquía)
    fine_code_to_idx = {c:i for i,c in enumerate(fine_codes)}
    gt_idx_fine = [fine_code_to_idx[c] for c in gt_labels if c in fine_code_to_idx]

    # Métricas simples por muestra (fine)
    y_true_vec = np.zeros_like(p_fine, dtype=np.int32)
    for i in gt_idx_fine:
        y_true_vec[i] = 1
    tp = int(np.sum((y_true_vec == 1) & (y_fine_bin == 1)))
    fp = int(np.sum((y_true_vec == 0) & (y_fine_bin == 1)))
    fn = int(np.sum((y_true_vec == 1) & (y_fine_bin == 0)))
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Salidas
    base = os.path.splitext(os.path.basename(hea_path))[0]
    plot_path = os.path.join(args.out_dir, f'{base}_ecg.png')
    plot_ecg(sig, plot_path, title=base)

    # Top-k fine
    order = np.argsort(-p_fine)
    topk = order[:int(args.topk)]
    lines = []
    lines.append(f'Registro: {hea_path}')
    lines.append('Etiquetas GT: ' + ', '.join(gt_labels) if len(gt_labels)>0 else 'Etiquetas GT: (ninguna/leídas)')
    lines.append(f'Umbral: {thr:0.2f}')
    lines.append(f'Métricas fine: TP={tp} FP={fp} FN={fn} | Precision={prec:0.3f} Recall={rec:0.3f}')
    lines.append('Top-K fine:')
    for i in topk:
        mark = '✓' if i in gt_idx_fine else ' '
        lines.append(f'  {fine_codes[i]:<10} p={p_fine[i]:0.3f} {mark}')
    # Coarse ordenado
    c_pairs = [(coarse_names[i], float(p_coarse[i])) for i in range(len(coarse_names))]
    c_pairs.sort(key=lambda t: t[1], reverse=True)
    lines.append('Coarse (ordenado):')
    for name, p in c_pairs:
        lines.append(f'  {name:<12}: {p:0.3f}')

    txt_path = os.path.join(args.out_dir, f'{base}_infer.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Guardado plot en:', plot_path)
    print('Guardado resumen en:', txt_path)


if __name__ == '__main__':
    main()


