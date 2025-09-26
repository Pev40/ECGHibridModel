import os
import json
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

# Asegura que la raíz del proyecto esté en sys.path cuando se ejecuta desde scripts/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.ecg12large import ECG12Large
from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST


def load_hierarchy(hierarchy_path):
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hier = json.load(f)
    fine_codes = hier['fine_codes']
    coarse_groups = hier['coarse_groups']
    coarse_names = list(coarse_groups.keys())
    fine_code_to_idx = {c: i for i, c in enumerate(fine_codes)}
    return hier, fine_codes, coarse_groups, coarse_names, fine_code_to_idx


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
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    return model


def plot_ecg_12leads(x_np, title, save_path=None):
    # x_np: [12, T]
    leads = x_np.shape[0]
    cols = 3
    rows = int(np.ceil(leads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), sharex=True)
    axes = axes.reshape(rows, cols)
    for i in range(leads):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.plot(x_np[i], lw=0.8)
        ax.set_title(f'Lead {i+1}', fontsize=9)
        ax.grid(True, ls='--', lw=0.4)
    for j in range(leads, rows*cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r, c])
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def format_topk_labels(y_true_vec, y_prob_vec, class_names, k=10, thr=0.5):
    # y_true_vec, y_prob_vec: shape [C]
    probs = [(i, float(y_prob_vec[i])) for i in range(len(class_names))]
    probs.sort(key=lambda t: t[1], reverse=True)
    top = probs[:k]
    lines = []
    for idx, p in top:
        true_mark = '✓' if (idx < len(y_true_vec) and float(y_true_vec[idx]) >= 0.5) else ' '
        pred_mark = '●' if p >= thr else ' '
        lines.append(f"{class_names[idx]:<10}  p={p:0.3f}  true={true_mark} pred={pred_mark}")
    return "\n".join(lines)


def inspect_one(sample, model, device, fine_codes, coarse_names, threshold=0.5, use_attn=False):
    x = sample['samples'].unsqueeze(0).to(device)
    y_fine = sample.get('labels_fine')
    y_coarse = sample.get('labels_coarse')
    hea = sample.get('hea')

    b = x.size(0)
    wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
    snomed_embed = y_coarse.unsqueeze(0).to(device) if y_coarse is not None else torch.zeros(b, len(coarse_names), device=device, dtype=x.dtype)

    model.eval()
    with torch.no_grad():
        logits_coarse, logits_fine, attn = model(x, wide_feats, snomed_embed, use_attn=use_attn)
        p_fine = torch.sigmoid(logits_fine).squeeze(0).cpu().numpy()
        p_coarse = torch.sigmoid(logits_coarse).squeeze(0).cpu().numpy()

    return {
        'hea': hea,
        'x': sample['samples'].cpu().numpy(),
        'y_fine': y_fine.cpu().numpy() if y_fine is not None else None,
        'y_coarse': y_coarse.cpu().numpy() if y_coarse is not None else None,
        'p_fine': p_fine,
        'p_coarse': p_coarse,
        'attn': attn,
        'patient_id': sample.get('patient_id')
    }


def _multilabel_prf(y_true_vec, y_prob_vec, thr):
    y_true_b = (y_true_vec >= 0.5).astype(np.int32)
    y_pred_b = (y_prob_vec >= float(thr)).astype(np.int32)
    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1


def _hit_at_k(y_true_vec, y_prob_vec, k):
    y_true_pos = set(np.where(y_true_vec >= 0.5)[0].tolist())
    order = np.argsort(-y_prob_vec)
    topk = order[:int(k)]
    hits = len([i for i in topk if i in y_true_pos])
    any_hit = 1 if hits > 0 else 0
    denom = max(1, len(y_true_pos))
    recall_at_k = hits / denom
    return hits, any_hit, recall_at_k


def compute_saliency(sample, model, device, coarse_dim, fine_topk=1, prefer_true=True):
    x = sample['samples'].unsqueeze(0).to(device)
    x.requires_grad_(True)
    y_fine = sample.get('labels_fine')
    y_coarse = sample.get('labels_coarse')

    b = x.size(0)
    wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
    if y_coarse is not None:
        snomed_embed = y_coarse.unsqueeze(0).to(device)
    else:
        snomed_embed = torch.zeros(b, int(coarse_dim), device=device, dtype=x.dtype)

    model.zero_grad(set_to_none=True)
    logits_coarse, logits_fine, _ = model(x, wide_feats, snomed_embed, use_attn=False)

    # Selección de objetivo: etiquetas verdaderas si existen, si no top-k predichas
    target = None
    if prefer_true and (y_fine is not None) and (y_fine.sum().item() > 0):
        pos_idx = torch.nonzero(y_fine >= 0.5, as_tuple=False).view(-1)
        target = logits_fine[0, pos_idx].sum()
    else:
        probs = torch.sigmoid(logits_fine[0])
        topk = torch.topk(probs, k=max(1, int(fine_topk)))
        target = logits_fine[0, topk.indices].sum()

    target.backward()
    grad = x.grad.detach().abs().squeeze(0)  # [12, T]
    grad_np = grad.cpu().numpy()
    # Saliency temporal: promedio sobre leads
    sal_time = grad_np.mean(axis=0)
    if sal_time.max() > 0:
        sal_time = sal_time / (sal_time.max() + 1e-8)
    return grad_np, sal_time


def plot_temporal_saliency(sal_time, title, save_path):
    plt.figure(figsize=(12, 3))
    plt.plot(sal_time, color='crimson', lw=1.0)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.grid(True, ls='--', lw=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_lead_time_heatmap(grad_abs, title, save_path):
    # grad_abs: [12, T]
    import seaborn as sns
    plt.figure(figsize=(12, 4))
    vmax = float(np.percentile(grad_abs, 99.0)) if np.isfinite(grad_abs).all() else None
    sns.heatmap(grad_abs, cmap='magma', cbar=True, vmin=0.0, vmax=vmax)
    plt.xlabel('Tiempo')
    plt.ylabel('Lead')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Inspección de registro(s) ECG: etiquetas reales vs predichas (v3)')
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--target_fs', type=float, default=500.0)
    parser.add_argument('--bandpass_low', type=float, default=0.5)
    parser.add_argument('--bandpass_high', type=float, default=45.0)
    parser.add_argument('--notch_hz', type=int, default=0)
    parser.add_argument('--hierarchy', type=str, default=os.path.join('datos','12Large','labels_hierarchy.json'))
    parser.add_argument('--data_root', type=str, default=os.path.join('datos','12Large','WFDBRecords'))
    parser.add_argument('--cache_dir', type=str, default=os.path.join('datos','pt_cache'))
    parser.add_argument('--ckpt', type=str, default=os.path.join('experiments_logs','full_run_v3','ckpt_best.pt'))
    parser.add_argument('--index', type=int, default=0, help='Índice del dataset a inspeccionar')
    parser.add_argument('--indices', type=str, default='', help='Lista de índices separada por comas (ej: 0,5,12)')
    parser.add_argument('--num', type=int, default=1, help='Cantidad de ejemplos a inspeccionar')
    parser.add_argument('--random', action='store_true', help='Seleccionar índices al azar')
    parser.add_argument('--hea_path', type=str, default='', help='Ruta .hea específica (opcional)')
    parser.add_argument('--save_dir', type=str, default='inspeccion_v3')
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_attn', action='store_true')
    parser.add_argument('--viz_saliency', action='store_true', help='Calcular y guardar saliency maps por muestra')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Jerarquía / nombres de clases
    hier, fine_codes, coarse_groups, coarse_names, fine_code_to_idx = load_hierarchy(args.hierarchy)

    # Dataset en modo evaluación
    ds = ECG12Large(
        args.data_root,
        sequence_len=args.sequence_len,
        files=None,
        multilabel=True,
        hierarchy_path=args.hierarchy,
        cache_dir=args.cache_dir,
        random_crop=False,
        target_fs=args.target_fs,
        bandpass_hz=(args.bandpass_low, args.bandpass_high),
        notch_hz=(args.notch_hz if args.notch_hz in (50, 60) else None),
        eval_mode=True,
    )

    # Determinar índices a evaluar
    indices = []
    if args.hea_path:
        # buscar índice del registro específico
        idx_match = None
        for i in range(len(ds)):
            rec = ds[i]
            if rec.get('hea') == args.hea_path:
                idx_match = i
                break
        if idx_match is None:
            raise FileNotFoundError(f'No se encontró el registro con hea={args.hea_path} en el dataset.')
        indices = [idx_match]
    elif args.indices:
        try:
            indices = [max(0, min(int(s.strip()), len(ds)-1)) for s in args.indices.split(',') if s.strip() != '']
        except Exception:
            raise ValueError('No se pudo parsear --indices. Usa formato 0,5,12')
    else:
        if args.num <= 1:
            indices = [max(0, min(int(args.index), len(ds) - 1))]
        else:
            all_idxs = list(range(len(ds)))
            if args.random:
                rng = np.random.default_rng()
                rng.shuffle(all_idxs)
            indices = all_idxs[:int(args.num)]

    device = torch.device(args.device)
    model = build_model(num_coarse=len(coarse_names), num_fine=len(fine_codes), device=device)
    model = load_checkpoint(model, args.ckpt, device)

    # CSV de métricas por muestra
    csv_path = os.path.join(args.save_dir, 'inspection_metrics.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        if write_header:
            writer.writerow([
                'record', 'patient_id',
                'fine_tp', 'fine_fp', 'fine_fn', 'fine_precision', 'fine_recall', 'fine_f1',
                'fine_hits_at_k', 'fine_any_hit_topk', 'fine_recall_at_k',
                'coarse_tp', 'coarse_fp', 'coarse_fn', 'coarse_precision', 'coarse_recall', 'coarse_f1',
                'coarse_hits_at_k', 'coarse_any_hit_topk', 'coarse_recall_at_k'
            ])

        for idx in indices:
            sample = ds[idx]
            out = inspect_one(sample, model, device, fine_codes, coarse_names, threshold=args.threshold, use_attn=args.use_attn)

            # Guardar ECG y TXT por muestra
            base = os.path.splitext(os.path.basename(out['hea']))[0] if out['hea'] else f'idx_{idx}'
            fig_path = os.path.join(args.save_dir, f'{base}_ecg.png')
            plot_ecg_12leads(out['x'], title=f'ECG {base}', save_path=fig_path)

            txt_lines = []
            txt_lines.append(f'Registro: {out["hea"]}')
            txt_lines.append('')
            txt_lines.append('Top-K etiquetas predichas (fine):')
            txt_lines.append(format_topk_labels(out['y_fine'] if out['y_fine'] is not None else np.zeros_like(out['p_fine']),
                                                out['p_fine'], fine_codes, k=args.topk, thr=args.threshold))
            txt_lines.append('')
            if out['y_fine'] is not None:
                true_idxs = np.where(out['y_fine'] >= 0.5)[0].tolist()
                txt_lines.append('Etiquetas verdaderas (fine): ' + ', '.join(fine_codes[i] for i in true_idxs))
            txt_lines.append('')
            txt_lines.append('Probabilidades coarse:')
            coarse_pairs = [(coarse_names[i], float(out['p_coarse'][i])) for i in range(len(coarse_names))]
            coarse_pairs.sort(key=lambda t: t[1], reverse=True)
            for name, p in coarse_pairs:
                txt_lines.append(f'  {name:<12}: {p:0.3f}')
            if args.use_attn and out['attn'] is not None:
                txt_lines.append('')
                txt_lines.append('[Info] Pesos de atención disponibles (no visualizados aún).')
            txt_path = os.path.join(args.save_dir, f'{base}_predicciones.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(txt_lines))

            # Visualizaciones de atención (placeholder) y saliency
            if args.viz_saliency:
                grad_abs, sal_time = compute_saliency(sample, model, device, coarse_dim=len(coarse_names), fine_topk=max(1, int(min(args.topk, len(fine_codes)))))
                sal_t_path = os.path.join(args.save_dir, f'{base}_saliency_time.png')
                heat_path = os.path.join(args.save_dir, f'{base}_saliency_heatmap.png')
                plot_temporal_saliency(sal_time, title=f'Saliency temporal {base}', save_path=sal_t_path)
                plot_lead_time_heatmap(grad_abs, title=f'Saliency por lead y tiempo {base}', save_path=heat_path)

            # Métricas por muestra
            y_fine = out['y_fine'] if out['y_fine'] is not None else np.zeros_like(out['p_fine'])
            y_coarse = out['y_coarse'] if out['y_coarse'] is not None else np.zeros_like(out['p_coarse'])
            p_fine = out['p_fine']
            p_coarse = out['p_coarse']

            f_tp, f_fp, f_fn, f_prec, f_rec, f_f1 = _multilabel_prf(y_fine, p_fine, args.threshold)
            f_hits, f_any_hit, f_rk = _hit_at_k(y_fine, p_fine, args.topk)
            c_tp, c_fp, c_fn, c_prec, c_rec, c_f1 = _multilabel_prf(y_coarse, p_coarse, args.threshold)
            c_hits, c_any_hit, c_rk = _hit_at_k(y_coarse, p_coarse, args.topk)

            writer.writerow([
                base, out.get('patient_id'),
                f_tp, f_fp, f_fn, f_prec, f_rec, f_f1,
                f_hits, f_any_hit, f_rk,
                c_tp, c_fp, c_fn, c_prec, c_rec, c_f1,
                c_hits, c_any_hit, c_rk
            ])

            print(f"Guardado: {fig_path} | {txt_path}")
    print('CSV de métricas por muestra:', csv_path)


if __name__ == '__main__':
    main()


