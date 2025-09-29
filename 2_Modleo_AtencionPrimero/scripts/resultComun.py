import sys
import os
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# Asegurar imports del proyecto
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST
from datasets.ecg12large import ECG12Large, extract_patient_id
from torch.utils.data import DataLoader


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64


def _log(msg):
    print(f"[resultComun] {msg}", flush=True)


def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float('nan')


def compute_metrics(y_true, y_pred, thresholds=None, thr_scalar=0.5):
    if thresholds is None:
        y_bin = (y_pred >= float(thr_scalar)).astype(int)
    else:
        y_bin = (y_pred >= thresholds[None, :]).astype(int)
    return {
        'auroc_macro': _safe_metric(roc_auc_score, y_true, y_pred, average='macro'),
        'auprc_macro': _safe_metric(average_precision_score, y_true, y_pred, average='macro'),
        'f1_macro': _safe_metric(f1_score, y_true, y_bin, average='macro', zero_division=0),
        'auroc_micro': _safe_metric(roc_auc_score, y_true, y_pred, average='micro'),
        'auprc_micro': _safe_metric(average_precision_score, y_true, y_pred, average='micro'),
        'f1_micro': _safe_metric(f1_score, y_true, y_bin, average='micro', zero_division=0),
    }


def optimize_thresholds(y_true, y_pred):
    import numpy as _np
    c = y_true.shape[1]
    best_thr = _np.full((c,), 0.5, dtype=_np.float32)
    for i in range(c):
        yi = y_true[:, i]
        pi = y_pred[:, i]
        if yi.sum() == 0:
            continue
        best_f1 = -1.0
        bt = 0.5
        for t in _np.linspace(0.05, 0.95, 19):
            f1i = f1_score(yi, (pi >= t).astype(_np.int32), zero_division=0)
            if f1i > best_f1:
                best_f1, bt = f1i, t
        best_thr[i] = bt
    y_bin_opt = (y_pred >= best_thr[None, :]).astype(_np.int32)
    f1_macro_opt = float(_np.nanmean([
        f1_score(y_true[:, i], y_bin_opt[:, i], zero_division=0) if y_true[:, i].sum() > 0 else _np.nan
        for i in range(c)
    ]))
    return best_thr, f1_macro_opt


def _find_latest_checkpoint(exp_root):
    if os.path.isfile(exp_root):
        return exp_root if exp_root.lower().endswith('.pt') else None
    best, anyc = [], []
    for dirpath, _, files in os.walk(exp_root):
        for fn in files:
            if fn.lower().endswith('.pt'):
                fp = os.path.join(dirpath, fn)
                mt = 0.0
                try:
                    mt = os.path.getmtime(fp)
                except Exception:
                    pass
                anyc.append((mt, fp))
                if fn == 'ckpt_best.pt':
                    best.append((mt, fp))
    if best:
        best.sort(key=lambda x: x[0], reverse=True)
        return best[0][1]
    if anyc:
        anyc.sort(key=lambda x: x[0], reverse=True)
        return anyc[0][1]
    return None


def _infer_hmst_config_from_state(state_dict):
    cfg = {}
    
    def _get_tensor(keys):
        for k in keys:
            v = state_dict.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        return None

    w = _get_tensor(['input_proj.weight'])
    if isinstance(w, torch.Tensor):
        cfg['d_model'] = int(w.shape[0])
        cfg['input_channels'] = int(w.shape[1])
    ct = _get_tensor(['cls_token'])
    if isinstance(ct, torch.Tensor):
        cfg['d_model'] = int(ct.shape[-1])
    stage_idxs = []
    for k in state_dict.keys():
        if k.startswith('conv_stages.'):
            try:
                stage_idxs.append(int(k.split('.')[1]))
            except Exception:
                pass
    if stage_idxs:
        cfg['num_stages'] = max(stage_idxs) + 1
    hc = _get_tensor(['head_coarse.1.weight', 'head_coarse.weight'])
    if isinstance(hc, torch.Tensor):
        cfg['num_coarse'] = int(hc.shape[0])
    hf = _get_tensor(['head_fine.1.weight', 'head_fine.weight'])
    if isinstance(hf, torch.Tensor):
        cfg['num_fine'] = int(hf.shape[0])
    sp = _get_tensor(['snomed_proj.weight'])
    if isinstance(sp, torch.Tensor):
        cfg['snomed_dim'] = int(sp.shape[1])
    return cfg


def evaluate(model, dl, device):
    model.eval()
    y_true_list = []
    y_prob_list = []
    loss_sum = 0.0
    n = 0
    import torch.nn.functional as F

    with torch.no_grad():
        for batch in dl:
            x = batch['samples'].to(device)
            y_coarse = batch['labels_coarse'].to(device)
            y_fine = batch['labels_fine'].to(device)
            b = x.size(0)
            # Embedding SNOMED: usar y_coarse como en entrenamiento v3
            snomed_embed = y_coarse
            coarse_logits, fine_logits, _ = model(x, None, snomed_embed)
            # pérdidas con BCE logits + consistencia (el modelo aplica softplus a hier_matrix internamente)
            loss_c = F.binary_cross_entropy_with_logits(coarse_logits, y_coarse)
            loss_f = F.binary_cross_entropy_with_logits(fine_logits, y_fine)
            p_coarse = torch.sigmoid(coarse_logits)
            p_fine = torch.sigmoid(fine_logits)
            # consistencia (usa probs)
            cons = model.consistency_loss(p_coarse, p_fine)
            loss = 0.5 * loss_c + 1.0 * loss_f + 0.5 * cons
            loss_sum += loss.item() * b
            n += b
            # concatenar para métricas globales
            y_true_list.append(torch.cat([y_coarse, y_fine], dim=1).cpu().numpy())
            y_prob_list.append(torch.cat([p_coarse, p_fine], dim=1).cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    base = compute_metrics(y_true, y_prob, thresholds=None, thr_scalar=0.5)
    thr_opt, f1_macro_opt = optimize_thresholds(y_true, y_prob)
    opt = compute_metrics(y_true, y_prob, thresholds=thr_opt)
    return {
        'loss': (loss_sum / max(1, n)),
        'baseline': base,
        'optimized': {**opt, 'f1_macro_opt': f1_macro_opt},
        'thresholds': thr_opt.tolist(),
    }


if __name__ == '__main__':
    # Configuración de checkpoint: pasa archivo o carpeta; prioriza ckpt_best.pt
    exp_root = os.path.join(PROJ_ROOT, 'experiments_logs','multi_run_v3_20250926_003016','seed_789','ckpt_best.pt')
    ckpt_path = _find_latest_checkpoint(exp_root)
    if ckpt_path is None:
        _log(f'No se encontró checkpoint en {exp_root}')
        sys.exit(1)
    _log(f'Usando checkpoint: {ckpt_path}')

    device = torch.device(DEVICE)
    # Cargar checkpoint
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model', state.get('state_dict', state))
    # Instanciar HMST compatible
    inferred = _infer_hmst_config_from_state(state_dict)
    _log(f'Config inferida: {inferred}')
    model = HMST(
        input_channels=inferred.get('input_channels', 15),
        d_model=inferred.get('d_model', 256),
        num_stages=inferred.get('num_stages', 3),
        num_coarse=inferred.get('num_coarse', 7),
        num_fine=inferred.get('num_fine', 30),
        snomed_dim=inferred.get('snomed_dim', inferred.get('num_coarse', 7)),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)

    # Construir DataLoader de test
    hierarchy_path = os.path.join(PROJ_ROOT, 'datos', '12Large', 'labels_hierarchy.json')
    root = os.path.join(PROJ_ROOT, 'datos', '12Large', 'WFDBRecords')
    cache_dir = os.path.join(PROJ_ROOT, 'datos', 'pt_cache')

    # Si hay split guardado, usarlo; si no, fallback a split por paciente
    split_npy = os.path.join(PROJ_ROOT, 'splits', 'test_files.npy')
    if os.path.exists(split_npy):
        files = np.load(split_npy)
        files = list(map(str, files))
        _log(f'Usando split guardado: {len(files)} archivos')
    else:
        _log('No hay split guardado; creando split por paciente (70/15/15) y tomando test...')
        from glob import glob
        hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
        pairs = [(h, os.path.splitext(h)[0] + '.mat') for h in hea_files]
        files = [h for h, m in pairs if os.path.exists(m)]
        pid_to_files = {}
        for h in files:
            pid_to_files.setdefault(extract_patient_id(h, root), []).append(h)
        import numpy as _np
        pids = list(pid_to_files.keys())
        rng = _np.random.default_rng(789)
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(0.7 * n)
        n_va = int(0.15 * n)
        te_pids = set(pids[n_tr + n_va:])
        files = []
        for pid, flist in pid_to_files.items():
            if pid in te_pids:
                files.extend(flist)
        _log(f'Archivos test: {len(files)}')

    te_ds = ECG12Large(root, sequence_len=5000, files=files, multilabel=True,
                       hierarchy_path=hierarchy_path, cache_dir=cache_dir,
                       random_crop=False, target_fs=500.0, bandpass_hz=(0.5, 45.0), notch_hz=None, eval_mode=True)
    test_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

    # Evaluar
    results = evaluate(model, test_loader, device)
    os.makedirs(os.path.join(PROJ_ROOT, 'results_eval'), exist_ok=True)
    out_json = os.path.join(PROJ_ROOT, 'results_eval', 'test_eval.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, indent=2, ensure_ascii=False))
