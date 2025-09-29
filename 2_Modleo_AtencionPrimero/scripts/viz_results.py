#!/usr/bin/env python3
"""
viz_results.py
Genera figuras de evaluación y explicabilidad para HMST:
- ROC/PR (macro/micro), curves por clase
- Calibration (reliability) plots + ECE
- Histogramas de probabilidades
- Metrics por clase (precision/recall/F1/support)
- TP/FP/FN counts por clase (pseudo confusion)
- Comparativa F1 (0.5 vs optimized thresholds)
- Ejemplos cualitativos: ECG 12 leads + GT vs Pred + attention heatmaps (si disponibles)

Inputs:
 - --exp_dir : directorio de la experimento (contiene test_predictions.npz, test_metrics.json, thresholds.json opcional)
 - --preds   : opcional .npz con y_true (N,C) y y_prob (N,C). Si no se provee, se busca en <exp_dir>/test_predictions.npz
 - --hierarchy : opcional labels_hierarchy.json. Por defecto: datos/12Large/labels_hierarchy.json si existe
 - --test_files : opcional path a test_files.npy (para recuperar muestras)
 - --n_examples : numero de ejemplos cualitativos
"""
import os, json, argparse, math
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, precision_recall_fscore_support,
                             f1_score, precision_score, recall_score)
from sklearn.calibration import calibration_curve
import pandas as pd
from zipfile import ZipFile
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

# Importar clases del proyecto para cómputo en vivo si faltan preds
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

try:
    from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST
    from datasets.ecg12large import ECG12Large, extract_patient_id
except Exception:
    HMST = None
    ECG12Large = None
    extract_patient_id = None

sns.set(style="whitegrid", rc={'figure.dpi':150})

def load_preds(preds_path):
    data = np.load(preds_path, allow_pickle=True)
    # supported keys: y_true, y_prob, y_true_fine, y_prob_fine
    if 'y_true' in data and 'y_prob' in data:
        return data['y_true'], data['y_prob']
    # try alternatives
    if 'y_true_fine' in data and 'y_prob_fine' in data:
        return data['y_true_fine'], data['y_prob_fine']
    # fallback: search first two arrays
    keys = list(data.keys())
    return data[keys[0]], data[keys[1]]

def ensure_outdir(base, suffix):
    od = os.path.join(base, suffix)
    os.makedirs(od, exist_ok=True)
    return od

def plot_roc_pr_macro_micro(y_true, y_prob, outdir, title_prefix=""):
    # micro/macro AUROC & AUPRC + curves
    n_classes = y_true.shape[1]
    # micro
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    prec_micro, rec_micro, _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())

    # macro averaged ROC: compute per-class then average interpolated TPR
    # We'll plot micro + per-class (subset) to avoid clutter
    plt.figure(figsize=(6,5))
    plt.plot(fpr_micro, tpr_micro, label=f"micro (AUROC={roc_auc_score(y_true, y_prob, average='micro'):.3f})")
    # plot a few random classes
    idxs = np.linspace(0, n_classes-1, min(8, n_classes)).astype(int)
    for i in idxs:
        try:
            fpr, tpr, _ = roc_curve(y_true[:,i], y_prob[:,i])
            auc = roc_auc_score(y_true[:,i], y_prob[:,i]) if (y_true[:,i].sum()>0 and (1-y_true[:,i]).sum()>0) else float('nan')
            plt.plot(fpr, tpr, alpha=0.6, label=f"c{i} (AUC={auc:.2f})")
        except Exception:
            continue
    plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title_prefix + " ROC curves (micro + sample classes)")
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_micro_sample_classes.png"))
    plt.close()

    # PR curve
    plt.figure(figsize=(6,5))
    plt.plot(rec_micro, prec_micro, label=f"micro (AUPRC={average_precision_score(y_true, y_prob, average='micro'):.3f})")
    for i in idxs:
        try:
            p, r, _ = precision_recall_curve(y_true[:,i], y_prob[:,i])
            ap = average_precision_score(y_true[:,i], y_prob[:,i]) if y_true[:,i].sum()>0 else float('nan')
            plt.plot(r, p, alpha=0.6, label=f"c{i} (AP={ap:.2f})")
        except Exception:
            continue
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title_prefix + " PR curves (micro + sample classes)")
    plt.legend(loc='lower left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_micro_sample_classes.png"))
    plt.close()

def per_class_metrics_table(y_true, y_prob, thresholds=None, outdir=None, class_names=None):
    n = y_true.shape[1]
    thr = np.full((n,), 0.5) if thresholds is None else np.asarray(thresholds)
    y_pred_bin = (y_prob >= thr[None,:]).astype(int)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred_bin, zero_division=0)
    rows = []
    for i in range(n):
        label = None
        if class_names is not None and i < len(class_names):
            try:
                label = str(class_names[i])
            except Exception:
                label = None
        rows.append({
            'class': i,
            'label': (label if label is not None else f'class_{i}'),
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'threshold': float(thr[i])
        })
    df = pd.DataFrame(rows).sort_values('f1', ascending=False)
    if outdir:
        df.to_csv(os.path.join(outdir, "per_class_metrics.csv"), index=False)
    return df

def reliability_plot(y_true, y_prob, outdir, n_bins=10):
    # Compute calibration curve across all classes (flatten)
    prob = y_prob.ravel()
    true = y_true.ravel()
    # remove NaNs where class absent? keep all
    frac_pos, mean_pred = calibration_curve(true, prob, n_bins=n_bins, strategy='uniform')
    plt.figure(figsize=(5,5))
    plt.plot(mean_pred, frac_pos, 's-', label='Reliability')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram (all classes flattened)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reliability_diagram.png"))
    plt.close()

def prob_histograms(y_prob, outdir, top_k=6):
    mean_prob = y_prob.mean(axis=0)
    idxs = np.argsort(mean_prob)[-top_k:][::-1]
    plt.figure(figsize=(12,3*top_k))
    for i, idx in enumerate(idxs):
        ax = plt.subplot(top_k,1,i+1)
        ax.hist(y_prob[:,idx], bins=40)
        ax.set_title(f"class {idx} prob distribution (mean={mean_prob[idx]:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "prob_hist_topk.png"))
    plt.close()

def confusion_counts(y_true, y_pred_bin, outdir):
    # For each class: TP, FP, FN, TN
    n = y_true.shape[1]
    rows=[]
    for i in range(n):
        yt = y_true[:,i].astype(int)
        yp = y_pred_bin[:,i].astype(int)
        tp = int(((yt==1)&(yp==1)).sum())
        fp = int(((yt==0)&(yp==1)).sum())
        fn = int(((yt==1)&(yp==0)).sum())
        tn = int(((yt==0)&(yp==0)).sum())
        rows.append({'class':i,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'support':int(yt.sum())})
    df = pd.DataFrame(rows).sort_values('support', ascending=False)
    df.to_csv(os.path.join(outdir,"per_class_confusion_counts.csv"), index=False)
    return df

def compare_f1_thresholds(y_true, y_prob, thr_opt, outdir):
    # f1 at 0.5
    thr0 = np.full((y_true.shape[1],), 0.5)
    y0 = (y_prob >= thr0[None,:]).astype(int)
    yopt = (y_prob >= thr_opt[None,:]).astype(int)
    f1_0 = []
    f1_opt = []
    for i in range(y_true.shape[1]):
        if y_true[:,i].sum()==0:
            f1_0.append(np.nan); f1_opt.append(np.nan)
        else:
            f1_0.append(f1_score(y_true[:,i], y0[:,i], zero_division=0))
            f1_opt.append(f1_score(y_true[:,i], yopt[:,i], zero_division=0))
    df = pd.DataFrame({'class':np.arange(len(f1_0)),'f1_0.5':f1_0,'f1_opt':f1_opt})
    df['delta'] = df['f1_opt'] - df['f1_0.5']
    df.sort_values('delta', ascending=False, inplace=True)
    df.to_csv(os.path.join(outdir,"f1_threshold_comparison.csv"), index=False)
    # plot top 20 delta
    top = df.head(20)
    plt.figure(figsize=(8,6))
    plt.barh(top['class'].astype(str), top['delta'])
    plt.xlabel("F1 improvement (opt - 0.5)")
    plt.title("Top classes improved by threshold optimization")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"f1_delta_top20.png"))
    plt.close()
    return df

def examples_qualitative(exp_dir, test_files_path, y_true, y_prob, thr_opt, n_examples, outdir, dataset_ctor):
    """
    Visualizar N ejemplos: plot 12 leads, GT list, Pred probs & bins (with thr_opt)
    dataset_ctor: callable(fn_list) -> dataset supporting __getitem__ returning dict with 'samples','labels_coarse','labels_fine'
    """
    if test_files_path is None or not os.path.exists(test_files_path):
        print("No test_files provided; cannot show file-level examples.")
        return
    files = np.load(test_files_path, allow_pickle=True)
    N = len(files)
    # choose examples where model made mistakes or high-confidence predictions
    # compute per-sample f1 (micro) with thr_opt
    bins = (y_prob >= thr_opt[None,:]).astype(int)
    sample_f1 = []
    for i in range(y_true.shape[0]):
        # micro f1 of sample compared to its binary labels
        try:
            si = f1_score(y_true[i], bins[i], zero_division=0, average='micro')
        except Exception:
            si = 0.0
        sample_f1.append(si)
    # pick worst and best
    worst_idx = np.argsort(sample_f1)[:n_examples//2]
    best_idx = np.argsort(sample_f1)[- (n_examples - len(worst_idx)) :][::-1]
    pick = np.concatenate([worst_idx, best_idx])
    # map to files (if shapes correspond)
    # We'll simply iterate and fetch examples by index modulo
    ds = dataset_ctor(files.tolist())
    for k, si in enumerate(pick):
        try:
            rec = ds[si]
            samples = rec['samples']  # [12, T]
            gt_coarse = rec['labels_coarse']
            gt_fine = rec['labels_fine']
            probs = y_prob[si]
            bins_s = bins[si]
            fig, axes = plt.subplots(4,3,figsize=(12,8))
            axes = axes.ravel()
            t = np.arange(samples.shape[-1]) / 500.0
            for lead in range(12):
                axes[lead].plot(t, samples[lead], linewidth=0.6)
                axes[lead].set_title(f"Lead {lead+1}")
            plt.suptitle(f"Sample idx {si} | sample_f1={sample_f1[si]:.3f}")
            plt.tight_layout()
            outpng = os.path.join(outdir, f"example_{k}_ecg.png")
            plt.savefig(outpng, dpi=300, bbox_inches='tight'); plt.close()
            # write a small table GT vs top-10 preds
            topk = np.argsort(probs)[-10:][::-1]
            lines = []
            for c in topk:
                lines.append(f"class {c} | prob={probs[c]:.3f} | bin={int(bins_s[c])} | gt={int(y_true[si,c])}")
            with open(os.path.join(outdir, f"example_{k}_meta.txt"), "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            print("example failed", e)
    print("Saved qualitative examples.")


def _find_latest_checkpoint(exp_dir):
    # Acepta archivo o carpeta; prioriza ckpt_best.pt
    if os.path.isfile(exp_dir):
        return exp_dir if exp_dir.lower().endswith('.pt') else None
    best, anyc = [], []
    for dirpath, _, files in os.walk(exp_dir):
        for fn in files:
            if fn.lower().endswith('.pt'):
                fp = os.path.join(dirpath, fn)
                try:
                    mt = os.path.getmtime(fp)
                except Exception:
                    mt = 0.0
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


def compute_preds_from_checkpoint(exp_dir, hierarchy_path, test_files_path=None, batch_size=64, device=None):
    if HMST is None or ECG12Large is None:
        raise RuntimeError("No se pudieron importar HMST/ECG12Large; ejecuta desde la raíz del proyecto.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Cargar jerarquía
    if not hierarchy_path or not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"No existe jerarquía en {hierarchy_path}")
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hier = json.load(f)
    num_fine = len(hier['fine_codes'])
    num_coarse = len(hier['coarse_groups'])

    # Resolver archivos de test
    root = os.path.join('datos', '12Large', 'WFDBRecords')
    if test_files_path and os.path.exists(test_files_path):
        files = np.load(test_files_path, allow_pickle=True)
        files = list(map(str, files.tolist()))
    else:
        # Split por paciente 70/15/15 y tomar test
        from glob import glob
        hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
        pairs = [(h, os.path.splitext(h)[0] + '.mat') for h in hea_files]
        files_all = [h for h, m in pairs if os.path.exists(m)]
        pid_to_files = {}
        for h in files_all:
            pid = extract_patient_id(h, root) if extract_patient_id else os.path.basename(h).split('_')[0]
            pid_to_files.setdefault(pid, []).append(h)
        rng = np.random.default_rng(42)
        pids = list(pid_to_files.keys())
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(0.7 * n); n_va = int(0.15 * n)
        te_pids = set(pids[n_tr + n_va:])
        files = []
        for pid, flist in pid_to_files.items():
            if pid in te_pids:
                files.extend(flist)

    # Dataset/DataLoader
    te_ds = ECG12Large(root, sequence_len=5000, files=files, multilabel=True,
                       hierarchy_path=hierarchy_path, cache_dir=os.path.join('datos','pt_cache'),
                       random_crop=False, target_fs=500.0, bandpass_hz=(0.5,45.0), notch_hz=None, eval_mode=True)
    te_dl = DataLoader(te_ds, batch_size=int(batch_size), shuffle=False, num_workers=0,
                       pin_memory=(device == 'cuda'))

    # Cargar checkpoint
    ckpt = _find_latest_checkpoint(exp_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No se encontró checkpoint en {exp_dir}")
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    state_dict = state.get('model', state.get('state_dict', state))
    inferred = _infer_hmst_config_from_state(state_dict)
    model = HMST(
        input_channels=inferred.get('input_channels', 12 + 3),
        d_model=inferred.get('d_model', 256),
        num_stages=inferred.get('num_stages', 3),
        num_coarse=inferred.get('num_coarse', num_coarse),
        num_fine=inferred.get('num_fine', num_fine),
        snomed_dim=inferred.get('snomed_dim', inferred.get('num_coarse', num_coarse)),
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    y_true_list = []
    y_prob_list = []
    with torch.no_grad():
        for batch in te_dl:
            x = batch['samples'].to(device)
            y_fine = batch['labels_fine'].to(device)
            y_coarse = batch['labels_coarse'].to(device)
            b = x.size(0)
            wide_feats = torch.zeros(b, 3, device=device, dtype=x.dtype)
            snomed_embed = y_coarse
            _, logits_fine, _ = model(x, wide_feats, snomed_embed)
            p_fine = torch.sigmoid(logits_fine)
            y_true_list.append(y_fine.cpu().numpy())
            y_prob_list.append(p_fine.cpu().numpy())
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    return y_true, y_prob

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default=None, help=".npz with y_true,y_prob; si no se provee se busca en <exp_dir>/test_predictions.npz")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--hierarchy", default=os.path.join('datos','12Large','labels_hierarchy.json'))
    p.add_argument("--test_files", default=None)
    p.add_argument("--n_examples", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64, help="Batch size para cómputo en vivo si faltan preds")
    args = p.parse_args()

    base_out = os.path.join("results_viz", os.path.basename(args.exp_dir.rstrip("/")))
    os.makedirs(base_out, exist_ok=True)

    # Resolver preds: usar argumento o fallback a <exp_dir>/test_predictions.npz
    preds_path = args.preds
    if preds_path is None:
        cand = os.path.join(args.exp_dir, 'test_predictions.npz')
        if os.path.exists(cand):
            preds_path = cand
    if preds_path is not None and os.path.exists(preds_path):
        y_true, y_prob = load_preds(preds_path)
    else:
        # Cómputo en vivo desde checkpoint
        print("No se encontraron predicciones .npz; calculando desde checkpoint...")
        y_true, y_prob = compute_preds_from_checkpoint(
            args.exp_dir, args.hierarchy, test_files_path=args.test_files, batch_size=args.batch_size,
            device=("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Guardar copia para reutilizar
        np.savez(os.path.join(base_out, 'computed_test_predictions.npz'), y_true=y_true, y_prob=y_prob)
    # ensure 2D
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    outdir = ensure_outdir(base_out, "figures")
    # 1. ROC/PR micro + sample per-class
    plot_roc_pr_macro_micro(y_true, y_prob, outdir)

    # 2. per-class table (baseline 0.5) con nombres de clases si hay jerarquía
    class_names = None
    try:
        if args.hierarchy and os.path.exists(args.hierarchy):
            with open(args.hierarchy, 'r', encoding='utf-8') as f:
                hier = json.load(f)
            # Se espera que test_predictions sean del cabezal fine
            class_names = hier.get('fine_codes', None)
    except Exception:
        class_names = None
    df = per_class_metrics_table(y_true, y_prob, thresholds=None, outdir=outdir, class_names=class_names)

    # 3. optimized thresholds: search per-class (same grid you used)
    n_classes = y_true.shape[1]
    best_thr = np.full((n_classes,), 0.5, dtype=float)
    from sklearn.metrics import f1_score as _f1
    for i in range(n_classes):
        yi = y_true[:, i]
        pi = y_prob[:, i]
        if yi.sum() == 0:
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f1i = _f1(yi, (pi >= t).astype(int), zero_division=0)
            if f1i > best_f1:
                best_f1 = f1i; best_t = t
        best_thr[i] = best_t
    # save thresholds
    with open(os.path.join(base_out, "thresholds_opt.json"), "w") as f:
        json.dump(best_thr.tolist(), f, indent=2)
    # 4. reliability
    reliability_plot(y_true, y_prob, outdir)
    # 5. prob histograms
    prob_histograms(y_prob, outdir, top_k=8)
    # 6. compare F1 improvements
    compare_f1_thresholds(y_true, y_prob, best_thr, outdir)
    # 7. confusion counts (with opt thresholds)
    y_bin_opt = (y_prob >= best_thr[None,:]).astype(int)
    confusion_counts(y_true, y_bin_opt, outdir)
    # 8. examples qualitative
    # dataset constructor: import the dataset used en entrenamiento
    try:
        from datasets.ecg12large import ECG12Large
        def ds_ctor(file_list):
            return ECG12Large(root=os.path.join('datos','12Large','WFDBRecords'),
                              sequence_len=5000, files=file_list, multilabel=True,
                              hierarchy_path=args.hierarchy, cache_dir=os.path.join('datos','pt_cache'),
                              random_crop=False, target_fs=500.0, bandpass_hz=(0.5,45.0), eval_mode=True)
        examples_qualitative(args.exp_dir, args.test_files, y_true, y_prob, best_thr, args.n_examples, outdir, ds_ctor)
    except Exception as e:
        print("Could not create dataset for qualitative examples:", e)

    # zip results
    zipf = os.path.join(base_out, "figures_bundle.zip")
    with ZipFile(zipf, 'w') as z:
        for root, _, files in os.walk(outdir):
            for fn in files:
                z.write(os.path.join(root, fn), arcname=os.path.join(os.path.relpath(root,outdir), fn))
    # manifest
    manifest = {
        "outdir": outdir,
        "zip": zipf,
        "notes": "Contains ROC/PR, reliability, per-class metrics, histograms, qualitative examples"
    }
    with open(os.path.join(base_out, "figures_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("All visualizations saved to:", outdir)

if __name__ == "__main__":
    main()
