import os
import json
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, multilabel_confusion_matrix


def find_run_dirs(root_dir):
    # Busca subcarpetas de datasets que contengan train_log.csv
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'train_log.csv' in filenames:
            candidates.append(dirpath)
    # Filtra típicas rutas Resultado/.../<dataset>
    # Mantener solo carpetas hoja (con csv y sin subcarpetas con csv)
    leaf_candidates = []
    set_cands = set(candidates)
    for d in candidates:
        if not any((c != d and c.startswith(d + os.sep)) for c in set_cands):
            leaf_candidates.append(d)
    return sorted(leaf_candidates)


def load_test_arrays(run_dir):
    npz_path = os.path.join(run_dir, 'test_predictions.npz')
    if not os.path.exists(npz_path):
        return None, None, None
    data = np.load(npz_path, allow_pickle=True)
    y_true = data.get('y_true')
    y_prob = data.get('y_prob')
    thresholds = data.get('thresholds')
    return y_true, y_prob, thresholds


def load_test_metrics(run_dir):
    jpath = os.path.join(run_dir, 'test_metrics.json')
    if not os.path.exists(jpath):
        return {}
    with open(jpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(run_dir, save=True):
    csv_path = os.path.join(run_dir, 'train_log.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ['train_loss', 'val_loss', 'val_auroc_macro', 'val_auprc_macro', 'val_f1_macro', 'lr']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss')
    axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, ls='--', lw=0.5)

    if 'val_auroc_macro' in df.columns:
        axes[1].plot(df['epoch'], df['val_auroc_macro'], label='AUROC')
    if 'val_auprc_macro' in df.columns:
        axes[1].plot(df['epoch'], df['val_auprc_macro'], label='AUPRC')
    if 'val_f1_macro' in df.columns:
        axes[1].plot(df['epoch'], df['val_f1_macro'], label='F1')
    axes[1].set_ylabel('Val Metrics'); axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(True, ls='--', lw=0.5)

    if 'lr' in df.columns:
        axes[2].plot(df['epoch'], df['lr'], label='LR', color='purple')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')
    axes[2].legend(); axes[2].grid(True, ls='--', lw=0.5)
    fig.suptitle(os.path.basename(run_dir))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save:
        out_path = os.path.join(run_dir, 'training_curves.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    return fig


def plot_multilabel_roc_pr(y_true, y_prob, class_names=None, run_dir=None):
    if y_true is None or y_prob is None:
        return None, None
    num_classes = y_true.shape[1]
    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        yi = y_true[:, i]
        pi = y_prob[:, i]
        if yi.sum() == 0 or (1 - yi).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yi, pi)
        ax_roc.plot(fpr, tpr, label=(class_names[i] if class_names else f'C{i}'))
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR'); ax_roc.legend(fontsize=8, ncol=2)
    ax_roc.grid(True, ls='--', lw=0.5)

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        yi = y_true[:, i]
        pi = y_prob[:, i]
        if yi.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(yi, pi)
        ap = average_precision_score(yi, pi)
        label = (class_names[i] if class_names else f'C{i}') + f' (AP={ap:.2f})'
        ax_pr.plot(recall, precision, label=label)
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision'); ax_pr.legend(fontsize=8, ncol=2)
    ax_pr.grid(True, ls='--', lw=0.5)

    if run_dir is not None:
        roc_path = os.path.join(run_dir, 'roc_curves.png')
        pr_path = os.path.join(run_dir, 'pr_curves.png')
        fig_roc.savefig(roc_path, dpi=150)
        fig_pr.savefig(pr_path, dpi=150)
        plt.close(fig_roc)
        plt.close(fig_pr)
        return roc_path, pr_path
    return fig_roc, fig_pr


def plot_multilabel_confusion(y_true, y_prob, thresholds=None, class_names=None, run_dir=None):
    if y_true is None or y_prob is None:
        return None
    if thresholds is None:
        thr = 0.5
        y_pred = (y_prob >= thr).astype(np.int32)
    else:
        y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)

    cms = multilabel_confusion_matrix(y_true, y_pred)
    num_classes = cms.shape[0]
    cols = min(4, num_classes)
    rows = int(np.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    for i in range(num_classes):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        tn, fp, fn, tp = cms[i].ravel()
        mat = np.array([[tn, fp],[fn, tp]], dtype=int)
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Real 0', 'Real 1'], ax=ax)
        ax.set_title(class_names[i] if class_names else f'C{i}')
    # Oculta ejes sobrantes
    for j in range(num_classes, rows*cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r, c])
    plt.tight_layout()
    if run_dir is not None:
        out_path = os.path.join(run_dir, 'confusion_matrices.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    return fig


def summarize_across_runs(run_dirs):
    rows = []
    for d in run_dirs:
        metrics = load_test_metrics(d)
        name = os.path.basename(d)
        rows.append({'run': name,
                     'dir': d,
                     'auroc_macro': metrics.get('auroc_macro', np.nan),
                     'auprc_macro': metrics.get('auprc_macro', np.nan),
                     'f1_macro': metrics.get('f1_macro', np.nan),
                     'auroc_micro': metrics.get('auroc_micro', np.nan),
                     'auprc_micro': metrics.get('auprc_micro', np.nan)})
    df = pd.DataFrame(rows)
    return df


def plot_compare_bar(df, out_path=None):
    if df.empty:
        return None
    metrics = ['auroc_macro', 'auprc_macro', 'f1_macro']
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.set_index('run')[metrics]
    df_plot.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Comparación entre datasets')
    ax.grid(True, axis='y', ls='--', lw=0.5)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    return fig


def compute_micro_roc(y_true, y_prob):
    try:
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    except Exception:
        return None, None, None


def plot_combined_micro_roc(run_dirs, out_path):
    print('Plotting combined micro ROC; run_dirs:', out_path)
    plt.figure(figsize=(8, 6))
    any_plotted = False
    for d in run_dirs:
        name = os.path.basename(d)
        y_true, y_prob, _ = load_test_arrays(d)
        if y_true is None or y_prob is None:
            continue
        fpr, tpr, roc_auc = compute_micro_roc(y_true, y_prob)
        if fpr is None:
            continue
        any_plotted = True
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC micro por dataset')
    plt.legend()
    plt.grid(True, ls='--', lw=0.5)
    if any_plotted:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_evolution_val_auroc(run_dirs, out_path):
    print('Plotting evolution of AUROC; run_dirs:', out_path)
    plt.figure(figsize=(10, 6))
    any_plotted = False
    for d in run_dirs:
        name = os.path.basename(d)
        csv_path = os.path.join(d, 'train_log.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'epoch' in df.columns and 'val_auroc_macro' in df.columns:
            try:
                x = pd.to_numeric(df['epoch'], errors='coerce')
                y = pd.to_numeric(df['val_auroc_macro'], errors='coerce')
                plt.plot(x, y, marker='o', ms=3, label=name)
                any_plotted = True
            except Exception:
                pass
    plt.xlabel('Epoch')
    plt.ylabel('Val AUROC (macro)')
    plt.title('Evolución de AUROC de validación por dataset')
    plt.ylim(0, 1.05)
    plt.grid(True, ls='--', lw=0.5)
    if any_plotted:
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_evolution_val_auprc(run_dirs, out_path):
    print('Plotting evolution of AUPRC; run_dirs:', out_path)
    plt.figure(figsize=(10, 6))
    any_plotted = False
    for d in run_dirs:
        name = os.path.basename(d)
        csv_path = os.path.join(d, 'train_log.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'epoch' in df.columns and 'val_auprc_macro' in df.columns:
            try:
                x = pd.to_numeric(df['epoch'], errors='coerce')
                y = pd.to_numeric(df['val_auprc_macro'], errors='coerce')
                plt.plot(x, y, marker='o', ms=3, label=name)
                any_plotted = True
            except Exception:
                pass
    plt.xlabel('Epoch')
    plt.ylabel('Val AUPRC (macro)')
    plt.title('Evolución de AUPRC de validación por dataset')
    plt.ylim(0, 1.05)
    plt.grid(True, ls='--', lw=0.5)
    if any_plotted:
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def _slope(x, y):
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) >= 2 and len(y) >= 2:
            m, _ = np.polyfit(x, y, 1)
            return float(m)
    except Exception:
        pass
    return float('nan')


def compute_overfit_indicators(run_dir, tail_epochs=5):
    import pandas as pd
    csv_path = os.path.join(run_dir, 'train_log.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ['epoch', 'train_loss', 'val_loss', 'val_auroc_macro']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # métricas básicas
    train_end = float(df['train_loss'].iloc[-1]) if 'train_loss' in df.columns else float('nan')
    val_end = float(df['val_loss'].iloc[-1]) if 'val_loss' in df.columns else float('nan')
    gap_end = val_end - train_end if (np.isfinite(val_end) and np.isfinite(train_end)) else float('nan')
    # mejor validación
    if 'val_loss' in df.columns:
        idx_best = int(df['val_loss'].idxmin())
        best_epoch = int(df['epoch'].iloc[idx_best]) if 'epoch' in df.columns else (idx_best + 1)
        best_val = float(df['val_loss'].iloc[idx_best])
    else:
        idx_best = len(df) - 1
        best_epoch = int(df['epoch'].iloc[-1]) if 'epoch' in df.columns else len(df)
        best_val = float('nan')
    # pendientes finales
    tail = df.tail(int(min(tail_epochs, len(df))))
    tr_slope = _slope(tail['epoch'], tail['train_loss']) if 'train_loss' in df.columns else float('nan')
    va_slope = _slope(tail['epoch'], tail['val_loss']) if 'val_loss' in df.columns else float('nan')
    # heurística de overfitting: val sube mientras train baja o gap grande
    overfit_flag = False
    if np.isfinite(tr_slope) and np.isfinite(va_slope):
        if tr_slope < 0 and va_slope > 0:
            overfit_flag = True
    if np.isfinite(gap_end) and gap_end > 0.3:  # umbral heurístico; ajustable por dataset
        overfit_flag = True
    return {
        'run': os.path.basename(run_dir),
        'dir': run_dir,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'train_end': train_end,
        'val_end': val_end,
        'gap_end': gap_end,
        'train_tail_slope': tr_slope,
        'val_tail_slope': va_slope,
        'overfit_flag': bool(overfit_flag),
    }


def compute_micro_pr(y_true, y_prob):
    try:
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
        ap = average_precision_score(y_true, y_prob, average='micro')
        return precision, recall, ap
    except Exception:
        return None, None, None


def plot_combined_micro_pr(run_dirs, out_path):
    plt.figure(figsize=(8, 6))
    any_plotted = False
    for d in run_dirs:
        name = os.path.basename(d)
        y_true, y_prob, _ = load_test_arrays(d)
        if y_true is None or y_prob is None:
            continue
        precision, recall, ap = compute_micro_pr(y_true, y_prob)
        if precision is None:
            continue
        any_plotted = True
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva PR micro por dataset')
    plt.legend()
    plt.grid(True, ls='--', lw=0.5)
    if any_plotted:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluación y gráficas a partir de resultados guardados')
    parser.add_argument('--root', type=str, default=os.path.join('Resultado',"DespuesTuning","all_datasets_20250923_154225"), help='Carpeta raíz con subcarpetas por dataset')
    parser.add_argument('--save', action='store_true', help='Guardar figuras en disco')
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.root)
    if len(run_dirs) == 0:
        print('No se encontraron carpetas con train_log.csv en', args.root)
        return

    # Por dataset
    for d in run_dirs:
        print('Procesando', d)
        if args.save:
            plot_training_curves(d, save=True)
        y_true, y_prob, thresholds = load_test_arrays(d)
        if y_true is not None and y_prob is not None:
            if args.save:
                plot_multilabel_roc_pr(y_true, y_prob, class_names=None, run_dir=d)
                plot_multilabel_confusion(y_true, y_prob, thresholds=thresholds, class_names=None, run_dir=d)

    # Comparación entre datasets
    df = summarize_across_runs(run_dirs)
    print('\nResumen de métricas:')
    print(df[['run','auroc_macro','auprc_macro','f1_macro']])
    if args.save:
        out_bar = os.path.join(args.root, 'comparacion_datasets.png')
        plot_compare_bar(df, out_bar)
        df.to_csv(os.path.join(args.root, 'resumen_metricas.csv'), index=False)
        # Curva ROC micro combinada por dataset
        plot_combined_micro_roc(run_dirs, os.path.join(args.root, 'combined_roc_micro.png'))
        # Evolución de AUROC de validación por dataset
        plot_evolution_val_auroc(run_dirs, os.path.join(args.root, 'val_auroc_evolution.png'))
        # Curva PR micro combinada por dataset
        plot_combined_micro_pr(run_dirs, os.path.join(args.root, 'combined_pr_micro.png'))
        # Evolución de AUPRC de validación por dataset
        plot_evolution_val_auprc(run_dirs, os.path.join(args.root, 'val_auprc_evolution.png'))

    # Diagnóstico de overfitting por run
    rows = []
    for d in run_dirs:
        r = compute_overfit_indicators(d)
        if r is not None:
            rows.append(r)
    if rows:
        import pandas as pd
        df_over = pd.DataFrame(rows)
        print('\nIndicadores de overfitting:')
        print(df_over[['run','best_epoch','train_end','val_end','gap_end','train_tail_slope','val_tail_slope','overfit_flag']])
        if args.save:
            df_over.to_csv(os.path.join(args.root, 'overfit_summary.csv'), index=False)


if __name__ == '__main__':
    main()


