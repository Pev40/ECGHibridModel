import os
from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import wandb  # Opcional para logging; instala si no tienes

def evaluate_hierarchical(
    model,
    dataloader,
    num_coarse=10,
    num_fine=27,
    device='cuda',
    use_attn=False,
    threshold=0.5,
    thresholds_coarse=None,
    thresholds_fine=None,
):
    """
    Evalúa el modelo HMST con métricas desglosadas por cabezal (coarse/fine),
    globales (macro/micro), y guarda resultados en dict para logging.
    
    Args:
        model: Instancia de HMST en eval mode.
        dataloader: DataLoader de test/val (batches con 'samples', 'wide_feats', 'snomed_embed', 'labels_coarse', 'labels_fine').
        num_coarse: Número de clases coarse.
        num_fine: Número de clases fine.
        device: 'cuda' o 'cpu'.
        use_attn: Si True, retorna weights para viz.
    
    Returns:
        dict con métricas: {'coarse': {...}, 'fine': {...}, 'global_macro': {...}, 'global_micro': {...}}.
    """
    model.eval()
    all_coarse_gt, all_fine_gt = [], []
    all_coarse_pred, all_fine_pred = [], []
    all_attn_weights = []  # Para viz si use_attn
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['samples'].to(device)  # [B,12,T]
            wide = batch['wide_feats'].to(device)  # [B,3]
            snomed = batch['snomed_embed'].to(device)  # [B,10]
            coarse_gt = batch['labels_coarse'].to(device)  # [B,num_coarse]
            fine_gt = batch['labels_fine'].to(device)  # [B,num_fine]
            
            # Forward: logits
            coarse_logits, fine_logits, attn_w = model(x, wide, snomed, use_attn=use_attn)

            # Probabilidades para métricas y consistency
            coarse_pred = torch.sigmoid(coarse_logits)
            fine_pred = torch.sigmoid(fine_logits)

            # Loss (si necesitas): BCE con logits + consistencia en probas
            bce_c = F.binary_cross_entropy_with_logits(coarse_logits, coarse_gt)
            bce_f = F.binary_cross_entropy_with_logits(fine_logits, fine_gt)
            cons = model.consistency_loss(coarse_pred, fine_pred)
            batch_loss = bce_c + bce_f + 0.1 * cons
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Acumula preds/GT
            all_coarse_gt.append(coarse_gt.cpu().numpy())
            all_fine_gt.append(fine_gt.cpu().numpy())
            all_coarse_pred.append(coarse_pred.cpu().numpy())
            all_fine_pred.append(fine_pred.cpu().numpy())
            if use_attn:
                all_attn_weights.append(attn_w)  # List of lists [stage, weights]
    
    # Stack
    coarse_gt = np.vstack(all_coarse_gt)
    fine_gt = np.vstack(all_fine_gt)
    coarse_pred = np.vstack(all_coarse_pred)
    fine_pred = np.vstack(all_fine_pred)
    
    # Preparar umbrales
    def _ensure_thresholds(thrs, default_thr, n):
        if thrs is None:
            return np.full((n,), float(default_thr), dtype=float)
        thrs = np.asarray(thrs, dtype=float).ravel()
        if thrs.size == 1:
            return np.full((n,), float(thrs[0]), dtype=float)
        assert thrs.size == n, f"Tamaño de umbrales esperado {n}, recibido {thrs.size}"
        return thrs

    thr_coarse = _ensure_thresholds(thresholds_coarse, threshold, num_coarse)
    thr_fine = _ensure_thresholds(thresholds_fine, threshold, num_fine)

    # Métricas por cabezal (macro/micro) con manejo robusto de clases vacías
    def _safe_metric(fn, *args, **kwargs):
        try:
            return float(fn(*args, **kwargs))
        except Exception:
            return float('nan')

    def compute_metrics(y_gt, y_pred, thr_vec):
        y_bin = (y_pred >= thr_vec[None, :]).astype(int)
        auroc_macro = _safe_metric(roc_auc_score, y_gt, y_pred, average='macro')
        auprc_macro = _safe_metric(average_precision_score, y_gt, y_pred, average='macro')
        f1_macro = _safe_metric(f1_score, y_gt, y_bin, average='macro', zero_division=0)
        auroc_micro = _safe_metric(roc_auc_score, y_gt, y_pred, average='micro')
        auprc_micro = _safe_metric(average_precision_score, y_gt, y_pred, average='micro')
        f1_micro = _safe_metric(f1_score, y_gt, y_bin, average='micro', zero_division=0)
        return {
            'auroc_macro': auroc_macro,
            'auprc_macro': auprc_macro,
            'f1_macro': f1_macro,
            'auroc_micro': auroc_micro,
            'auprc_micro': auprc_micro,
            'f1_micro': f1_micro,
        }
    
    metrics_coarse = compute_metrics(coarse_gt, coarse_pred, thr_coarse)
    metrics_fine = compute_metrics(fine_gt, fine_pred, thr_fine)
    
    # Global (concat coarse + fine)
    global_gt = np.hstack([coarse_gt, fine_gt])
    global_pred = np.hstack([coarse_pred, fine_pred])
    thr_global = np.concatenate([thr_coarse, thr_fine], axis=0)
    metrics_global_macro = compute_metrics(global_gt, global_pred, thr_global)  # Incluye f1_macro
    # Para claridad, extrae solo micro en bloque separado (equivale al anterior compute)
    y_bin_global = (global_pred >= thr_global[None, :]).astype(int)
    metrics_global_micro = {
        'auroc_micro': _safe_metric(roc_auc_score, global_gt, global_pred, average='micro'),
        'auprc_micro': _safe_metric(average_precision_score, global_gt, global_pred, average='micro'),
        'f1_micro': _safe_metric(f1_score, global_gt, y_bin_global, average='micro', zero_division=0),
    }
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    results = {
        'loss': avg_loss,
        'coarse': metrics_coarse,
        'fine': metrics_fine,
        'global_macro': metrics_global_macro,  # Incluye macro para global
        'global_micro': metrics_global_micro,
        'attn_weights': all_attn_weights if use_attn else None,
        'threshold': float(threshold),
        'thresholds_coarse': thr_coarse.tolist(),
        'thresholds_fine': thr_fine.tolist(),
    }
    
    # Logging opcional
    if wandb.run is not None:
        wandb.log(results)
    
    return results

def ablation_study(model_class, dataloader, device='cuda', ablations=['vtt', 'snomed', 'wide']):
    """
    Estudios de ablations: Entrena/eval variantes (e.g., w/o VTT: usa MHSA estándar).
    Simplificado: Modifica model_class en runtime (e.g., attn_stages=None para w/o VTT).
    
    Args:
        model_class: Clase HMST.
        dataloader: Train/val/test loaders.
        device: 'cuda'.
        ablations: Lista de componentes a ablatar (e.g., 'vtt' desactiva VariableAttn).
    
    Returns:
        dict {ablation_name: metrics_dict}.
    """
    # Se asume que model_class puede instanciarse sin argumentos o que el caller proporciona una clase parcial
    baseline_metrics = evaluate_hierarchical(model_class(), dataloader['test'], device=device)
    ablation_results = {'baseline': baseline_metrics}
    
    for abl in ablations:
        class AblatedModel(model_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if abl == 'vtt':
                    # Ablate VTT: Reemplaza atención variable por MHSA simple en cada etapa
                    self.attn_stages = nn.ModuleList([
                        nn.MultiheadAttention(self.d_model, 8, batch_first=True)
                        for _ in range(self.num_stages)
                    ])
                elif abl == 'snomed':
                    # Ablate SNOMED: snomed_proj = nn.Identity()
                    self.snomed_proj = nn.Identity()
                elif abl == 'wide':
                    # Ablate wide: Input proj sin wide (C=12)
                    self.input_proj = nn.Linear(12, self.d_model)
        
        abl_model = AblatedModel()
        abl_metrics = evaluate_hierarchical(abl_model, dataloader['test'], device=device)
        ablation_results[abl] = abl_metrics
        # Diferencia vs. baseline
        delta_f1 = abl_metrics['global_macro']['f1_macro'] - baseline_metrics['global_macro']['f1_macro']
        print(f"Ablation {abl}: Delta F1 macro = {delta_f1:.4f}")
    
    return ablation_results

def qualitative_analysis(
    model,
    dataloader,
    num_samples=5,
    device='cuda',
    save_dir='viz_results',
    fs=500,
    num_coarse=10,
    num_fine=27,
    coarse_class_names=None,
    fine_class_names=None,
    threshold=0.5,
    thresholds_coarse=None,
    thresholds_fine=None,
):
    """
    Análisis cualitativo: Para N samples, plot ECG signal, tabla GT vs. pred, y attn heatmap.
    
    Args:
        model: HMST en eval.
        dataloader: Test loader (1 batch size para samples individuales).
        num_samples: Número de ECGs a visualizar.
        device: 'cuda'.
        save_dir: Carpeta para guardar plots.
    
    Outputs: Plots PNG guardados (e.g., 'sample_0.png').
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Nombres de clases
    if coarse_class_names is None:
        coarse_class_names = [f'C{i}' for i in range(num_coarse)]
    if fine_class_names is None:
        fine_class_names = [f'F{i}' for i in range(num_fine)]

    # Umbrales
    def _ensure_thresholds(thrs, default_thr, n):
        if thrs is None:
            return np.full((n,), float(default_thr), dtype=float)
        thrs = np.asarray(thrs, dtype=float).ravel()
        if thrs.size == 1:
            return np.full((n,), float(thrs[0]), dtype=float)
        assert thrs.size == n, f"Tamaño de umbrales esperado {n}, recibido {thrs.size}"
        return thrs

    thr_coarse = _ensure_thresholds(thresholds_coarse, threshold, num_coarse)
    thr_fine = _ensure_thresholds(thresholds_fine, threshold, num_fine)
    
    for i in range(num_samples):
        batch = next(iter(dataloader))  # Asume batch_size=1
        x = batch['samples'].to(device)  # [1,12,T]
        wide = batch['wide_feats'].to(device)
        snomed = batch['snomed_embed'].to(device)
        coarse_gt = batch['labels_coarse'].cpu().numpy()[0]  # [num_coarse]
        fine_gt = batch['labels_fine'].cpu().numpy()[0]      # [num_fine]
        
        with torch.no_grad():
            coarse_pred, fine_pred, attn_weights = model(x, wide, snomed, use_attn=True)
            coarse_pred = coarse_pred.cpu().numpy()[0]
            fine_pred = fine_pred.cpu().numpy()[0]
        
        # 1. Plot ECG Signal
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        axes = axes.ravel()
        t = np.arange(x.shape[-1]) / float(fs)
        for lead in range(12):
            axes[lead].plot(t, x[0, lead].cpu().numpy(), color='blue', linewidth=0.5)
            axes[lead].set_title(f'Lead {lead+1}')
            axes[lead].grid(True, alpha=0.3)
        plt.suptitle(f'ECG Sample {i}: GT vs. Pred')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ecg_sample_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Tabla GT vs. Pred (coarse/fine)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Coarse tabla
        coarse_table = ax1.table(
            cellText=[[coarse_class_names[j], int(coarse_gt[j]), f'{coarse_pred[j]:.3f}', int(coarse_pred[j] >= thr_coarse[j])]
                      for j in range(num_coarse)],
                                 colLabels=['Clase', 'GT', 'Pred Prob', 'Pred Bin'], loc='center')
        ax1.axis('off')
        ax1.set_title('Coarse Head: GT vs. Pred')
        
        # Fine tabla (top 10 para fit)
        top_k = min(10, num_fine)
        fine_table_data = [[str(fine_class_names[j])[:12], int(fine_gt[j]), f'{fine_pred[j]:.3f}', int(fine_pred[j] >= thr_fine[j])]
                           for j in range(top_k)]
        fine_table = ax2.table(cellText=fine_table_data, colLabels=['Clase (Top 10)', 'GT', 'Pred Prob', 'Pred Bin'], loc='center')
        ax2.axis('off')
        ax2.set_title('Fine Head: GT vs. Pred (Top 10)')
        
        plt.savefig(f'{save_dir}/comparison_table_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Attn Heatmap (si disponible)
        if attn_weights:
            try:
                aw = attn_weights  # lista por etapa
                # Toma la primera etapa si existe
                first = aw[0]
                # Si viene como tupla/lista de tensores, intenta promediar última dim
                if isinstance(first, (list, tuple)):
                    first = first[0]
                attn_map = first
                if isinstance(attn_map, torch.Tensor):
                    attn_map = attn_map.mean(dim=-1).detach().cpu().numpy()
                attn_map = np.asarray(attn_map)
                sns.heatmap(attn_map, cmap='viridis', xticklabels=1,
                            yticklabels=range(0, attn_map.shape[0], max(1, attn_map.shape[0] // 10)))
                plt.title(f'Attn Heatmap Sample {i} (Stage 1)')
                plt.xlabel('Feature')
                plt.ylabel('Timesteps')
                plt.savefig(f'{save_dir}/attn_heatmap_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                pass
    
    print(f"Viz guardada en {save_dir}: {num_samples} samples con plots, tablas y heatmaps.")

def _find_latest_checkpoint(exp_root):
    """Busca el checkpoint más reciente en exp_root. Prioriza 'ckpt_best.pt'."""
    best_candidates = []
    any_candidates = []
    for dirpath, _, filenames in os.walk(exp_root):
        for fn in filenames:
            if fn.endswith('.pt'):
                fp = os.path.join(dirpath, fn)
                try:
                    mtime = os.path.getmtime(fp)
                except Exception:
                    mtime = 0.0
                any_candidates.append((mtime, fp))
                if fn == 'ckpt_best.pt':
                    best_candidates.append((mtime, fp))
    # Prioriza ckpt_best.pt
    if best_candidates:
        best_candidates.sort(key=lambda x: x[0], reverse=True)
        return best_candidates[0][1]
    if any_candidates:
        any_candidates.sort(key=lambda x: x[0], reverse=True)
        return any_candidates[0][1]
    return None


if __name__ == '__main__':
    # Ruta base de experiments_logs relativa a este archivo
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    exp_root = os.path.join(proj_root, 'experiments_logs')
    ckpt_path = _find_latest_checkpoint(exp_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if ckpt_path is None:
        print('No se encontró ningún checkpoint en', exp_root)
    else:
        print('Usando checkpoint:', ckpt_path)
        # Instancia modelo y carga state_dict
        model = HMST().to(device)
        state = torch.load(ckpt_path, map_location=device)
        # Soporta tanto dict plano como {'state_dict': ...}
        state_dict = state.get('state_dict', state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print('Advertencia: pesos faltantes:', missing)
        if unexpected:
            print('Advertencia: pesos inesperados:', unexpected)

        # Ejecuta evaluación sólo si test_dl está presente en el entorno
        if 'test_dl' in globals() and globals()['test_dl'] is not None:
            metrics = evaluate_hierarchical(model, globals()['test_dl'], num_coarse=model.head_coarse[-1].out_features,
                                            num_fine=model.head_fine[-1].out_features, device=device)
            print('Métricas Coarse:', metrics['coarse'])
            print('Métricas Fine:', metrics['fine'])
        else:
            print('test_dl no está definido. Proporciona un DataLoader de test para evaluar.')