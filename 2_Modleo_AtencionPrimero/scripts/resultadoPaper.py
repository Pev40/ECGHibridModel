import os
import sys
import time
# Asegura que el directorio raíz del proyecto esté en sys.path para importar paquetes hermanos
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST
from datasets.ecg12large import ECG12Large, extract_patient_id


def _log(msg):
    print(f"[resultadoPaper] {msg}", flush=True)
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
    
    start_eval = time.time()
    with torch.no_grad():
        try:
            total_batches = len(dataloader)
        except Exception:
            total_batches = None
        t0 = time.time()
        for bi, batch in enumerate(dataloader, start=1):
            x = batch['samples'].to(device)  # [B,12,T]
            # Campos opcionales: usar None -> el modelo construye zeros internos
            wide = batch.get('wide_feats', None)
            if wide is not None:
                wide = wide.to(device)
            snomed = batch.get('snomed_embed', None)
            if snomed is not None:
                snomed = snomed.to(device)
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
            # Log de progreso
            if total_batches is not None and (bi % max(1, total_batches // 20) == 0 or bi == total_batches):
                _log(f"Eval batch {bi}/{total_batches} (tiempo acumulado {time.time()-t0:.1f}s)")
    
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
    _log(f"Evaluación finalizada en {time.time()-start_eval:.1f}s. Pérdida promedio: {avg_loss:.4f}")
    
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
    """Acepta carpeta o archivo. Si es archivo .pt, lo usa; si es carpeta, busca el más reciente priorizando 'ckpt_best.pt'."""
    # Si es ruta a archivo, úsalo directamente
    if os.path.isfile(exp_root):
        return exp_root if exp_root.lower().endswith('.pt') else None

    best_candidates = []
    any_candidates = []
    for dirpath, _, filenames in os.walk(exp_root):
        for fn in filenames:
            if fn.lower().endswith('.pt'):
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


def _infer_hmst_config_from_state(state_dict):
    """Infiera parámetros de HMST a partir del state_dict del checkpoint."""
    cfg = {}
    
    def _get_tensor(keys):
        for k in keys:
            v = state_dict.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        return None
    # d_model e input_channels desde input_proj
    w = _get_tensor(['input_proj.weight'])
    if isinstance(w, torch.Tensor):
        cfg['d_model'] = int(w.shape[0])
        cfg['input_channels'] = int(w.shape[1])
    # Alternativa: cls_token
    ct = _get_tensor(['cls_token'])
    if isinstance(ct, torch.Tensor):
        cfg['d_model'] = int(ct.shape[-1])
    # num_stages: cuenta etapas conv
    stage_idxs = []
    for k in state_dict.keys():
        if k.startswith('conv_stages.'):
            try:
                idx = int(k.split('.')[1])
                stage_idxs.append(idx)
            except Exception:
                pass
    if stage_idxs:
        cfg['num_stages'] = max(stage_idxs) + 1
    # num_coarse/num_fine desde cabezales
    hc = _get_tensor(['head_coarse.1.weight', 'head_coarse.weight'])
    if isinstance(hc, torch.Tensor):
        cfg['num_coarse'] = int(hc.shape[0])
    hf = _get_tensor(['head_fine.1.weight', 'head_fine.weight'])
    if isinstance(hf, torch.Tensor):
        cfg['num_fine'] = int(hf.shape[0])
    # snomed_dim
    sp = _get_tensor(['snomed_proj.weight'])
    if isinstance(sp, torch.Tensor):
        cfg['snomed_dim'] = int(sp.shape[1])
    return cfg


if __name__ == '__main__':
    # Ruta base de experiments_logs relativa a este archivo
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ))
    exp_root = os.path.join(proj_root, 'experiments_logs',"multi_run_v3_20250926_003016","seed_789","ckpt_best.pt")
    print(exp_root)
    ckpt_path = _find_latest_checkpoint(exp_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if ckpt_path is None:
        print('No se encontró ningún checkpoint en', exp_root)
    else:
        print('Usando checkpoint:', ckpt_path)
        _log('Cargando checkpoint...')
        # Carga segura del checkpoint (evita objetos pickle no confiables si es posible)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        _log('Detectando formato del checkpoint...')
        # Detecta layout del checkpoint: {'model': ...} o {'state_dict': ...} o dict plano
        if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            state_dict = state['model']
        elif isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state_dict = state['state_dict']
        else:
            state_dict = state
        _log('Infiriendo configuración del modelo a partir del checkpoint...')
        # Inferir config para instanciar HMST compatible con el checkpoint
        inferred = _infer_hmst_config_from_state(state_dict)
        _log(f"Config inferida: {inferred}")
        _log('Instanciando HMST con la configuración inferida...')
        model = HMST(
            input_channels=inferred.get('input_channels', 12 + 3),
            d_model=inferred.get('d_model', 512),
            num_stages=inferred.get('num_stages', 3),
            num_coarse=inferred.get('num_coarse', 10),
            num_fine=inferred.get('num_fine', 27),
            snomed_dim=inferred.get('snomed_dim', 10),
        ).to(device)
        _log('Cargando pesos del checkpoint en el modelo...')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _log(f"Advertencia: pesos faltantes: {len(missing)} elementos")
        if unexpected:
            _log(f"Advertencia: pesos inesperados: {len(unexpected)} elementos")

        # Construir test_dl replicando la lógica de train_full_v3 (12Large por defecto)
        try:
            _log('Cargando jerarquía de etiquetas...')
            # Cargar jerarquía para num_fine/num_coarse y embeddings
            hierarchy_path = os.path.join('datos', '12Large', 'labels_hierarchy.json')
            if not os.path.exists(hierarchy_path):
                raise FileNotFoundError(f'Falta {hierarchy_path}. Genera la jerarquía antes de evaluar.')
            import json as _json
            with open(hierarchy_path, 'r', encoding='utf-8') as _f:
                _hier = _json.load(_f)
            fine_codes = _hier['fine_codes']
            coarse_groups = _hier['coarse_groups']
            num_fine = len(fine_codes)
            num_coarse = len(coarse_groups)
            _log(f"Jerarquía: num_fine={num_fine}, num_coarse={num_coarse}")

            _log('Enumerando archivos HEA y construyendo split de test por paciente...')
            # Construir lista de archivos de test por paciente
            from glob import glob as _glob
            root = os.path.join('datos', '12Large', 'WFDBRecords')
            hea_files = _glob(os.path.join(root, '**', '*.hea'), recursive=True)
            pairs = [(hea, os.path.splitext(hea)[0] + '.mat') for hea in hea_files]
            files = [hea for hea, mat in pairs if os.path.exists(mat)]
            _log(f"HEA encontrados: {len(hea_files)} | con .mat: {len(files)}")
            pid_to_files = {}
            for hea in files:
                pid = extract_patient_id(hea, root)
                pid_to_files.setdefault(pid, []).append(hea)
            import numpy as _np
            rng = _np.random.default_rng(42)
            pids = list(pid_to_files.keys())
            rng.shuffle(pids)
            n = len(pids)
            n_tr = int(0.7 * n)
            n_va = int(0.15 * n)
            te_pids = set(pids[n_tr+n_va:])
            te_files = []
            for pid, flist in pid_to_files.items():
                if pid in te_pids:
                    te_files.extend(flist)
            _log(f"Pacientes totales: {n} | test: {len(te_pids)} | archivos test: {len(te_files)}")

            _log('Construyendo dataset de test (esto puede tardar)...')
            t_ds = time.time()
            # Dataset de test
            te_ds = ECG12Large(root, sequence_len=5000, files=te_files, multilabel=True,
                               hierarchy_path=hierarchy_path, cache_dir=os.path.join('datos', 'pt_cache'),
                               random_crop=False, target_fs=500.0, bandpass_hz=(0.5,45.0), notch_hz=None, eval_mode=True)
            _log(f"Dataset test listo en {time.time()-t_ds:.1f}s | muestras: {len(te_ds)}")
            from torch.utils.data import DataLoader as _DataLoader
            
            def _build_dl(bs, workers, pin):
                return _DataLoader(
                    te_ds,
                    batch_size=int(bs),
                    shuffle=False,
                    num_workers=int(workers),
                    pin_memory=bool(pin),
                    persistent_workers=(int(workers) > 0),
                    prefetch_factor=(2 if int(workers) > 0 else None),
                )

            # Intentos con distintos batch sizes y dispositivo
            tried = []
            success = False
            metrics = None
            devices_try = [device] + (['cpu'] if device == 'cuda' else [])
            for dev_try in devices_try:
                if dev_try == 'cuda' and not torch.cuda.is_available():
                    continue
                # Mover modelo al dispositivo de intento
                if next(model.parameters()).device.type != dev_try:
                    _log(f"Moviendo modelo a {dev_try}...")
                    model = model.to(dev_try)
                bs_candidates = [64, 32, 16, 8, 4] if dev_try == 'cuda' else [8, 4, 2, 1]
                for bs in bs_candidates:
                    try:
                        pin = (dev_try == 'cuda')
                        workers = 0  # más estable en Windows para inferencia
                        test_dl = _build_dl(bs, workers, pin)
                        _log(f"DataLoader test creado | device={dev_try} | bs={bs} | batches={len(test_dl)} | workers={workers}")
                        _log('Iniciando evaluación...')
                        t_eval = time.time()
                        metrics = evaluate_hierarchical(
                            model,
                            test_dl,
                            num_coarse=num_coarse,
                            num_fine=num_fine,
                            device=dev_try,
                        )
                        _log(f"Evaluación completada en {time.time()-t_eval:.1f}s en device={dev_try} bs={bs}")
                        success = True
                        break
                    except RuntimeError as re:
                        tried.append((dev_try, bs, str(re).split('\n')[0]))
                        _log(f"Fallo en device={dev_try} bs={bs}: {re}")
                        if dev_try == 'cuda':
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        continue
                    except Exception as e2:
                        tried.append((dev_try, bs, str(e2).split('\n')[0]))
                        _log(f"Error en device={dev_try} bs={bs}: {e2}")
                        continue
                if success:
                    break
            if not success:
                _log(f"No se pudo evaluar tras intentos: {tried}")
            else:
                print('Métricas Coarse:', metrics['coarse'])
                print('Métricas Fine:', metrics['fine'])
        except Exception as e:
            _log(f"No se pudo construir test_dl automáticamente: {e}")
            _log('Define test_dl manualmente o ajusta las rutas de datos antes de evaluar.')