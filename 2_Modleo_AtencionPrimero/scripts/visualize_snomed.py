import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt

# Asegura que el paquete raíz esté en sys.path al ejecutar desde scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization import (
    plot_snomed_from_loader_bar,
    plot_snomed_from_loader_heatmap,
    plot_snomed_from_loader_projection,
)
from train_full_v3 import build_loaders_12large
from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST


def main():
    parser = argparse.ArgumentParser(description='Visualización de SNOMED embed/coarse labels')
    parser.add_argument('--mode', type=str, default='bar', choices=['bar', 'heatmap', 'proj'])
    parser.add_argument('--sequence_len', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--target_fs', type=float, default=500.0)
    parser.add_argument('--bandpass_low', type=float, default=0.5)
    parser.add_argument('--bandpass_high', type=float, default=45.0)
    parser.add_argument('--notch_hz', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=5)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()

    hierarchy_path = os.path.join('datos', '12Large', 'labels_hierarchy.json')
    tr_dl, va_dl, te_dl = build_loaders_12large(
        args.sequence_len, hierarchy_path, args.batch_size, args.workers,
        cache_dir=None, target_fs=args.target_fs,
        bandpass_hz=(args.bandpass_low, args.bandpass_high),
        notch_hz=(args.notch_hz if args.notch_hz in (50, 60) else None),
        seed=42, smoke_test=False, smoke_n=256,
        aug_jitter_std=0.0, aug_shift_max=0, aug_lead_drop_prob=0.0,
        aug_amp_scale_min=1.0, aug_amp_scale_max=1.0,
        aug_lead_noise_scale_max=1.0, aug_time_warp_max=0.0, aug_time_warp_p=0.0,
    )

    # obtener nombres coarse desde el dataset de validación
    va_ds = va_dl.dataset
    if hasattr(va_ds, 'coarse_names'):
        coarse_names = list(va_ds.coarse_names)
    elif hasattr(va_ds, 'hierarchy') and 'coarse_groups' in va_ds.hierarchy:
        coarse_names = list(va_ds.hierarchy['coarse_groups'].keys())
    else:
        coarse_names = None

    if args.mode == 'bar':
        ax = plot_snomed_from_loader_bar(va_dl, coarse_names=coarse_names, idx=args.idx)
    elif args.mode == 'heatmap':
        ax = plot_snomed_from_loader_heatmap(va_dl, coarse_names=coarse_names, max_batches=args.max_batches)
    else:
        # para proyección, crear un modelo para usar snomed_proj (no carga pesos, es solo para dimensión)
        num_coarse = len(coarse_names) if coarse_names is not None else 10
        model = HMST(num_coarse=num_coarse, num_fine=27, snomed_dim=num_coarse)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        ax = plot_snomed_from_loader_projection(va_dl, model=model, device=device, max_batches=args.max_batches)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f'Guardado en {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()


