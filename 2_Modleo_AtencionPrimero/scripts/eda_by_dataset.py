import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


def summarize(arr):
    if len(arr) == 0:
        return {}
    a = np.asarray(arr, dtype=float)
    return {
        'count': int(a.size),
        'min': float(np.min(a)),
        'p25': float(np.percentile(a, 25)),
        'median': float(np.median(a)),
        'p75': float(np.percentile(a, 75)),
        'max': float(np.max(a)),
        'mean': float(np.mean(a))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['12large','ptbxl','georgia','incart'])
    ap.add_argument('--outdir', type=str, default=os.path.join('eda_outputs'))
    ap.add_argument('--max_files', type=int, default=500)
    args = ap.parse_args()

    if args.dataset == '12large':
        root = os.path.join('datos', 'WFDBRecords')
        hea_glob = os.path.join(root, '**', '*.hea')
    elif args.dataset == 'georgia':
        root = os.path.join('datos', 'Georgia12LeadECGDatabase')
        hea_glob = os.path.join(root, '*.hea')
    elif args.dataset == 'ptbxl':
        root = os.path.join('datos', 'PTBXL')
        hea_glob = os.path.join(root, 'records500', '**', '*_hr.hea')
    else:
        root = os.path.join('datos', 'StPetersburgIncart12LeadArrhythmiaDatabase', 'files')
        hea_glob = os.path.join(root, '*.hea')

    os.makedirs(args.outdir, exist_ok=True)
    out_prefix = os.path.join(args.outdir, args.dataset)

    hea_files = glob(hea_glob, recursive=True)[:args.max_files]
    fs_list, n_sig_list, siglen_list = [], [], []
    import re
    for hea in hea_files:
        try:
            with open(hea, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [ln.strip() for ln in f.readlines()]
            first = lines[0].split()
            if len(first) >= 4:
                try:
                    n_sig_list.append(int(first[1]))
                except Exception:
                    pass
                try:
                    fs_list.append(float(first[2]))
                except Exception:
                    # buscar patrón "Hz"
                    m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", lines[0])
                    if m:
                        fs_list.append(float(m.group(1)))
                try:
                    siglen_list.append(int(first[3]))
                except Exception:
                    pass
        except Exception:
            continue

    summary = {
        'fs': summarize(fs_list),
        'n_signals_header': summarize(n_sig_list),
        'signal_length_samples': summarize(siglen_list),
        'n_files_scanned': len(hea_files)
    }
    with open(out_prefix + '_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # simple hist plots
    try:
        if fs_list:
            plt.figure()
            plt.hist(fs_list, bins=30)
            plt.title(f'fs {args.dataset}')
            plt.savefig(out_prefix + '_fs_hist.png', dpi=120)
            plt.close()
        if n_sig_list:
            plt.figure()
            plt.hist(n_sig_list, bins=24)
            plt.title(f'n_signals {args.dataset}')
            plt.savefig(out_prefix + '_n_signals_hist.png', dpi=120)
            plt.close()
        if siglen_list:
            plt.figure()
            plt.hist(siglen_list, bins=30)
            plt.title(f'signal length {args.dataset}')
            plt.savefig(out_prefix + '_siglen_hist.png', dpi=120)
            plt.close()
    except Exception:
        pass

    print('EDA escrito en', args.outdir)


if __name__ == '__main__':
    main()


