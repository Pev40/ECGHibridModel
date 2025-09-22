import os
import json
import ast
import argparse
from collections import Counter, defaultdict

import pandas as pd


def parse_scp_dict(s):
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=os.path.join('datos', 'PTBXL'))
    ap.add_argument('--csv', type=str, default=None)
    ap.add_argument('--top_k', type=int, default=30)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.root, 'ptbxl_database.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    out_path = args.output or os.path.join(args.root, 'labels_hierarchy.json')

    df = pd.read_csv(csv_path)
    counter = Counter()
    for s in df['scp_codes'].tolist():
        d = parse_scp_dict(s)
        counter.update(d.keys())

    fine_codes = [c for c, _ in counter.most_common(args.top_k)]
    # agrupar por familias básicas usando scp_statements.csv si existe
    groups = defaultdict(set)
    scp_map_path = os.path.join(args.root, 'scp_statements.csv')
    if os.path.exists(scp_map_path):
        mdf = pd.read_csv(scp_map_path)
        # usar 'diagnostic_class' como coarse
        for _, row in mdf.iterrows():
            code = row['Unnamed: 0'] if 'Unnamed: 0' in row else row.get('scp_code')
            if code in fine_codes:
                coarse = row.get('diagnostic_class', 'other')
                if pd.isna(coarse):
                    coarse = 'other'
                groups[str(coarse)].add(code)
    # asegurar que todos fine estén en algún grupo
    for c in fine_codes:
        found = False
        for g, s in groups.items():
            if c in s:
                found = True
                break
        if not found:
            groups['other'].add(c)

    groups_out = {g: sorted(list(v)) for g, v in groups.items()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'fine_codes': fine_codes, 'coarse_groups': groups_out}, f, ensure_ascii=False, indent=2)
    print('Escrito', out_path, 'con', len(fine_codes), 'fine codes y', len(groups_out), 'grupos')


if __name__ == '__main__':
    main()


