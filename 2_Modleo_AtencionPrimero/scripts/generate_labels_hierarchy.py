import os
import sys
import json
import argparse
from glob import glob
from collections import Counter, defaultdict


def parse_header_labels(hea_path):
    labels = []
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                if ln.startswith('#Dx:'):
                    txt = ln.replace('#Dx:', '').strip()
                    parts = [p for p in re_split(txt) if p]
                    labels = parts
                    break
    except Exception:
        pass
    return labels


def re_split(s):
    # split by comma/semicolon/space
    import re
    return re.split(r'[,:;\s]+', s)


def load_condition_csv(csv_path):
    # optional: map SNOMED to acronyms/names if needed
    m = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            next(f, None)
            for ln in f:
                parts = [p.strip() for p in ln.strip().split(',')]
                if len(parts) >= 3:
                    acr, fullname, snomed = parts[0], \
                                            parts[1], \
                                            parts[2]
                    m[snomed] = {'acronym': acr, 'name': fullname}
    return m


def default_coarse_groups():
    # Clinical grouping rules (expandable). Any unmapped code goes to "other".
    return {
        'rhythm': {
            '426177001', '426783006', '427084000', '164890007', '164889003',
            '426761007', '713422000', '713427006', '427172004'
        },
        'conduction': {
            '270492004', '195042002', '27885002', '59118001', '164909002',
            '698252002', '426995002', '251164006', '74390002'
        },
        'st_t_ischemia_mi': {
            '429622005', '164930006', '164934002', '59931005', '164917005',
            '164865005', '111975006'
        },
        'hypertrophy': {
            '164873001', '89792004', '446358003'
        },
        'ectopy': {
            '284470004', '17338001', '75532003'
        },
        'axis_rotation_voltage': {
            '39732003', '47665007', '251146004', '251198002', '251199005',
            '365413008'
        },
        'other': {
            '55827005', '55930002', '10370003', '61721007', '6374002', '428417006'
        }
    }


def assign_to_coarse(code, groups):
    for g, codes in groups.items():
        if code in codes and g != 'other':
            return g
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=os.path.join('datos', 'WFDBRecords'), help='Directorio raíz de WFDBRecords')
    ap.add_argument('--top_k', type=int, default=30, help='Número de códigos SNOMED más frecuentes a usar como fine')
    ap.add_argument('--output', type=str, default=os.path.join('datos', 'labels_hierarchy.json'))
    ap.add_argument('--condition_csv', type=str, default=os.path.join('datos', 'ConditionNames_SNOMED-CT.csv'))
    args = ap.parse_args()

    hea_files = glob(os.path.join(args.root, '**', '*.hea'), recursive=True)
    if not hea_files:
        print('No se encontraron .hea en', args.root)
        sys.exit(1)

    counter = Counter()
    for hea in hea_files:
        labels = parse_header_labels(hea)
        counter.update(labels)

    stats = counter.most_common()
    fine_codes = [c for c, _ in stats[:args.top_k]]

    groups = default_coarse_groups()
    # aseguremos que cualquier fine no mapeado quede en other
    for c in fine_codes:
        g = assign_to_coarse(c, groups)
        if c not in groups[g]:
            groups[g].add(c)

    # serializar sets a listas
    groups_out = {g: sorted(list(codes)) for g, codes in groups.items()}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'fine_codes': fine_codes, 'coarse_groups': groups_out}, f, ensure_ascii=False, indent=2)
    print('Escrito', args.output, 'con', len(fine_codes), 'fine codes y', len(groups_out), 'grupos')


if __name__ == '__main__':
    main()


