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


def generate_labels_hierarchy(root_dir, output_path, top_k=30, condition_csv=None):
    """Genera un archivo labels_hierarchy.json a partir de cabeceras WFDB (#Dx).

    Parameters:
        root_dir (str): Directorio raíz donde buscar recursivamente archivos .hea
        output_path (str): Ruta de salida para escribir el JSON de jerarquía
        top_k (int): Número de códigos SNOMED más frecuentes a mantener como fine codes
        condition_csv (str|None): CSV opcional para mapear SNOMED a nombres/acrónimos (no obligatorio)

    Returns:
        dict: {'fine_codes': [...], 'coarse_groups': {...}}
    """
    hea_files = glob(os.path.join(root_dir, '**', '*.hea'), recursive=True)
    if not hea_files:
        raise FileNotFoundError(f'No se encontraron .hea en {root_dir}')

    counter = Counter()
    for hea in hea_files:
        labels = parse_header_labels(hea)
        counter.update(labels)

    stats = counter.most_common()
    fine_codes = [c for c, _ in stats[:int(top_k)]]

    groups = default_coarse_groups()
    # asegurar que cualquier fine no mapeado quede en other
    for c in fine_codes:
        g = assign_to_coarse(c, groups)
        if c not in groups[g]:
            groups[g].add(c)

    groups_out = {g: sorted(list(codes)) for g, codes in groups.items()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'fine_codes': fine_codes, 'coarse_groups': groups_out}, f, ensure_ascii=False, indent=2)
    return {'fine_codes': fine_codes, 'coarse_groups': groups_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=os.path.join('datos', 'WFDBRecords'), help='Directorio raíz de WFDBRecords')
    ap.add_argument('--top_k', type=int, default=30, help='Número de códigos SNOMED más frecuentes a usar como fine')
    ap.add_argument('--output', type=str, default=os.path.join('datos', 'labels_hierarchy.json'))
    ap.add_argument('--condition_csv', type=str, default=os.path.join('datos', 'ConditionNames_SNOMED-CT.csv'))
    args = ap.parse_args()

    try:
        res = generate_labels_hierarchy(args.root, args.output, top_k=args.top_k, condition_csv=args.condition_csv)
        print('Escrito', args.output, 'con', len(res['fine_codes']), 'fine codes y', len(res['coarse_groups']), 'grupos')
    except FileNotFoundError:
        print('No se encontraron .hea en', args.root)
        sys.exit(1)


if __name__ == '__main__':
    main()


