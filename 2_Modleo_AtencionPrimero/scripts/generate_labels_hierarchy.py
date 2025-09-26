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
    """
    Clinical coarse grouping (SNOMED CT codes as strings).
    Any unmapped/uncertain codes were put in 'other'.
    """
    return {
        'rhythm': {
            # Sinus / atrial / tachyarrhythmias / ventricular arrhythmias / paced rhythms
            '426177001',  # Sinus Bradycardia (SB)
            '426783006',  # Sinus Rhythm (SR)
            '427084000',  # Sinus Tachycardia (ST)
            '164890007',  # Atrial Flutter (AF)
            '164889003',  # Atrial Fibrillation (AFIB)
            '426761007',  # Supraventricular Tachycardia (SVT)
            '713422000',  # Atrial Tachycardia (AT)
            '233896004',  # AVNRT (Atrioventricular Node Reentrant Tachycardia)
            '233897008',  # AVRT (Atrioventricular Reentrant Tachycardia)
            '251166008',  # AVNRT (disorder) - alternate code
            '17366009',   # Atrial arrhythmia (AA)
            '29320008',   # Ectopic rhythm (ER) -- rhythm abnormality
            '425856008',  # Paroxysmal ventricular tachycardia (PVT)
            '164896001',  # Ventricular fibrillation (VF)
            '111288001',  # Ventricular flutter (VFL)
            '81898007',   # Ventricular escape rhythm (VER) - escape rhythm
            '233892002',  # Ectopic atrial tachycardia (EAT)
            '426648003',  # Junctional tachycardia (JT)
            '61277005',   # Accelerated idioventricular rhythm (AIVR)
            '426664006',  # Accelerated junctional rhythm (AJR)
            '5609005',    # Sinus arrest (SAR)
            '10370003',   # Rhythm from artificial pacing (APACE)
            '426627000',  # Electrocardiogram: bradycardia (BRADY)
            '106068003',  # Atrial rhythm (AR)
            '195101003',  # Wandering atrial/AV node pacemaker (WAVN / SAAWR)
        },

        'conduction': {
            # AV blocks, bundle branch blocks, fascicular blocks, pre-excitation, PR changes
            '270492004',  # 1st degree AV block (1AVB)
            '195042002',  # 2nd degree AV block (2AVB)
            '54016002',   # 2nd degree AV block (type I)
            '28189009',   # 2nd degree AV block (type II)
            '27885002',   # 3rd degree AV block (3AVB)
            '233917008',  # Atrioventricular block (AVB)
            '698252002',  # Intraventricular block / interior differences conduction (IDC / IVB)
            '164909002',  # Left bundle branch block (LBBB) and variants
            '59118001',   # Right bundle branch block (RBBB)
            '713427006',  # Complete right bundle branch block (CRBBB)
            '713426002',  # Incomplete right bundle branch block (IRBBB)
            '733534002',  # Complete left bundle branch block (CLBBB)
            '445118002',  # Left anterior fascicular block (LAFB)
            '445211001',  # Left posterior fascicular block (LPFB)
            '426183003',  # Mobitz type II (MOB2)
            '65778007',   # Sinoatrial block (SAB)
            '50799005',   # Atrioventricular dissociation (AVD)
            '195060002',  # Ventricular preexcitation (VPE) - WPW-like physiology
            '74390002',   # WPW
            '49578007',   # Shortened PR interval (SPR)
            '251120003',  # Incomplete LBBB (ILBBB)
            '6374002',    # Bundle branch block (BBB) general
        },

        'st_t_ischemia_mi': {
            # ST/T changes, Q waves, MI codes, repolarization abnormalities (QT/U)
            '164865005',  # Myocardial infarction (MI) and variants
            '164917005',  # Abnormal Q wave
            '428417006',  # Early repolarization of the ventricles (ER)
            '164942001',  # fQRS (fragmented QRS) - often related to scar/MI
            '429622005',  # ST drop down (STDD)
            '164930006',  # ST extension (STE)
            '428750005',  # ST-T Change (STTC)
            '164931005',  # ST tilt up (STTU)
            '164934002',  # T wave Change (TWC)
            '59931005',   # T wave opposite (TWO)
            '164937009',  # U wave (UW)
            '111975006',  # QT interval extension (QTIE)
            '55930002',   # ST segment changes (STC)
            '57054005',   # Acute myocardial infarction (AMI)
            '54329005',   # AMI anterior wall (AMI-AW)
            '77867006',   # Shortened QT (SQT)
        },

        'hypertrophy': {
            # Atrial/ventricular hypertrophy or atrial enlargement / P-wave changes suggestive of chamber enlargement
            '164873001',  # Left ventricle hypertrophy (LVH) - finding
            '55827005',   # Left ventricular hypertrophy (disorder) (duplicate style)
            '446358003',  # Right atrial hypertrophy (RAH)
            '89792004',   # Right ventricle hypertrophy (RVH)
            '67751000119106', # Right atrial enlargement (RAE)
            '446813000',  # Left atrial hypertrophy (LAH)
            '67741000119109', # Left atrial enlargement (LAE)
            '251223006',  # Tall P wave (TPW) - often atrial enlargement
            '251205003',  # Prolonged P wave (PPW) - LA abnormality
            '164912004',  # P wave Change (PWC) -> atrial changes (moved to hypertrophy/atrial category)
        },

        'ectopy': {
            # Premature/ectopic beats, bigeminy/trigeminy, fusion, escape beats/complexes
            '284470004',  # Atrial premature beats (APB)
            '17338001',   # Ventricular premature beat (VPB)
            '75532003',   # Ventricular escape beat (VEB)
            '11157007',   # Ventricular bigeminy (VB)
            '13640000',   # Ventricular fusion wave (VFW)
            '251180001',  # Ventricular escape trigeminy (VET)
            '251187003',  # Atrial escape complex (AEC)
            '251170000',  # Blocked premature atrial contraction (BPAC)
            '251164006',  # Junctional premature beat (JPT)
            '426995002',  # Junctional escape beat (JEB)
            '427172004',  # Premature ventricular contractions (PVC)
            '63593006',   # Supraventricular premature beats (SVPB)
            '251173003',  # Atrial bigeminy (ABI)
        },

        'axis_rotation_voltage': {
            # Axis shifts, rotations, low voltage QRS, R-wave findings, vectorcardiographic loops
            '39732003',   # Axis left shift (ALS)
            '47665007',   # Axis right shift (ARS)
            '251146004',  # Lower voltage QRS in all leads (LVQRSAL)
            '251148003',  # Lower voltage QRS in chest leads (LVQRSCL)
            '251147008',  # Lower voltage QRS in limb leads (LVQRSLL)
            '251199005',  # Counterclockwise rotation (CCR)
            '251198002',  # Clockwise rotation (CR)
            '365413008',  # Finding of R wave (RW)
            '61721007',   # Counterclockwise vectorcardiographic loop (CCVL)
        },

        'other': {
            # Syndromes or miscellaneous items that don't fit cleanly in the above coarse buckets
            '418818005',  # Brugada syndrome (BRUG) - channelopathy / syndrome (kept in other)
            # (If you prefer, BRUG can be placed in 'st_t_ischemia_mi' because of ST-elevation phenotype,
            #  or in 'conduction' because it's an ion-channel/conduction disorder.)
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
    ap.add_argument('--root', type=str, default=os.path.join('datos', '12Large', 'WFDBRecords'), help='Directorio raíz de WFDBRecords')
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


