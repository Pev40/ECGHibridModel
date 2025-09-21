import os
import sys
import json
from glob import glob
from collections import Counter

# Permitir ejecución directa: añadir el root del repo al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.wfdb_dataset import _parse_header_labels


def main():
    root = os.path.join('datos', 'WFDBRecords')
    hea_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
    counter = Counter()
    for hea in hea_files:
        labels = _parse_header_labels(hea)
        counter.update(labels)

    stats = counter.most_common()
    print(f"Total etiquetas distintas: {len(stats)}")
    print("Top 50 etiquetas:")
    for code, cnt in stats[:50]:
        print(code, cnt)

    out = os.path.join('datos', 'label_stats.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print('Guardado:', out)


if __name__ == '__main__':
    main()


