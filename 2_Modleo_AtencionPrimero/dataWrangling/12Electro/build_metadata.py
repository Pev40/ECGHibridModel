import argparse
import csv
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def parse_wfdb_header_first_line(hea_path: str) -> Dict[str, Optional[float]]:
    """Intenta extraer numero de derivaciones, fs y numero de muestras de la primera línea WFDB.

    Formato típico: <record_name> <n_signals> <fs> <n_samples> ...
    """
    result = {"num_leads": None, "fs": None, "num_samples": None}
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        parts = first.split()
        if len(parts) >= 4:
            # parts[1]: n_signals, parts[2]: fs, parts[3]: n_samples (convención WFDB)
            try:
                result["num_leads"] = int(float(parts[1]))
            except Exception:
                pass
            try:
                result["fs"] = float(parts[2])
            except Exception:
                # fallback: buscar Hz
                m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", first, re.IGNORECASE)
                if m:
                    result["fs"] = float(m.group(1))
            try:
                result["num_samples"] = int(float(parts[3]))
            except Exception:
                pass
    except Exception:
        pass
    return result


def parse_12large_labels(hea_path: str) -> List[str]:
    labels: List[str] = []
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("#Dx:"):
                    txt = ln.replace("#Dx:", "").strip()
                    parts = re.split(r"[,:;\s]+", txt)
                    labels = [p for p in parts if p]
                    break
    except Exception:
        pass
    return labels


def extract_patient_id_from_header(hea_path: str, root_dir: Optional[str] = None) -> Optional[str]:
    pid = None
    try:
        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("#"):
                    m = re.search(r"(Patient ID|Patient|Subject ID|Subject|Record ID|PID)\s*:?\s*([\w\-]+)", ln, re.IGNORECASE)
                    if m:
                        pid = m.group(2)
                        break
    except Exception:
        pass
    if pid is None and root_dir is not None:
        try:
            rel = os.path.relpath(hea_path, root_dir)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                pid = parts[0]
        except Exception:
            pid = None
    if pid is None:
        stem = os.path.splitext(os.path.basename(hea_path))[0]
        pid = re.split(r"[_\-]", stem)[0]
    return pid


def build_metadata_12large(root_dir: str, out_csv: str) -> None:
    rows: List[Dict[str, object]] = []
    for r, _, fs in os.walk(root_dir):
        for f in fs:
            if f.lower().endswith(".hea"):
                hea = os.path.join(r, f)
                stem = os.path.splitext(hea)[0]
                mat_path = stem + ".mat"
                info = parse_wfdb_header_first_line(hea)
                labels = parse_12large_labels(hea)
                pid = extract_patient_id_from_header(hea, root_dir)
                rows.append(
                    {
                        "hea_path": os.path.relpath(hea, root_dir),
                        "mat_exists": os.path.exists(mat_path),
                        "patient_id": pid,
                        "num_leads": info.get("num_leads"),
                        "fs": info.get("fs"),
                        "num_samples": info.get("num_samples"),
                        "num_labels": len(labels),
                        "labels": ",".join(labels),
                    }
                )
    ensure_dir(out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _ptbxl_parse_scp_codes(s: object) -> Dict[str, float]:
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            try:
                import ast

                return ast.literal_eval(s)
            except Exception:
                return {}
    return {}


def build_metadata_ptbxl(root_dir: str, out_csv: str, use_high_res: bool = True) -> None:
    csv_path = os.path.join(root_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró ptbxl_database.csv en {root_dir}")
    df = pd.read_csv(csv_path)
    base_subdir = "records500" if use_high_res else "records100"
    rel_col = "filename_hr" if use_high_res else "filename_lr"

    # Derivados
    scp = df.get("scp_codes")
    num_codes: List[int] = []
    for v in scp:
        d = _ptbxl_parse_scp_codes(v)
        num_codes.append(len(d))

    out = pd.DataFrame(
        {
            "record_rel": df[rel_col] + ".hea",
            "patient_id": df.get("patient_id", pd.Series([None] * len(df))),
            "age": df.get("age", pd.Series([None] * len(df))),
            "sex": df.get("sex", pd.Series([None] * len(df))),
            "strat_fold": df.get("strat_fold", pd.Series([None] * len(df))),
            "n_scp_codes": num_codes,
        }
    )

    ensure_dir(out_csv)
    out.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye CSVs de metadatos para 12Large y PTBXL")
    parser.add_argument("--dataset", choices=["12large", "ptbxl"], required=True)
    parser.add_argument("--root", required=True, help="Ruta raíz del dataset en disco")
    parser.add_argument("--out", required=True, help="Ruta del CSV de salida")
    parser.add_argument("--lowres", action="store_true", help="Para PTBXL, usar registros 100Hz")
    args = parser.parse_args()

    if args.dataset == "12large":
        build_metadata_12large(args.root, args.out)
    else:
        build_metadata_ptbxl(args.root, args.out, use_high_res=not args.lowres)


if __name__ == "__main__":
    main()


