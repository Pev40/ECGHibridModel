import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_output_dir(base_outdir: Optional[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = base_outdir or os.path.join("reports", "eda", timestamp)
    ensure_directory(outdir)
    ensure_directory(os.path.join(outdir, "plots"))
    ensure_directory(os.path.join(outdir, "value_counts"))
    ensure_directory(os.path.join(outdir, "outliers"))
    ensure_directory(os.path.join(outdir, "duplicates"))
    return outdir


def infer_file_type(path: str, explicit_type: Optional[str]) -> str:
    if explicit_type and explicit_type.lower() in {"csv", "parquet"}:
        return explicit_type.lower()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    # Default to CSV if unknown
    return "csv"


def list_input_files(input_path: str, file_type: str, glob_pattern: Optional[str]) -> List[str]:
    if os.path.isdir(input_path):
        candidates: List[str] = []
        for root, _, files in os.walk(input_path):
            for f in files:
                lf = f.lower()
                if file_type == "csv" and lf.endswith(".csv"):
                    candidates.append(os.path.join(root, f))
                elif file_type == "parquet" and (lf.endswith(".parquet") or lf.endswith(".pq")):
                    candidates.append(os.path.join(root, f))
        if glob_pattern:
            import fnmatch

            candidates = [p for p in candidates if fnmatch.fnmatch(os.path.basename(p), glob_pattern)]
        candidates.sort()
        return candidates
    else:
        return [input_path]


def read_csv_safely(
    path: str,
    sep: str,
    encoding: Optional[str],
    nrows: Optional[int],
    parse_dates: Optional[List[str]],
    dtype: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        nrows=nrows,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
        engine="python",
    )


def read_parquet_safely(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def load_dataset(
    input_path: str,
    file_type: str,
    sep: str,
    encoding: Optional[str],
    limit_rows: Optional[int],
    glob_pattern: Optional[str],
    parse_date_cols: Optional[List[str]],
) -> pd.DataFrame:
    files = list_input_files(input_path, file_type=file_type, glob_pattern=glob_pattern)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos en {input_path} con tipo {file_type}")

    dataframes: List[pd.DataFrame] = []
    remaining = limit_rows

    for fp in files:
        if file_type == "csv":
            nrows = None
            if remaining is not None:
                if remaining <= 0:
                    break
                nrows = remaining
            df = read_csv_safely(fp, sep=sep, encoding=encoding, nrows=nrows, parse_dates=parse_date_cols)
            dataframes.append(df)
            if remaining is not None:
                remaining -= len(df)
        else:
            # For parquet, read full file; we'll sample after concat if limit_rows is set
            df = read_parquet_safely(fp)
            dataframes.append(df)

    if not dataframes:
        raise RuntimeError("No se pudo cargar ningún DataFrame")

    df_all = pd.concat(dataframes, ignore_index=True)

    if limit_rows is not None and file_type != "csv":
        df_all = df_all.head(limit_rows)

    return df_all


def try_parse_datetimes(df: pd.DataFrame, user_time_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    time_col = None
    if user_time_col and user_time_col in df.columns:
        time_col = user_time_col
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
        return df, time_col

    # Heurística básica
    for col in df.columns:
        if df[col].dtype == "O" and any(k in col.lower() for k in ["time", "fecha", "date", "ts", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            # Aceptar si al menos 80% son válidos
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= 0.8:
                df[col] = parsed
                time_col = col
                break
    return df, time_col


def summarize_overview(df: pd.DataFrame) -> Dict[str, object]:
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "memory_mb": round(memory_mb, 3),
        "columns": list(df.columns),
    }


def get_dtype_name(dtype: np.dtype) -> str:
    try:
        return str(dtype)
    except Exception:
        return "unknown"


def classify_feature_types(df: pd.DataFrame, max_categories: int = 50) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        num_unique = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(dtype):
            ratio = num_unique / max(1, n)
            feature_kind = "discreta" if (num_unique <= max_categories or ratio < 0.05) else "continua"
            inferred = "numerica"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            feature_kind = "tiempo"
            inferred = "datetime"
        else:
            feature_kind = "categorica"
            inferred = "object"
        rows.append(
            {
                "columna": col,
                "dtype": get_dtype_name(dtype),
                "tipo_inferido": inferred,
                "cardinalidad": int(num_unique),
                "clase_variable": feature_kind,
            }
        )
    return pd.DataFrame(rows)


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for col in df.columns:
        na_count = int(df[col].isna().sum())
        na_pct = (na_count / max(1, total)) * 100
        rows.append({"columna": col, "nulos": na_count, "nulos_%": round(na_pct, 3)})
    return pd.DataFrame(rows).sort_values("nulos_%", ascending=False)


def numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    desc = num_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc.rename(
        columns={
            "25%": "q1",
            "50%": "mediana",
            "75%": "q3",
            "std": "desv_std",
            "mean": "media",
            "min": "min",
            "max": "max",
            "count": "conteo",
        },
        inplace=True,
    )
    desc["sesgo"] = num_df.skew(numeric_only=True)
    desc["curtosis"] = num_df.kurtosis(numeric_only=True)

    # Outliers por IQR
    def iqr_outliers_count(col: pd.Series) -> Tuple[int, float]:
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return 0, 0.0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (col < lower) | (col > upper)
        count = int(mask.sum())
        pct = (count / max(1, len(col))) * 100
        return count, pct

    out_count = []
    out_pct = []
    for col in num_df.columns:
        c, p = iqr_outliers_count(num_df[col].dropna())
        out_count.append(c)
        out_pct.append(p)

    desc["outliers_iqr"] = out_count
    desc["outliers_iqr_%"] = [round(p, 3) for p in out_pct]

    # Normalidad (D'Agostino) sobre una muestra si es muy grande
    pvals = []
    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 8:
            pvals.append(np.nan)
            continue
        sample = series
        if len(series) > 50000:
            sample = series.sample(50000, random_state=42)
        try:
            k2, p = stats.normaltest(sample)
            pvals.append(float(p))
        except Exception:
            pvals.append(np.nan)
    desc["p_normalidad_dagostino"] = pvals

    return desc.reset_index().rename(columns={"index": "columna"})


def categorical_value_counts(df: pd.DataFrame, max_categories: int = 50) -> Dict[str, pd.DataFrame]:
    cat_df = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, tz]"])
    results: Dict[str, pd.DataFrame] = {}
    for col in cat_df.columns:
        vc = cat_df[col].astype("object").value_counts(dropna=False).head(max_categories)
        results[col] = vc.rename("conteo").to_frame().reset_index().rename(columns={"index": col})
    return results


def compute_correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    corr = num_df.corr(method=method)
    return corr


def duplicates_summary(
    df: pd.DataFrame, subset_cols: Optional[List[str]] = None
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    # Duplicados por fila completa
    full_dups = int(df.duplicated().sum())
    summary["duplicados_completos"] = full_dups
    # Duplicados por subconjunto de columnas
    if subset_cols:
        missing = [c for c in subset_cols if c not in df.columns]
        if missing:
            summary["subset_invalido"] = {
                "faltantes": missing,
            }
        else:
            subset_dups = int(df.duplicated(subset=subset_cols).sum())
            summary["duplicados_por_subset"] = {
                "subset": subset_cols,
                "conteo": subset_dups,
            }
    return summary


def class_balance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset")
    vc = df[target].value_counts(dropna=False)
    pct = (vc / max(1, len(df))) * 100
    out = pd.DataFrame({"clase": vc.index.astype(str), "conteo": vc.values, "porcentaje": pct.values})
    return out


def save_plots(
    df: pd.DataFrame,
    outdir: str,
    plots_enabled: bool,
    correlation_method: str,
    max_categories: int,
    time_col: Optional[str],
    target: Optional[str],
) -> None:
    if not plots_enabled:
        return

    plots_dir = os.path.join(outdir, "plots")

    # Config visual
    sns.set(style="whitegrid")

    # Histogramas y boxplots para numéricos
    num_df = df.select_dtypes(include=[np.number])
    for col in tqdm(num_df.columns, desc="Graficando numéricos"):
        series = num_df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(series, kde=True, ax=ax)
        ax.set_title(f"Histograma - {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"hist_{col}.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x=series, ax=ax)
        ax.set_title(f"Boxplot - {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"box_{col}.png"))
        plt.close(fig)

    # Barras para categóricas (top N)
    cat_df = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, tz]"])
    for col in tqdm(cat_df.columns, desc="Graficando categóricas"):
        vc = cat_df[col].astype("object").value_counts(dropna=False).head(max_categories)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h")
        ax.set_title(f"Barras - {col}")
        ax.set_xlabel("conteo")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"bar_{col}.png"))
        plt.close(fig)

    # Correlación heatmap
    corr = compute_correlations(df, method=correlation_method)
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(max(6, int(corr.shape[0] * 0.6)), max(5, int(corr.shape[1] * 0.6))))
        sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
        ax.set_title(f"Matriz de correlación ({correlation_method})")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "corr_heatmap.png"))
        plt.close(fig)

    # Nulos por columna (barras)
    na_summary = summarize_missingness(df)
    if not na_summary.empty:
        fig, ax = plt.subplots(figsize=(max(7, int(df.shape[1] * 0.5)), 4))
        sns.barplot(data=na_summary, x="nulos_%", y="columna", ax=ax, orient="h")
        ax.set_title("Porcentaje de nulos por columna")
        ax.set_xlabel("nulos (%)")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "nulos_por_columna.png"))
        plt.close(fig)

    # Series temporales (si hay columna de tiempo)
    if time_col and time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        ts = df[[time_col]].dropna().copy()
        if not ts.empty:
            ts["fecha"] = ts[time_col].dt.to_period("D").dt.to_timestamp()
            per_day = ts.groupby("fecha").size().rename("conteo").reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=per_day, x="fecha", y="conteo", ax=ax)
            ax.set_title("Conteo de registros por día")
            ax.set_xlabel("fecha")
            ax.set_ylabel("conteo")
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "time_series_counts_daily.png"))
            plt.close(fig)

            if target and target in df.columns:
                tmp = df[[time_col, target]].dropna().copy()
                tmp["fecha"] = tmp[time_col].dt.to_period("D").dt.to_timestamp()
                counts = tmp.groupby(["fecha", target]).size().rename("conteo").reset_index()
                total = counts.groupby("fecha")["conteo"].transform("sum")
                counts["porcentaje"] = counts["conteo"] / total * 100.0
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=counts, x="fecha", y="porcentaje", hue=target, ax=ax)
                ax.set_title("Balance de clases por día (%)")
                ax.set_xlabel("fecha")
                ax.set_ylabel("%")
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "time_series_class_balance_daily.png"))
                plt.close(fig)


def export_outlier_indices(df: pd.DataFrame, outdir: str) -> None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return
    out_dir = os.path.join(outdir, "outliers")
    ensure_directory(out_dir)
    for col in num_df.columns:
        series = num_df[col]
        s = series.dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        idx = df.index[mask]
        if len(idx) == 0:
            continue
        sample_idx = idx[:1000]
        out = pd.DataFrame({"index": sample_idx})
        out.to_csv(os.path.join(out_dir, f"outliers_iqr_indices_{col}.csv"), index=False)


def export_duplicates(
    df: pd.DataFrame, outdir: str, subset_cols: Optional[List[str]] = None
) -> None:
    dup_dir = os.path.join(outdir, "duplicates")
    ensure_directory(dup_dir)

    # Duplicados globales (filas completas)
    mask_full = df.duplicated(keep=False)
    if mask_full.any():
        sample = df[mask_full].head(1000)
        sample.to_csv(os.path.join(dup_dir, "duplicates_full_sample.csv"), index=False)

    # Duplicados por subset
    if subset_cols and all(c in df.columns for c in subset_cols):
        mask_subset = df.duplicated(subset=subset_cols, keep=False)
        if mask_subset.any():
            sample_sub = df[mask_subset].head(1000)
            sample_sub.to_csv(
                os.path.join(dup_dir, "duplicates_subset_sample.csv"), index=False
            )


def run_eda(
    input_path: str,
    file_type: str,
    sep: str,
    encoding: Optional[str],
    limit_rows: Optional[int],
    glob_pattern: Optional[str],
    outdir: str,
    target: Optional[str],
    time_col: Optional[str],
    correlation_method: str,
    max_categories: int,
    plots_enabled: bool,
    dup_subset: Optional[List[str]] = None,
) -> None:
    print(f"Cargando datos desde: {input_path}")
    df = load_dataset(
        input_path=input_path,
        file_type=file_type,
        sep=sep,
        encoding=encoding,
        limit_rows=limit_rows,
        glob_pattern=glob_pattern,
        parse_date_cols=[time_col] if time_col else None,
    )

    df, inferred_time_col = try_parse_datetimes(df, user_time_col=time_col)
    if inferred_time_col:
        print(f"Columna temporal detectada: {inferred_time_col}")

    overview = summarize_overview(df)
    print(json.dumps(overview, ensure_ascii=False, indent=2))

    # Tipos y categorías
    types_df = classify_feature_types(df, max_categories=max_categories)
    na_df = summarize_missingness(df)
    num_desc = numeric_describe(df)
    cat_vcs = categorical_value_counts(df, max_categories=max_categories)
    corr = compute_correlations(df, method=correlation_method)

    # Duplicados
    dups = duplicates_summary(df, subset_cols=dup_subset)

    # Exportar CSVs
    pd.DataFrame([overview]).to_csv(os.path.join(outdir, "overview.csv"), index=False)
    types_df.to_csv(os.path.join(outdir, "schema_dtypes.csv"), index=False)
    na_df.to_csv(os.path.join(outdir, "na_summary.csv"), index=False)
    if not num_desc.empty:
        num_desc.to_csv(os.path.join(outdir, "numeric_summary.csv"), index=False)
    for col, vcdf in cat_vcs.items():
        vcdf.to_csv(os.path.join(outdir, "value_counts", f"value_counts_{col}.csv"), index=False)
    if not corr.empty:
        corr.to_csv(os.path.join(outdir, "correlation_numeric.csv"))

    # Duplicados
    with open(os.path.join(outdir, "duplicates_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dups, f, ensure_ascii=False, indent=2)
    export_duplicates(df, outdir, subset_cols=dup_subset)

    # Balance de clases si target
    if target:
        try:
            bal = class_balance(df, target)
            bal.to_csv(os.path.join(outdir, "class_balance.csv"), index=False)
        except Exception as e:
            print(f"Aviso: no se pudo calcular balance de clases: {e}")

    # Outliers indices (muestras)
    export_outlier_indices(df, outdir)

    # Plots
    save_plots(
        df,
        outdir,
        plots_enabled=plots_enabled,
        correlation_method=correlation_method,
        max_categories=max_categories,
        time_col=inferred_time_col,
        target=target,
    )

    # Guardar un resumen JSON
    summary = {
        "overview": overview,
        "num_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "cat_columns": df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime64[ns, tz]"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime64[ns]"]).columns.tolist(),
        "inferred_time_col": inferred_time_col,
        "target": target,
        "correlation_method": correlation_method,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"EDA completado. Reportes guardados en: {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA rápido y parametrizable para datasets tabulares (ECG u otros).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Ruta a archivo o carpeta con datos (CSV/Parquet)")
    parser.add_argument(
        "--file-type",
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="Tipo de archivo de entrada",
    )
    parser.add_argument("--sep", default=",", help="Separador CSV")
    parser.add_argument("--encoding", default=None, help="Encoding del archivo CSV")
    parser.add_argument("--limit-rows", type=int, default=None, help="Limitar filas cargadas (para memoria)")
    parser.add_argument("--glob", default=None, help="Filtro glob cuando input es carpeta (p.ej. '*.csv')")
    parser.add_argument("--outdir", default=None, help="Carpeta de salida para reportes")
    parser.add_argument("--target", default=None, help="Nombre de la columna objetivo (si aplica)")
    parser.add_argument("--time-col", default=None, help="Nombre de la columna temporal (si aplica)")
    parser.add_argument(
        "--corr",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Método de correlación",
    )
    parser.add_argument("--max-categories", type=int, default=50, help="Top categorías a considerar")
    parser.add_argument("--no-plots", action="store_true", help="Desactivar generación de gráficos")
    parser.add_argument(
        "--id-cols",
        default=None,
        help="Columnas (separadas por coma) para detectar duplicados por identificador",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ftype = infer_file_type(args.input, None if args.file_type == "auto" else args.file_type)
    outdir = resolve_output_dir(args.outdir)
    run_eda(
        input_path=args.input,
        file_type=ftype,
        sep=args.sep,
        encoding=args.encoding,
        limit_rows=args.limit_rows,
        glob_pattern=args.glob,
        outdir=outdir,
        target=args.target,
        time_col=args.time_col,
        correlation_method=args.corr,
        max_categories=args.max_categories,
        plots_enabled=not args.no_plots,
        dup_subset=[c.strip() for c in args.id_cols.split(",")] if args.id_cols else None,
    )


if __name__ == "__main__":
    main()


