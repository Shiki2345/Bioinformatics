import json
import csv
import tempfile
import itertools
import re
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
import pyreadr

from src.config import OUTPUT_DIR, RAW_DATA_DIR


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_table_from_zip(zip_path: Path, txt_member: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    with ZipFile(zip_path, "r") as zf:
        with zf.open(txt_member, "r") as f:
            first_line = f.readline().decode("utf-8", "ignore")
            if "\t" in first_line:
                delimiter = "\t"
                skip_initial_space = False
            else:
                delimiter = " "
                skip_initial_space = True

            lines = itertools.chain(
                [first_line],
                (line.decode("utf-8", "ignore") for line in f),
            )
            reader = csv.reader(lines, delimiter=delimiter, quotechar='"', skipinitialspace=skip_initial_space)
            rows = []
            for idx, row in enumerate(reader):
                if not row:
                    continue
                rows.append(row)
                if max_rows is not None and idx >= max_rows:
                    break
    if not rows:
        raise ValueError(f"Empty text table in archive member: {txt_member}")
    header = rows[0]
    data_rows = rows[1:]
    cleaned = []
    for row in data_rows:
        if len(row) < len(header):
            row = row + [None] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        cleaned.append(row)
    df = pd.DataFrame(cleaned, columns=header)
    df = df.replace({"NA": pd.NA, "": pd.NA})
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Keep mixed categorical columns (e.g. disease/control) as text.
        if converted.notna().sum() >= max(5, int(0.95 * len(df))):
            df[col] = converted
    return df


def _probe_rdata_object(zip_path: Path, rdata_member: str) -> dict:
    with ZipFile(zip_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_rdata = Path(tmp_dir) / Path(rdata_member).name
            tmp_rdata.write_bytes(zf.read(rdata_member))
            result = pyreadr.read_r(str(tmp_rdata))

    probe = {}
    for obj_name, obj in result.items():
        obj_probe = {"type": str(type(obj))}
        if hasattr(obj, "shape"):
            obj_probe["shape"] = [int(obj.shape[0]), int(obj.shape[1])]
        if isinstance(obj, pd.DataFrame):
            obj_probe["columns_preview"] = [str(c) for c in obj.columns[:20]]
        probe[str(obj_name)] = obj_probe
    return probe


def probe_sample_archive(
    zip_path: Path = Path("sample_age_methylation_v1.zip"),
    rdata_member: str = "sample_age.RData",
    txt_member: str = "sample_age.txt",
) -> dict:
    _ensure_dirs()
    txt_df = _read_table_from_zip(zip_path, txt_member)
    rdata_probe = _probe_rdata_object(zip_path, rdata_member)
    summary = {
        "archive": str(zip_path),
        "txt_shape": [int(txt_df.shape[0]), int(txt_df.shape[1])],
        "txt_columns_preview": [str(c) for c in txt_df.columns[:25]],
        "rdata_probe": rdata_probe,
    }
    out = OUTPUT_DIR / "sample_archive_probe.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _pick_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["disease", "sample_type", "status", "group", "label"]
    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=True) >= 2:
            return col
    for col in df.columns:
        unique_n = df[col].nunique(dropna=True)
        if 2 <= unique_n <= 20 and not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _pick_age_column(df: pd.DataFrame) -> str:
    candidates = ["age", "Age", "patient_age"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Cannot infer age column from full dataset")


def _pick_sample_id_column(df: pd.DataFrame) -> str:
    candidates = ["sample_id", "SampleID", "id", "ID"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def _build_label_from_metadata(df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[str]]:
    if "sample_id" not in df.columns:
        return None, None

    if "disease" in df.columns:
        disease = df["disease"].astype("string").str.strip().str.lower()
        positive = disease.notna() & ~disease.isin({"", "na", "nan", "none", "null"})
        if positive.nunique(dropna=True) > 1 and positive.sum() > 0:
            return positive.astype(int), "disease_non_empty"

    candidates = ["label", "status", "group", "condition", "phenotype", "class", "sample_type"]
    pos_keywords = ("disease", "cancer", "tumor", "tumour", "case", "patient", "affected")
    neg_keywords = ("control", "normal", "healthy", "adjacent normal")
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col]
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().sum() >= int(0.95 * len(df)):
            uniq = set(as_num.dropna().astype(int).unique().tolist())
            if uniq.issubset({0, 1}) and len(uniq) >= 2:
                return as_num.fillna(0).astype(int), f"{col}_numeric_binary"

        as_txt = series.astype("string").str.strip().str.lower()
        pos = as_txt.str.contains("|".join(re.escape(k) for k in pos_keywords), na=False, regex=True)
        neg = as_txt.str.contains("|".join(re.escape(k) for k in neg_keywords), na=False, regex=True)
        mapped = pd.Series(pd.NA, index=df.index, dtype="Int64")
        mapped[pos] = 1
        mapped[neg] = 0
        if mapped.notna().sum() >= int(0.5 * len(df)) and mapped.nunique(dropna=True) >= 2:
            return mapped.fillna(0).astype(int), f"{col}_keyword_binary"

    return None, None


def _load_sample_metadata(
    zip_path: Path = Path("sample_age_methylation_v1.zip"),
    txt_member: str = "sample_age.txt",
) -> Optional[pd.DataFrame]:
    if not zip_path.exists():
        return None
    metadata = None
    with ZipFile(zip_path, "r") as zf:
        if "sample_age.RData" in zf.namelist():
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_rdata = Path(tmp_dir) / "sample_age.RData"
                tmp_rdata.write_bytes(zf.read("sample_age.RData"))
                r_objects = pyreadr.read_r(str(tmp_rdata))
                for obj in r_objects.values():
                    if isinstance(obj, pd.DataFrame) and "sample_id" in obj.columns:
                        metadata = obj.copy()
                        break
    if metadata is None:
        metadata = _read_table_from_zip(zip_path, txt_member)
    if "sample_id" not in metadata.columns:
        return None

    out = pd.DataFrame({"sample_id": metadata["sample_id"].astype(str)})
    if "age" in metadata.columns:
        out["age_from_metadata"] = pd.to_numeric(metadata["age"], errors="coerce")
    label, source = _build_label_from_metadata(metadata)
    if label is not None:
        out["label_from_metadata"] = label
        out.attrs["label_source"] = source
    return out


def _export_standard_csvs_from_table(df: pd.DataFrame, sample_metadata: Optional[pd.DataFrame] = None) -> dict:
    # Some provided txt files are feature x sample tables:
    # first column stores feature names and header stores sample IDs.
    if "age" not in df.columns and "sample_id" in str(df.columns[0]):
        feature_col = df.columns[0]
        wide = df.set_index(feature_col).transpose().reset_index()
        wide = wide.rename(columns={"index": "sample_id"})
        df = wide

    sample_col = _pick_sample_id_column(df)
    age_col = _pick_age_column(df)
    label_col = _pick_label_column(df)

    clinical = df.copy()
    rename_map = {sample_col: "sample_id", age_col: "age"}
    if label_col is not None:
        rename_map[label_col] = "label"
    clinical = clinical.rename(columns=rename_map)
    clinical["sample_id"] = clinical["sample_id"].astype(str)
    label_source = str(label_col) if label_col is not None else None
    if sample_metadata is not None and "sample_id" in sample_metadata.columns:
        merged = clinical.merge(sample_metadata, on="sample_id", how="left")
        if "age_from_metadata" in merged.columns:
            merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
            merged["age"] = merged["age_from_metadata"].combine_first(merged["age"])
        if "label_from_metadata" in merged.columns and merged["label_from_metadata"].notna().sum() > 0:
            merged["label"] = merged["label_from_metadata"]
            label_source = f"sample_metadata:{sample_metadata.attrs.get('label_source', 'unknown')}"
        clinical = merged.drop(columns=[c for c in ["age_from_metadata", "label_from_metadata"] if c in merged.columns])

    if "label" not in clinical.columns or clinical["label"].isna().all():
        age_median = pd.to_numeric(clinical["age"], errors="coerce").median()
        clinical["label"] = (pd.to_numeric(clinical["age"], errors="coerce") >= age_median).astype(int)
        label_source = "derived_from_age_median"
    else:
        clinical["label"] = pd.to_numeric(clinical["label"], errors="coerce")
        if clinical["label"].isna().sum() > 0:
            fill_value = int(clinical["label"].mode(dropna=True).iloc[0]) if clinical["label"].notna().any() else 0
            clinical["label"] = clinical["label"].fillna(fill_value)
        clinical["label"] = clinical["label"].astype(int)

    feature_blacklist = {"sample_id", "age", "label"}
    numeric_cols = [c for c in clinical.columns if c not in feature_blacklist and pd.api.types.is_numeric_dtype(clinical[c])]
    if len(numeric_cols) < 4:
        raise ValueError("Not enough numeric columns to construct feature matrix")

    mid = max(2, len(numeric_cols) // 2)
    expr_cols = numeric_cols[:mid]
    meth_cols = numeric_cols[mid:]
    if len(meth_cols) < 2:
        meth_cols = numeric_cols

    expr = clinical.set_index("sample_id")[expr_cols].transpose()
    expr.insert(0, "feature_id", expr.index.astype(str))
    meth = clinical.set_index("sample_id")[meth_cols].transpose()
    meth.insert(0, "feature_id", meth.index.astype(str))

    expression_path = RAW_DATA_DIR / "expression.csv"
    methylation_path = RAW_DATA_DIR / "methylation.csv"
    clinical_path = RAW_DATA_DIR / "clinical.csv"

    expr.to_csv(expression_path, index=False, encoding="utf-8-sig")
    meth.to_csv(methylation_path, index=False, encoding="utf-8-sig")
    clinical[["sample_id", "label", "age"]].to_csv(clinical_path, index=False, encoding="utf-8-sig")

    return {
        "sample_id_column": sample_col,
        "age_column": age_col,
        "label_column": label_source,
        "label_distribution": clinical["label"].value_counts(dropna=False).to_dict(),
        "expression_features": len(expr_cols),
        "methylation_features": len(meth_cols),
        "samples": int(clinical.shape[0]),
        "exported_files": {
            "expression": str(expression_path),
            "methylation": str(methylation_path),
            "clinical": str(clinical_path),
        },
    }


def export_full_archive_to_standard_csvs(
    zip_path: Path = Path("age_methylation_v1 (1).zip"),
    txt_member: str = "age_methylation_v1.txt",
    rdata_member: Optional[str] = "age_methylation_v1.RData",
    rdata_size_limit_bytes: int = 2 * 1024 * 1024 * 1024,
    txt_max_rows: Optional[int] = 120000,
) -> dict:
    _ensure_dirs()
    probe = {"archive": str(zip_path)}

    with ZipFile(zip_path, "r") as zf:
        members = {i.filename: i for i in zf.infolist()}
        probe["members"] = {k: int(v.file_size) for k, v in members.items()}
        use_rdata = bool(rdata_member and rdata_member in members and members[rdata_member].file_size <= rdata_size_limit_bytes)

    if use_rdata:
        # Keep architecture preference: RData-first when size is manageable.
        r_probe = _probe_rdata_object(zip_path, rdata_member)  # pragma: no cover (depends on archive size)
        probe["rdata_probe"] = r_probe

    full_df = _read_table_from_zip(zip_path, txt_member, max_rows=txt_max_rows)
    sample_metadata = _load_sample_metadata()
    export_info = _export_standard_csvs_from_table(full_df, sample_metadata=sample_metadata)
    probe["export_info"] = export_info
    probe["txt_shape"] = [int(full_df.shape[0]), int(full_df.shape[1])]
    probe["txt_max_rows_used"] = txt_max_rows

    out = OUTPUT_DIR / "full_archive_export_summary.json"
    out.write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
    return probe


def apply_real_labels_to_existing_clinical(
    sample_meta_zip_path: Path = Path("sample_age_methylation_v1.zip"),
    sample_meta_txt_member: str = "sample_age.txt",
) -> dict:
    _ensure_dirs()
    clinical_path = RAW_DATA_DIR / "clinical.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing clinical file: {clinical_path}")

    clinical = pd.read_csv(clinical_path)
    if "sample_id" not in clinical.columns:
        raise ValueError("clinical.csv must contain sample_id column")

    sample_metadata = _load_sample_metadata(sample_meta_zip_path, sample_meta_txt_member)
    if sample_metadata is None or "label_from_metadata" not in sample_metadata.columns:
        return {"updated": False, "reason": "sample metadata missing or no real label detected"}

    merged = clinical.merge(sample_metadata, on="sample_id", how="left")
    if merged["label_from_metadata"].notna().sum() == 0:
        return {"updated": False, "reason": "metadata label has zero overlap with clinical sample_id"}

    merged["label"] = merged["label_from_metadata"].combine_first(pd.to_numeric(merged.get("label"), errors="coerce"))
    merged["label"] = merged["label"].fillna(0).astype(int)
    if "age_from_metadata" in merged.columns:
        merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
        merged["age"] = merged["age_from_metadata"].combine_first(merged["age"])
    merged = merged.drop(columns=[c for c in ["label_from_metadata", "age_from_metadata"] if c in merged.columns])

    merged.to_csv(clinical_path, index=False, encoding="utf-8-sig")
    return {
        "updated": True,
        "label_source": f"sample_metadata:{sample_metadata.attrs.get('label_source', 'unknown')}",
        "label_distribution": merged["label"].value_counts(dropna=False).to_dict(),
        "rows": int(merged.shape[0]),
    }
