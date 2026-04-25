import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    AGE_COLUMN_CANDIDATES,
    CLINICAL_FILE,
    EXPRESSION_FILE,
    LABEL_COLUMN_CANDIDATES,
    METHYLATION_FILE,
    OUTPUT_DIR,
    PROCESSED_DIR,
    SAMPLE_ID_COLUMN_CANDIDATES,
)


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    age_train: np.ndarray
    age_test: np.ndarray
    feature_matrix: np.ndarray
    labels: np.ndarray
    ages: np.ndarray


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _detect_column(candidates: list[str], columns: list[str]) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"Cannot find required column in {columns}")


def _infer_sample_ids(df: pd.DataFrame) -> pd.Index:
    columns = list(df.columns)
    for candidate in SAMPLE_ID_COLUMN_CANDIDATES:
        if candidate in columns:
            return pd.Index(df[candidate].astype(str))
    # For omics files, columns are expected to be sample IDs.
    return pd.Index(columns[1:] if len(columns) > 1 else columns).astype(str)


def run_data_check() -> dict:
    _ensure_dirs()

    expression = _read_required_csv(EXPRESSION_FILE)
    methylation = _read_required_csv(METHYLATION_FILE)
    clinical = _read_required_csv(CLINICAL_FILE)

    sample_id_col = _detect_column(SAMPLE_ID_COLUMN_CANDIDATES, list(clinical.columns))
    label_col = _detect_column(LABEL_COLUMN_CANDIDATES, list(clinical.columns))
    age_col = _detect_column(AGE_COLUMN_CANDIDATES, list(clinical.columns))

    exp_ids = _infer_sample_ids(expression)
    meth_ids = _infer_sample_ids(methylation)
    clin_ids = pd.Index(clinical[sample_id_col].astype(str))
    common_ids = sorted(set(exp_ids).intersection(set(meth_ids)).intersection(set(clin_ids)))

    summary = {
        "files_loaded": {
            "expression": str(EXPRESSION_FILE),
            "methylation": str(METHYLATION_FILE),
            "clinical": str(CLINICAL_FILE),
        },
        "rows_cols": {
            "expression": [int(expression.shape[0]), int(expression.shape[1])],
            "methylation": [int(methylation.shape[0]), int(methylation.shape[1])],
            "clinical": [int(clinical.shape[0]), int(clinical.shape[1])],
        },
        "sample_counts": {
            "expression_samples": int(len(exp_ids)),
            "methylation_samples": int(len(meth_ids)),
            "clinical_samples": int(len(clin_ids)),
            "common_samples": int(len(common_ids)),
        },
        "missing_ratio": {
            "expression": float(expression.isna().mean().mean()),
            "methylation": float(methylation.isna().mean().mean()),
            "clinical": float(clinical.isna().mean().mean()),
        },
        "clinical_columns": {
            "sample_id": sample_id_col,
            "label": label_col,
            "age": age_col,
        },
        "label_distribution": clinical[label_col].value_counts(dropna=False).to_dict(),
    }

    summary_path = OUTPUT_DIR / "data_check_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _reindex_omics_samples(df: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    if df.columns[0] in SAMPLE_ID_COLUMN_CANDIDATES:
        df = df.set_index(df.columns[0])
    return df.loc[:, sample_ids]


def prepare_features(
    pca_components: int = 50,
    variance_threshold: float = 0.01,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    _ensure_dirs()
    _ = run_data_check()

    expression = _read_required_csv(EXPRESSION_FILE)
    methylation = _read_required_csv(METHYLATION_FILE)
    clinical = _read_required_csv(CLINICAL_FILE)

    sample_id_col = _detect_column(SAMPLE_ID_COLUMN_CANDIDATES, list(clinical.columns))
    label_col = _detect_column(LABEL_COLUMN_CANDIDATES, list(clinical.columns))
    age_col = _detect_column(AGE_COLUMN_CANDIDATES, list(clinical.columns))

    exp_ids = set(_infer_sample_ids(expression))
    meth_ids = set(_infer_sample_ids(methylation))
    clin_ids = set(clinical[sample_id_col].astype(str))
    common_ids = sorted(exp_ids.intersection(meth_ids).intersection(clin_ids))
    if not common_ids:
        raise ValueError("No common samples among expression/methylation/clinical")

    expression_mat = _reindex_omics_samples(expression, common_ids).transpose()
    methylation_mat = _reindex_omics_samples(methylation, common_ids).transpose()
    clinical_sub = clinical[clinical[sample_id_col].astype(str).isin(common_ids)].copy()
    clinical_sub = clinical_sub.set_index(sample_id_col).loc[common_ids]

    x_raw = pd.concat([expression_mat, methylation_mat], axis=1)
    y = clinical_sub[label_col].values
    age = clinical_sub[age_col].values

    feature_missing = x_raw.isna().mean(axis=0)
    kept_features = feature_missing[feature_missing <= 0.3].index
    x_filtered = x_raw[kept_features]

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_filtered)

    selector = VarianceThreshold(threshold=variance_threshold)
    x_var = selector.fit_transform(x_imputed)

    n_components = min(pca_components, x_var.shape[1], x_var.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    x_pca = pca.fit_transform(x_var)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_pca)

    x_train, x_test, y_train, y_test, age_train, age_test = train_test_split(
        x_scaled,
        y,
        age,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    np.save(PROCESSED_DIR / "x_scaled.npy", x_scaled)
    np.save(PROCESSED_DIR / "labels.npy", y)
    np.save(PROCESSED_DIR / "ages.npy", age)
    np.save(PROCESSED_DIR / "x_train.npy", x_train)
    np.save(PROCESSED_DIR / "x_test.npy", x_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "age_train.npy", age_train)
    np.save(PROCESSED_DIR / "age_test.npy", age_test)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        age_train=age_train,
        age_test=age_test,
        feature_matrix=x_scaled,
        labels=y,
        ages=age,
    )
