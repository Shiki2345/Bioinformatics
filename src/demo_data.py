from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR


def generate_demo_data(
    n_samples: int = 120,
    n_genes: int = 200,
    n_cpgs: int = 300,
    random_state: int = 42,
) -> None:
    rng = np.random.default_rng(random_state)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample_ids = [f"S{i:03d}" for i in range(1, n_samples + 1)]
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(y)
    age = rng.normal(loc=55 + 8 * y, scale=7, size=n_samples).round(1)

    expr = rng.normal(0, 1, size=(n_genes, n_samples)) + y * 0.35
    meth = rng.normal(0, 1, size=(n_cpgs, n_samples)) + y * 0.20

    expression = pd.DataFrame(expr, columns=sample_ids)
    expression.insert(0, "feature_id", [f"GENE_{i:04d}" for i in range(1, n_genes + 1)])
    methylation = pd.DataFrame(meth, columns=sample_ids)
    methylation.insert(0, "feature_id", [f"CPG_{i:04d}" for i in range(1, n_cpgs + 1)])

    clinical = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": y,
            "age": age,
        }
    )

    expression.to_csv(RAW_DATA_DIR / "expression.csv", index=False, encoding="utf-8-sig")
    methylation.to_csv(RAW_DATA_DIR / "methylation.csv", index=False, encoding="utf-8-sig")
    clinical.to_csv(RAW_DATA_DIR / "clinical.csv", index=False, encoding="utf-8-sig")
