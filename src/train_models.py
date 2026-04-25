import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.config import METRICS_DIR
from src.data_prep import PreparedData


def _ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate_classifiers(data: PreparedData, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    _ensure_metrics_dir()

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(data.y_train)
    y_test_enc = encoder.transform(data.y_test)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM": SVC(kernel="linear", probability=False, random_state=random_state),
    }

    metrics_rows = []
    fitted_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in models.items():
        cv_scores = cross_val_score(model, data.x_train, y_train_enc, cv=cv, scoring="accuracy")
        model.fit(data.x_train, y_train_enc)
        y_pred = model.predict(data.x_test)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test_enc, y_pred),
            "precision": precision_score(y_test_enc, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test_enc, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test_enc, y_pred, average="weighted", zero_division=0),
            "cv_mean_accuracy": cv_scores.mean(),
        }

        if len(encoder.classes_) == 2:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(data.x_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(data.x_test)
            else:
                y_score = y_pred
            row["auc"] = roc_auc_score(y_test_enc, y_score)
        else:
            row["auc"] = float("nan")

        metrics_rows.append(row)
        fitted_models[name] = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("accuracy", ascending=False)
    metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_df, {"models": fitted_models, "label_encoder": encoder}


def train_and_evaluate_regressor(data: PreparedData, random_state: int = 42) -> tuple[pd.DataFrame, RandomForestRegressor]:
    _ensure_metrics_dir()

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
    )
    model.fit(data.x_train, data.age_train)
    pred = model.predict(data.x_test)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "RandomForestRegressor",
                "mse": mean_squared_error(data.age_test, pred),
                "mae": mean_absolute_error(data.age_test, pred),
                "r2": r2_score(data.age_test, pred),
            }
        ]
    )
    metrics_df.to_csv(METRICS_DIR / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_df, model
