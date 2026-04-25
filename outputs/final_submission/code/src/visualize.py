import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from src.config import PLOTS_DIR
from src.data_prep import PreparedData


def _ensure_plot_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_pca_scatter(data: PreparedData) -> None:
    _ensure_plot_dir()
    pca = PCA(n_components=2, random_state=42)
    x2 = pca.fit_transform(data.feature_matrix)
    plot_df = pd.DataFrame({"pc1": x2[:, 0], "pc2": x2[:, 1], "label": data.labels})
    plt.figure(figsize=(8, 6))
    for label in sorted(plot_df["label"].unique()):
        sub = plot_df[plot_df["label"] == label]
        plt.scatter(sub["pc1"], sub["pc2"], s=20, label=str(label), alpha=0.8)
    plt.legend(title="label")
    plt.title("PCA Scatter by Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_scatter.png", dpi=180)
    plt.close()


def plot_confusion_and_roc(data: PreparedData, best_classifier, y_test_encoded, y_pred_encoded) -> None:
    _ensure_plot_dir()
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    if hasattr(best_classifier, "predict_proba") and len(set(y_test_encoded)) == 2:
        RocCurveDisplay.from_estimator(best_classifier, data.x_test, y_test_encoded)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=180)
        plt.close()


def plot_model_comparison(classification_metrics: pd.DataFrame) -> None:
    _ensure_plot_dir()
    metric_cols = ["accuracy", "f1", "auc", "cv_mean_accuracy"]
    plot_df = classification_metrics[["model"] + metric_cols].melt(
        id_vars="model", value_vars=metric_cols, var_name="metric", value_name="value"
    )
    plt.figure(figsize=(9, 6))
    pivot_df = plot_df.pivot(index="model", columns="metric", values="value")
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.ylim(0, 1.05)
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=180)
    plt.close()


def plot_age_scatter(y_true, y_pred) -> None:
    _ensure_plot_dir()
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=25, alpha=0.8)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("Age Prediction Scatter")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "age_prediction_scatter.png", dpi=180)
    plt.close()
