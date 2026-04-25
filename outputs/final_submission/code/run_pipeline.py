import argparse
import shutil
from pathlib import Path

from sklearn.metrics import accuracy_score

from src.archive_ingest import apply_real_labels_to_existing_clinical
from src.config import CLINICAL_FILE, EXPRESSION_FILE, METHYLATION_FILE, OUTPUT_DIR
from src.data_prep import prepare_features, run_data_check
from src.demo_data import generate_demo_data
from src.reporting import generate_auto_summary, generate_final_report
from src.train_models import train_and_evaluate_classifiers, train_and_evaluate_regressor
from src.visualize import plot_age_scatter, plot_confusion_and_roc, plot_model_comparison, plot_pca_scatter


def _missing_required_files() -> bool:
    required = [EXPRESSION_FILE, METHYLATION_FILE, CLINICAL_FILE]
    return any(not Path(p).exists() for p in required)


def _build_final_submission_folder() -> Path:
    root = Path(__file__).resolve().parent
    target = OUTPUT_DIR / "final_submission"
    if target.exists():
        shutil.rmtree(target)
    (target / "code" / "src").mkdir(parents=True, exist_ok=True)
    (target / "metrics").mkdir(parents=True, exist_ok=True)
    (target / "plots").mkdir(parents=True, exist_ok=True)
    (target / "report").mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        root / "run_pipeline.py",
        root / "run_architecture.py",
        root / "requirements.txt",
        root / "README.md",
        root / "report" / "final_report_template.md",
    ]
    for src in files_to_copy:
        if src.exists():
            shutil.copy2(src, target / "code" / src.name)
    shutil.copytree(root / "src", target / "code" / "src", dirs_exist_ok=True)
    shutil.copytree(OUTPUT_DIR / "metrics", target / "metrics", dirs_exist_ok=True)
    shutil.copytree(OUTPUT_DIR / "plots", target / "plots", dirs_exist_ok=True)
    shutil.copytree(OUTPUT_DIR / "report", target / "report", dirs_exist_ok=True)

    run_guide = target / "RUN_ME.md"
    run_guide.write_text(
        "\n".join(
            [
                "# 运行与复现说明",
                "",
                "1. 安装依赖：`pip install -r requirements.txt`",
                "2. 运行主流程：`python run_pipeline.py`",
                "3. 查看指标：`outputs/metrics/`",
                "4. 查看图表：`outputs/plots/`",
                "5. 查看报告：`outputs/report/final_report.md`",
            ]
        ),
        encoding="utf-8",
    )
    return target


def _write_final_delivery_checklist(best_model_name: str, cls_metrics, reg_metrics, label_update: dict) -> Path:
    report_dir = OUTPUT_DIR / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    best = cls_metrics[cls_metrics["model"] == best_model_name].iloc[0]
    reg = reg_metrics.iloc[0]
    content = f"""# 最终核对清单

## 模型结果
- 最优分类模型：{best_model_name}
- Accuracy：{best['accuracy']:.4f}
- F1：{best['f1']:.4f}
- AUC：{best['auc']:.4f}
- 回归 MSE：{reg['mse']:.4f}
- 回归 MAE：{reg['mae']:.4f}
- 回归 R²：{reg['r2']:.4f}

## 标签接入
- 标签更新状态：{label_update.get('updated', False)}
- 标签来源：{label_update.get('label_source', 'fallback')}
- 标签分布：{label_update.get('label_distribution', 'unknown')}

## 交付文件确认
- 代码：`outputs/final_submission/code/`
- 指标：`outputs/final_submission/metrics/`
- 图表：`outputs/final_submission/plots/`
- 报告：`outputs/final_submission/report/final_report.md`
- 运行说明：`outputs/final_submission/RUN_ME.md`
"""
    out = report_dir / "final_delivery_checklist.md"
    out.write_text(content, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-demo-data",
        action="store_true",
        help="Generate synthetic demo data if real CSV files are not ready.",
    )
    args = parser.parse_args()

    if args.use_demo_data or _missing_required_files():
        print("Generating demo data in data/raw/ ...")
        generate_demo_data()

    label_update = apply_real_labels_to_existing_clinical()
    if label_update.get("updated"):
        print("Clinical labels updated from metadata:", label_update["label_source"])
        print("Updated label distribution:", label_update["label_distribution"])
    else:
        print("Clinical labels not updated:", label_update.get("reason", "unknown reason"))

    summary = run_data_check()
    print("Data check done:", summary["sample_counts"])

    data = prepare_features()
    cls_metrics, cls_bundle = train_and_evaluate_classifiers(data)
    reg_metrics, reg_model = train_and_evaluate_regressor(data)

    best_model_name = cls_metrics.iloc[0]["model"]
    best_model = cls_bundle["models"][best_model_name]
    encoder = cls_bundle["label_encoder"]
    y_test_encoded = encoder.transform(data.y_test)
    y_pred_encoded = best_model.predict(data.x_test)
    age_pred = reg_model.predict(data.x_test)

    plot_pca_scatter(data)
    plot_confusion_and_roc(data, best_model, y_test_encoded, y_pred_encoded)
    plot_model_comparison(cls_metrics)
    plot_age_scatter(data.age_test, age_pred)

    report_path = generate_auto_summary(
        classification_metrics=cls_metrics,
        regression_metrics=reg_metrics,
        data=data,
        best_model_name=best_model_name,
        y_test_encoded=y_test_encoded,
        y_pred_encoded=y_pred_encoded,
    )
    template_path, final_report_path = generate_final_report(
        classification_metrics=cls_metrics,
        regression_metrics=reg_metrics,
        data=data,
        data_check_summary=summary,
        best_model_name=best_model_name,
        y_test_encoded=y_test_encoded,
        y_pred_encoded=y_pred_encoded,
    )
    final_submission_path = _build_final_submission_folder()
    checklist_path = _write_final_delivery_checklist(best_model_name, cls_metrics, reg_metrics, label_update)
    shutil.copy2(checklist_path, final_submission_path / "report" / checklist_path.name)

    print("Best model:", best_model_name)
    print("Test accuracy:", round(accuracy_score(y_test_encoded, y_pred_encoded), 4))
    print("Summary report:", report_path)
    print("Final report template updated:", template_path)
    print("Final report:", final_report_path)
    print("Final submission folder:", final_submission_path)
    print("Final checklist:", checklist_path)


if __name__ == "__main__":
    main()
