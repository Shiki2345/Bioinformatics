from pathlib import Path

import pandas as pd

from run_pipeline import main as run_pipeline_main
from src.archive_ingest import export_full_archive_to_standard_csvs, probe_sample_archive
from src.config import METRICS_DIR, OUTPUT_DIR, PLOTS_DIR, REPORT_DIR


def _write_delivery_note() -> Path:
    cls_path = METRICS_DIR / "classification_metrics.csv"
    reg_path = METRICS_DIR / "regression_metrics.csv"
    cls_df = pd.read_csv(cls_path)
    reg_df = pd.read_csv(reg_path)
    best = cls_df.sort_values("accuracy", ascending=False).iloc[0]
    reg = reg_df.iloc[0]

    note = f"""# 最终交付摘要

## 分类任务
- 最优模型：{best['model']}
- Accuracy：{best['accuracy']:.4f}
- F1：{best['f1']:.4f}
- AUC：{best['auc']:.4f}

## 年龄回归任务
- MSE：{reg['mse']:.4f}
- MAE：{reg['mae']:.4f}
- R²：{reg['r2']:.4f}

## 关键产出
- 数据探测：`outputs/sample_archive_probe.json`
- 导出摘要：`outputs/full_archive_export_summary.json`
- 分类指标：`outputs/metrics/classification_metrics.csv`
- 回归指标：`outputs/metrics/regression_metrics.csv`
- 图表目录：`outputs/plots/`
- 自动摘要：`outputs/report/auto_result_summary.md`
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "final_delivery_checklist.md"
    out.write_text(note, encoding="utf-8")
    return out


def main() -> None:
    print("Step 1/4 probing sample archive...")
    sample_info = probe_sample_archive()
    print("sample txt shape:", sample_info["txt_shape"])

    print("Step 2/4 exporting full archive to standard CSVs...")
    full_info = export_full_archive_to_standard_csvs()
    print("full txt shape:", full_info["txt_shape"])

    print("Step 3/4 running dual-task ML pipeline...")
    run_pipeline_main()

    print("Step 4/4 finalizing report assets...")
    delivery_note = _write_delivery_note()
    print("delivery note:", delivery_note)
    print("done. outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
