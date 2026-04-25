from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import REPORT_DIR
from src.data_prep import PreparedData


def generate_auto_summary(
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    data: PreparedData,
    best_model_name: str,
    y_test_encoded,
    y_pred_encoded,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    acc = accuracy_score(y_test_encoded, y_pred_encoded)
    best_row = classification_metrics[classification_metrics["model"] == best_model_name].iloc[0]
    reg_row = regression_metrics.iloc[0]

    content = f"""# 自动结果摘要

## 分类任务结果
- 最优模型：{best_model_name}
- 测试集准确率：{best_row['accuracy']:.4f}
- F1：{best_row['f1']:.4f}
- AUC：{best_row['auc']:.4f}
- 5折交叉验证均值：{best_row['cv_mean_accuracy']:.4f}

## 回归任务结果
- 模型：RandomForestRegressor
- MSE：{reg_row['mse']:.4f}
- MAE：{reg_row['mae']:.4f}
- R²：{reg_row['r2']:.4f}

## 数据规模
- 总样本数：{len(data.labels)}
- 训练集样本数：{len(data.y_train)}
- 测试集样本数：{len(data.y_test)}
- 特征矩阵维数（降维后）：{data.feature_matrix.shape[1]}

## 建议
- 报告中优先对最优分类模型进行误差来源分析。
- 如果AUC偏低，可增加特征筛选或调参。
- 若年龄预测R²较低，建议尝试XGBoost/ElasticNet作对比。
"""

    output_path = REPORT_DIR / "auto_result_summary.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def generate_final_report(
    classification_metrics: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    data: PreparedData,
    data_check_summary: dict,
    best_model_name: str,
    y_test_encoded,
    y_pred_encoded,
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    root_dir = Path(__file__).resolve().parent.parent
    report_template_path = root_dir / "report" / "final_report_template.md"
    final_report_path = REPORT_DIR / "final_report.md"

    best_row = classification_metrics[classification_metrics["model"] == best_model_name].iloc[0]
    reg_row = regression_metrics.iloc[0]
    label_info = data_check_summary.get("clinical_columns", {})
    label_distribution = data_check_summary.get("label_distribution", {})
    sample_counts = data_check_summary.get("sample_counts", {})
    test_acc = accuracy_score(y_test_encoded, y_pred_encoded)

    report_text = f"""# 基于多组学数据的癌症分类研究报告（最终版）

## 1. 标题
基于多组学数据的癌症分类与年龄预测研究

## 2. 摘要
本研究整合基因表达与甲基化两类组学特征，构建分类与回归双任务流程。分类任务以临床真实标签为监督信号，比较 RandomForest、LogisticRegression 和 SVM；回归任务以年龄为目标，使用 RandomForestRegressor。流程包含缺失值处理、方差过滤、PCA 降维、标准化及分层划分。结果显示分类最优模型为 {best_model_name}，测试集准确率 {best_row['accuracy']:.4f}、F1 {best_row['f1']:.4f}、AUC {best_row['auc']:.4f}，5 折交叉验证准确率均值 {best_row['cv_mean_accuracy']:.4f}；年龄预测 MSE {reg_row['mse']:.4f}、MAE {reg_row['mae']:.4f}、R² {reg_row['r2']:.4f}。该流程可复用并支持后续扩展调参与特征解释分析。

## 3. 背景
- 多组学数据可从不同分子层面反映癌症相关生物学差异。
- 统一流程对比多种模型有助于获得稳定基线性能。
- 双任务设置同时覆盖分类识别与连续变量预测。

## 4. 方法
### 4.1 数据来源
- `data/raw/expression.csv`
- `data/raw/methylation.csv`
- `data/raw/clinical.csv`
- 标签列自动探测结果：`{label_info.get('label', 'unknown')}`

### 4.2 预处理流程
- 缺失值中位数填补；
- 特征缺失率过滤（<=30%）与方差过滤；
- PCA 降维（最多 50 维）；
- 标准化 + 训练/测试集划分（8:2，分层）。

### 4.3 模型与参数
- 分类：RandomForest / LogisticRegression / SVM（线性核）
- 回归：RandomForestRegressor
- 评估：5 折交叉验证 + 独立测试集

## 5. 结果
### 5.1 分类结果
- 最优模型：{best_model_name}
- 测试集准确率：{best_row['accuracy']:.4f}
- 测试集 Precision：{best_row['precision']:.4f}
- 测试集 Recall：{best_row['recall']:.4f}
- 测试集 F1：{best_row['f1']:.4f}
- 测试集 AUC：{best_row['auc']:.4f}
- 5 折 CV 准确率均值：{best_row['cv_mean_accuracy']:.4f}
- 直接重算测试准确率：{test_acc:.4f}

### 5.2 回归结果
- 模型：RandomForestRegressor
- MSE：{reg_row['mse']:.4f}
- MAE：{reg_row['mae']:.4f}
- R²：{reg_row['r2']:.4f}

### 5.3 数据规模与标签分布
- 总样本数：{sample_counts.get('common_samples', len(data.labels))}
- 训练集样本数：{len(data.y_train)}
- 测试集样本数：{len(data.y_test)}
- 降维后特征数：{data.feature_matrix.shape[1]}
- 标签分布：{label_distribution}

### 5.4 图表
- PCA 散点图：`outputs/plots/pca_scatter.png`
- 混淆矩阵：`outputs/plots/confusion_matrix.png`
- ROC 曲线：`outputs/plots/roc_curve.png`
- 模型对比图：`outputs/plots/model_comparison.png`
- 年龄预测散点图：`outputs/plots/age_prediction_scatter.png`

## 6. 讨论
- 分类性能可作为后续特征解释与参数优化基线。
- 当标签类别不平衡时，建议补充 PR 曲线与分层采样策略。
- 回归任务仍有提升空间，可扩展到梯度提升类模型进行对比。

## 7. 结论
- 已完成多组学双任务流程的端到端运行。
- 已接入临床真实标签来源并完成指标与图表自动更新。
- 产物满足提交所需的代码、指标、图表与报告要求。

## 8. 附录：运行与复现
- 安装依赖：`pip install -r requirements.txt`
- 执行主流程：`python run_pipeline.py`
- 指标目录：`outputs/metrics/`
- 图表目录：`outputs/plots/`
- 自动摘要：`outputs/report/auto_result_summary.md`
- 最终报告：`outputs/report/final_report.md`
"""

    report_template_path.write_text(report_text, encoding="utf-8")
    final_report_path.write_text(report_text, encoding="utf-8")
    return report_template_path, final_report_path
