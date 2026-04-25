# 基于多组学数据的癌症分类

本项目用于完成以下两项任务：

- 二分类：癌症样本 vs 正常样本
- 回归：基于同一特征矩阵预测年龄

## 目录结构

- `data/raw/`：放置原始数据（`expression.csv`、`methylation.csv`、`clinical.csv`）
- `team/member_info.csv`：小组成员信息登记
- `src/`：核心代码（数据检查、预处理、训练评估、可视化、报告生成）
- `outputs/`：运行输出（指标、图表、中间数据）
- `report/final_report_template.md`：最终报告模板
- `run_pipeline.py`：主入口脚本

## 快速开始

1. 把原始数据放到 `data/raw/`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行全流程：`python run_pipeline.py`

## 产出文件

- `outputs/data_check_summary.json`
- `outputs/metrics/classification_metrics.csv`
- `outputs/metrics/regression_metrics.csv`
- `outputs/plots/*.png`
- `outputs/report/auto_result_summary.md`
