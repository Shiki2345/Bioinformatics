# 基于多组学数据的癌症分类研究报告（最终版）

## 1. 标题
基于多组学数据的癌症分类与年龄预测研究

## 2. 摘要
本研究整合基因表达与甲基化两类组学特征，构建分类与回归双任务流程。分类任务以临床真实标签为监督信号，比较 RandomForest、LogisticRegression 和 SVM；回归任务以年龄为目标，使用 RandomForestRegressor。流程包含缺失值处理、方差过滤、PCA 降维、标准化及分层划分。结果显示分类最优模型为 RandomForest，测试集准确率 0.9684、F1 0.9655、AUC 0.9923，5 折交叉验证准确率均值 0.9658；年龄预测 MSE 227.4474、MAE 12.0021、R² 0.5965。该流程可复用并支持后续扩展调参与特征解释分析。

## 3. 背景
- 多组学数据可从不同分子层面反映癌症相关生物学差异。
- 统一流程对比多种模型有助于获得稳定基线性能。
- 双任务设置同时覆盖分类识别与连续变量预测。

## 4. 方法
### 4.1 数据来源
- `data/raw/expression.csv`
- `data/raw/methylation.csv`
- `data/raw/clinical.csv`
- 标签列自动探测结果：`label`

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
- 最优模型：RandomForest
- 测试集准确率：0.9684
- 测试集 Precision：0.9674
- 测试集 Recall：0.9684
- 测试集 F1：0.9655
- 测试集 AUC：0.9923
- 5 折 CV 准确率均值：0.9658
- 直接重算测试准确率：0.9684

### 5.2 回归结果
- 模型：RandomForestRegressor
- MSE：227.4474
- MAE：12.0021
- R²：0.5965

### 5.3 数据规模与标签分布
- 总样本数：8374
- 训练集样本数：6699
- 测试集样本数：1675
- 降维后特征数：50
- 标签分布：{0: 7745, 1: 629}

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
