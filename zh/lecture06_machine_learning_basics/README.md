# Lecture 06 — 机器学习入门

> 不再把规则一条条硬编码，而是让机器从数据中学出规律 — 本课从头到尾讲透这件事。
> 我们会亲手跑一遍回归、分类、聚类的代表模型，并一直讲到"该如何信任模型"(评估、过拟合、解释)。

## 这门课你将学到

- 机器学习 (Machine Learning) 与传统编程有什么区别，什么时候该用、什么时候不该用
- 如何把业务诉求 ("想降低客户流失") 翻译成一份 ML 问题说明书
- 线性回归、逻辑回归、决策树、随机森林、梯度提升的原理与用法
- 训练/验证/测试划分、评估指标、过拟合与正则化 — "正确地信任模型"的技术
- 用聚类和降维，在没有标准答案的数据中发现结构
- 用特征重要性和单条预测解释，把模型讲给非技术同事听

## 先修课程

- **lecture02 — Python 编程基础** (要能读懂函数、列表、字典)
- **lecture03 — 数据处理** (NumPy 数组、Pandas DataFrame、缺失值处理)

数学只需要高中水平的直觉 (直线的斜率、概率) 就足够了。我们靠比喻和实验来讲，而不是靠公式。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_rules_vs_learning/README.md) | 机器学习是什么 — 规则 vs 学习 | ⭐ |
| [level01](level01_learning_paradigms/README.md) | 监督、无监督与强化学习 | ⭐ |
| [level02](level02_framing_business_problems/README.md) | 把业务问题转化为 ML 问题 | ⭐⭐ |
| [level03](level03_linear_regression/README.md) | 线性回归 | ⭐⭐ |
| [level04](level04_train_valid_test/README.md) | 训练集、验证集与测试集的划分 | ⭐⭐ |
| [level05](level05_logistic_regression/README.md) | 逻辑回归 — 分类的起点 | ⭐⭐⭐ |
| [level06](level06_evaluation_metrics/README.md) | 评估指标 — 准确率的陷阱 | ⭐⭐⭐ |
| [level07](level07_decision_trees/README.md) | 决策树 | ⭐⭐⭐ |
| [level08](level08_random_forest_ensembles/README.md) | 随机森林与集成学习 | ⭐⭐⭐ |
| [level09](level09_overfitting_regularization/README.md) | 过拟合、正则化与超参数 | ⭐⭐⭐⭐ |
| [level10](level10_clustering_dimreduction/README.md) | 聚类与降维 | ⭐⭐⭐⭐ |
| [level11](level11_boosting_interpretation/README.md) | 梯度提升与模型解释 | ⭐⭐⭐⭐⭐ |

## 快速路线 (时间紧就只学这 5 关)

1. **level00** — 规则 vs 学习: 用身体理解机器学习到底是什么
2. **level03** — 线性回归: 你的第一个预测模型
3. **level04** — 数据划分: 不懂"自己出题自己考"的陷阱，一切结果都是谎言
4. **level05** — 逻辑回归: 实战分类问题的出发点
5. **level06** — 评估指标: 不被"准确率 98.5%"忽悠的方法

## 这门课在实际工作中的应用场景

- **订阅/会员运营**: 提前找出下个月可能退订的客户，集中投放优惠券和挽留回访 (level05, 07, 11)
- **营销预算会议**: 对"广告费多投 1000 万韩元，销售额能涨多少?"给出有依据的数字 (level03)
- **异常交易与欺诈监控**: 在欺诈比例仅 1.5% 的数据上，学会用召回率、精确率说话，而不是准确率 (level06)
- **客户细分**: 没有标准答案，仅凭购买行为把客户分组，再按组制定策略 (level10)
- **与外部供应商、数据团队协作**: 面对"我们的 AUC 是 0.85"这样的汇报，能提出自己验证的问题 (level04, 06, 09)
- **向管理层汇报**: 用特征重要性和单条预测解释，讲清"模型为什么认为这位客户有风险" (level11)

## 学习建议

- 每一关都设计成边运行 `main.py` 边阅读。只看讲义不动手，等于只学了一半。
- 运行方式: 用仓库根目录的虚拟环境执行 `python3 main.py` (环境配置详见根目录的 `SETUP.md`)
- 所有数据都在仓库内部合成生成，不需要联网。
