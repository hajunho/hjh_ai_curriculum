# Lecture 05 — 统计与数据可视化

这门课教你三件事：把数字"概括"起来，用"图"揭示概括所隐藏的东西，最后判断
"这个差异是真实的，还是碰巧的"。如果你在 Excel 里按过 AVERAGE，
那你已经完成了一半。剩下的一半——什么时候不能相信那个平均数——
就是这门课要教的。

## 学什么

- 代表值 (平均数·中位数·方差) 概括数据的方式与其中的陷阱
- 用分布、直方图和 Matplotlib 把数据变成图的方法
- 一双能分辨"骗人的图表"和"诚实的图表"的眼睛
- 相关与因果、概率、正态分布与中心极限定理
- 用样本推断总体的方法：置信区间、假设检验、p 值
- 实务中的重头戏：A/B 测试的设计与解读，用贝叶斯思维汇报不确定性

## 先修课程

- **lecture02 — Python 编程基础** (必须)
- **lecture03 — 数据处理 (NumPy · Pandas)** (必须：会用到数组和数据框)
- 学完 **lecture04** 会更轻松。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_summarizing_reality/README.md) | 用数字概括现实意味着什么 | ⭐ |
| [level01](level01_mean_median_variance/README.md) | 平均数、中位数与方差 | ⭐ |
| [level02](level02_distributions_histograms/README.md) | 分布与直方图 | ⭐⭐ |
| [level03](level03_matplotlib_basics/README.md) | Matplotlib 基础图表 | ⭐⭐ |
| [level04](level04_good_vs_bad_charts/README.md) | 好图表 vs 坏图表 | ⭐⭐ |
| [level05](level05_correlation_causation/README.md) | 相关关系与因果关系 | ⭐⭐⭐ |
| [level06](level06_probability_basics/README.md) | 概率的基础 | ⭐⭐⭐ |
| [level07](level07_normal_clt/README.md) | 正态分布与中心极限定理 | ⭐⭐⭐ |
| [level08](level08_sampling_confidence/README.md) | 样本与置信区间 | ⭐⭐⭐ |
| [level09](level09_hypothesis_pvalue/README.md) | 假设检验与 p 值 | ⭐⭐⭐⭐ |
| [level10](level10_ab_testing/README.md) | A/B 测试的设计与解读 | ⭐⭐⭐⭐ |
| [level11](level11_bayesian_thinking/README.md) | 贝叶斯思维与不确定性沟通 | ⭐⭐⭐⭐⭐ |

## 快速通道 (时间紧就只看这 5 个)

1. **level01** — 平均数、中位数与方差：一切数字汇报的基础
2. **level04** — 好图表 vs 坏图表：不被骗，也不骗人
3. **level05** — 相关与因果：会议室里最常出错的地方
4. **level09** — 假设检验与 p 值：回答"这个差异是真的吗？"
5. **level10** — A/B 测试：数据驱动型组织做决策的标准工具

## 这门课在实务中的应用场景

- **月度汇报**："平均销售额上涨 12%" 这句话背后是否藏着一个极端值，
  用中位数和分布来核实 (level01~02)。
- **给管理层的汇报材料**：避免用截断坐标轴的柱状图夸大业绩，
  也能一眼看穿别人做的歪曲图表 (level03~04)。
- **营销效果分析**："加大了广告投入，销售额就上去了" 究竟是因果，
  还是旺季这个混杂变量在作怪，仔细掰扯清楚 (level05)。
- **新产品·界面改版决策**：设计 A/B 测试，并理解为什么中途偷看结果
  (peeking) 是危险的 (level09~10)。
- **汇报不确定性**：不说"行/不行"，而说"B 方案更优的概率是 92%"，
  用管理层能据以决策的语言说话 (level08, 11)。

## 运行方法

在每个关卡文件夹里用仓库的虚拟环境运行。

```bash
cd lecture05_statistics_visualization/level00_summarizing_reality
python3 main.py
```

画图的关卡在运行后会在该文件夹的 `outputs/` 里生成 PNG 文件，
路径会打印在屏幕上。所有数据都由 `common/hjh_data.py` 或代码本身生成，
不需要联网。
