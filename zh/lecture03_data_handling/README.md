# Lecture 03 — 数据处理 (NumPy · Pandas)

> 把你以前在 Excel 里做的所有表格活儿搬进代码，并且做成 Excel 根本做不到的事。

## 这一讲你会学到

- 数据为什么总是被整理成"行与列的表"，把这张表交给代码之后到底有什么不同
- 用 NumPy 数组 (ndarray) 一次性算完几十万个数字的"向量化"思维方式
- 用 Pandas DataFrame 走完读取 → 清洗 → 筛选 → 聚合 → 合并 → 时间序列 → 透视的完整实战分析流水线
- 数据变大之后怎么省内存、怎么分块处理，以及什么时候该换成数据库 / Spark

所有实战都使用一家虚构咖啡连锁店 (朝阳店·海淀店·浦东店·天河店·南山店 五家门店) 的销售数据。数据由 `common/hjh_data.py` 用代码当场生成，所以完全不需要联网，而且像真实业务一样，里面已经埋好了缺失值和负数污染。

## 先修课程

- **lecture01 — 计算机与编程基础**、**lecture02 — Python 基础** (变量、列表、字典、循环、函数、文件读写)
- 如果你在 Excel 里用过 SUM / 筛选 / 数据透视表，本讲的比喻会更容易共鸣。

## 关卡构成

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_what_is_data/README.md) | 数据到底是什么 — 表格的结构 | ⭐ |
| [level01](level01_excel_to_python/README.md) | 从 Excel 到 Python | ⭐ |
| [level02](level02_numpy_basics/README.md) | NumPy 数组基础 | ⭐⭐ |
| [level03](level03_pandas_dataframe/README.md) | Pandas — Series 与 DataFrame | ⭐⭐ |
| [level04](level04_loading_data/README.md) | 读取数据 (CSV · Excel · JSON) | ⭐⭐ |
| [level05](level05_filter_sort_select/README.md) | 筛选、排序与选取 | ⭐⭐ |
| [level06](level06_missing_outliers/README.md) | 缺失值与异常值处理 | ⭐⭐⭐ |
| [level07](level07_groupby_aggregation/README.md) | 分组与聚合 (groupby) | ⭐⭐⭐ |
| [level08](level08_merge_join/README.md) | 合并与连接 (merge · concat) | ⭐⭐⭐ |
| [level09](level09_time_series/README.md) | 时间序列数据处理 | ⭐⭐⭐⭐ |
| [level10](level10_pivot_reshape_window/README.md) | 透视表、重塑与窗口运算 | ⭐⭐⭐⭐ |
| [level11](level11_large_data_strategies/README.md) | 大规模数据处理策略 | ⭐⭐⭐⭐⭐ |

## 快速通道 (时间紧就只看这 5 关)

1. **level03** — 不懂 DataFrame 的结构，后面什么都干不了。
2. **level05** — 实际工作中 80% 的问题就是"挑出来、排个序"。
3. **level06** — 现实中的数据一定是脏的。
4. **level07** — "各门店销售额是多少?"这类问题的答案就是 groupby。
5. **level08** — 没有哪次分析是一张表就能搞定的。合并才是实战。

走完快速通道后，如果工作里有时间序列相关的活儿就补 level09，如果报表类的活儿多就补 level10。

## 这一讲在实际工作中的用场

- **月度业绩报告**: 打开 30 家门店的销售 CSV，5 分钟内拉出各门店、各品类的合计和环比增长率。(level04·07·09)
- **数据质检**: 交易日志里混进了空值和负数金额时，诊断出有多少条被污染，并定下处理方针。(level06)
- **目标管理**: 把销售表、门店信息表和目标表合到一起，做出各门店目标达成率排行。(level08·10)
- **自动化**: 把每周重复的 Excel 手工活换成一个脚本，让结果每次都能一模一样地复现。(level01·11)

## 运行方法

在每个关卡文件夹里按下面的方式运行。图片、CSV 之类的产物会生成在各关卡的 `outputs/` 文件夹里。

```bash
cd zh/lecture03_data_handling/level03_pandas_dataframe
python3 main.py
```
