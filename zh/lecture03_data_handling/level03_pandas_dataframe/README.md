# Lecture 03 · Level 03 — Pandas — Series 与 DataFrame

> 解剖 DataFrame 的结构 — 这个能把一整张 Excel 工作表装进 Python 变量里的工具。

**难度** ⭐⭐ / **先修** level02_numpy_basics / **预计时间** 40分钟

## 1. 为什么要学 — 从业务视角出发

实际数据分析工作的标准工具就是 Pandas。打开销售 CSV、加筛选、按门店聚合、做出报表用的表格 — 这些活儿的开头和结尾都站着 DataFrame。本讲接下来的所有关卡 (读取、清洗、聚合、合并、时间序列、透视) 全都在 DataFrame 上进行。也就是说，这一关是"组装你以后要一直用的工作台"的时间。

不懂 DataFrame 就去看下一关的代码，`df.loc`、`df.dtypes`、`df["revenue"]` 这些写法就跟密码一样。反过来，把结构好好理解一次之后，哪怕见到没见过的 Pandas 功能，你也能推断出"哦，归根结底还是在操作一张行列都贴了名牌的表"。招聘要求里"熟悉 Pandas 者优先"所指的最低门槛，正是这一关的内容。

## 2. 用比喻来理解

想一想 Excel 工作簿。

- **Series** 就是 Excel 的**一列**。比如把"销售额"这一列整条撕下来，但有一个重要的差别: 每个值都跟着一个**行名牌 (index)**。在 Excel 里复制 D 列，"第 3 行的值"这个位置信息就丢了，而 Series 会把名牌一直带在身上。
- **DataFrame** 是把好几个 Series 并排贴在一起的**一张工作表**。所有列共享同一套行名牌 (index)，每一列还挂着列名 (columns)。

另一个比喻是**文件夹**。DataFrame 是一叠贴了标签 (列名) 的文件夹，每份文件 (行) 上都盖了收件编号 (index)。之所以能说"把销售文件夹里收件编号 100 号的那份拿给我看"，全靠这套名牌体系。Excel 是用眼睛看、用鼠标指，Pandas 是用名牌说话。正是这个差别让自动化成为可能。

## 3. 核心概念

### 3.1 Series — 挂了名牌的一维值集合

```python
s = pd.Series([310, 250, 480], index=["朝阳店", "海淀店", "浦东店"])
s["浦东店"]   # 480
```

Series 就是给 NumPy 数组穿上了一层叫 index 的名牌。所以它既保留了 NumPy 那样的 `s.mean()`、`s * 1.1` 之类向量化运算，又能像 `s["浦东店"]` 这样按名字取值。

### 3.2 DataFrame — Series 的集合

DataFrame 的三要素如下。

| 要素 | Excel 里的对应 | 查看方法 |
|---|---|---|
| 值 (values) | 单元格里写的内容 | `df.values` |
| 行名牌 (index) | 行号 1, 2, 3... | `df.index` |
| 列名 (columns) | 标题行 | `df.columns` |

像 `df["revenue"]` 这样按列名取出来，得到的这一列就是 Series。也就是说，请记住这个来回关系: "从 DataFrame 里抽一列就是 Series，把多个 Series 凑起来就是 DataFrame"。

### 3.3 dtype — 每列一个数据类型

每一列各持有一个数据类型 (dtype)。数值列是 `int64` 或 `float64`，文字列在最新版 Pandas 里显示为 `str`，旧版里显示为 `object`。这里有一条实务上很重要的事实: **数值列里一混进缺失值 (NaN)，本来是整数也会变成 float64。** 销售额出来的时候带着小数点、像 `758123.0` 这样，就要养成怀疑"哪儿有空值"的习惯。

### 3.4 打招呼四件套 — head / info / describe / shape

拿到第一次见的数据，永远按这个顺序打招呼。

1. `df.shape` — 几行几列 (掌握规模)
2. `df.head()` — 预览前 5 行 (掌握长相)
3. `df.info()` — 各列的 dtype 和非缺失值个数 (体检)
4. `df.describe()` — 数值列的平均、标准差、最小、最大 (扫一眼分布)

尤其是 `describe()` 的 min 出现负数、或者 max 异常地大，那就是异常值 (outlier) 的信号。

### 3.5 造新列

```python
df["roas"] = df["revenue"] / df["ad_cost"]
```

在 Excel 里往 E 列输入 `=C2/D2` 再往下拖的活儿，一行就完了。因为是向量化运算，就算有 10 万行也是一瞬间。

### 3.6 数一数类别 — value_counts

想看文字列的构成比，就用 `df["store"].value_counts()`。在 Excel 里按门店数重复 COUNTIF 的活儿，这里一行。

## 4. 实战 — main.py

运行方法:

```bash
cd zh/lecture03_data_handling/level03_pandas_dataframe
python3 main.py
```

main.py 把咖啡连锁店 90 天的销售数据 (`hjh_data.sales_table`) 做成 DataFrame，然后一点一点解剖它的结构。金额沿用韩元 (KRW) 计价。

- **[1]** 用 `pd.DataFrame(rows)` 一行把字典列表变成表。本讲前面用列表加循环费劲处理的那张表，这里瞬间就有了 (2,250 行 × 7 列)。
- **[2]~[3]** 输出 `shape`、`index`、`columns`、`dtypes`。用眼睛确认 revenue 为什么是 float64 (因为含缺失)。
- **[4]~[6]** 走一遍 head / info / describe 四件套。describe 的 min 里你会看到负数销售额 (−599,832 韩元) — 这是数据被污染的第一条线索。
- **[7]** 抽出 `df["revenue"]`，确认类型是 Series，并直接算出 `.mean()`、`.max()` 这些摘要。
- **[8]** 造出广告费产出销售额比 (`roas`) 这个派生列，看看靠前的几行 (平均 roas 3.13)。
- **[9]** 用 `value_counts()` 确认门店、星期的构成比。因为是合成数据，门店完全均等 (各 450 行) 也是一个确认点。
- **[10]** 用 `isna().sum()` 数出各列的缺失个数 (revenue 33 个，比例 1.47%)。处理会在 level06 正式展开，这里只做到"知道有多少条"。

输出里值得留意的地方: info() 里 `Non-Null Count` 小于总行数的那一列，就是有缺失的列。

## 5. 动手试试

1. **(基础)** 输出 `df["ad_cost"]` 的平均值和最大值。提示: Series 上也有 `.mean()`、`.max()`。
2. **(应用)** 加一个把销售额换成万元单位的 `revenue_manwon` 列，用 `head()` 确认。提示: 整列除以 10000 就行。
3. **(挑战)** 输出 `df["weekday"].value_counts()`，然后结合数据的生成方式 (90天 × 5门店 × 5品类) 解释为什么各星期的次数并不完全相同。提示: 算一下 `90 % 7` 就知道每个星期的天数其实不一样。实测里周一到周六各 325 行、周日 300 行。和实际输出对照看看。

## 6. 常见错误

- **混淆 `df["revenue"]` 和 `df[["revenue"]]`** — 一层方括号是 Series，两层是只有 1 列的 DataFrame。后面能接的功能是不一样的。
- **只看 describe() 就放心** — describe 只摘要数值列。文字列的错别字 (比如带了空格的 `"朝阳店 "`) 必须另外用 value_counts 确认。
- **不确认 dtype 就开算** — 看着像数字，但 dtype 是 object (文字) 的话，求和就变成字符串拼接，或者直接报错。算之前先看 `df.dtypes` 是必须动作。
- **想把 info() 的结果装进变量** — `info()` 只往屏幕上打印，返回的是 None。需要值的话，用 `df.dtypes`、`df.isna().sum()`。

## 下一关预告

现在数据还是在 Python 代码里造出来的，但实际工作中的数据是以 CSV、Excel、JSON 文件的形式送到你手上的。下一关我们学 `read_csv` 的主要参数，以及那个百分之百会折腾中文办公族的编码问题 (gbk) 该怎么破。
