# Lecture 05 · Level 03 — Matplotlib 基础图表

> 图表不是"画"，而是"文章"。figure 是纸张，axes 是段落。
**难度** ⭐⭐ / **先修** level02 / **预计时间** 50分钟

## 1. 为什么要学 — 业务视角

用 Excel 图表也能画图，那为什么要用代码画？
第一，**重复**。每周更新的 30 张门店销售图，用 Excel 做要半天，
用代码做只需运行一次。第二，**复现**。"上个月报告里那张图，
换个条件再来一张"——代码可以立刻交差。第三，**扩展**。
之后要学的机器学习、A/B 测试的结果，最终都要靠 Matplotlib 来查看。

Matplotlib 是 Python 可视化的标准库。seaborn、pandas 的 `.plot()` 等
大多数可视化工具内部用的都是 Matplotlib。所以把 Matplotlib 的结构
(figure/axes) 认真理解一次，其他工具的文档也能读得顺畅。

## 2. 用类比来理解

Matplotlib 的结构和**写报告**是一样的。

- **figure** 是一张 A4 纸。先定好大小 (figsize)，写完就存成文件 (savefig)。
- **axes** 是纸上的段落 (一格图)。一张纸可以只放一个段落
  (`plt.subplots()`)，也可以 2×2 放四个段落 (`plt.subplots(2, 2)`)。
- **plot/bar/scatter** 是段落里的句子。画线、立柱、打点。
- **标题、轴名、图例**是段落的小标题和脚注。少了这些的图表，
  就像没有主语的句子，除了作者本人谁也读不懂。

"取纸 (figure) → 分段 (axes) → 写句子 (plot) → 加小标题 (title/label)
→ 提交 (savefig)"——这五步是所有 Matplotlib 代码的骨架。

## 3. 核心概念

### 3.1 基本骨架代码

```python
import matplotlib
matplotlib.use("Agg")            # 不弹窗口、只存文件的模式
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))   # 纸张 + 1 格段落
ax.plot(x, y)                              # 画线
ax.set_title("Monthly Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (million KRW)")
fig.savefig("outputs/monthly.png", dpi=120)
plt.close(fig)                             # 收起纸张 (清理内存)
```

也有像 `plt.plot(...)` 这样不经过 ax 直接画的方式，但这门课永远以
`fig, ax = plt.subplots()` 开头，采用**面向对象风格**。
图一多起来，这种方式明显更不容易搞混。

### 3.2 图表三兄弟 — 什么时候用哪个

| 种类 | 函数 | 使用场景 | 例子 |
|---|---|---|---|
| 折线图 | `ax.plot` | **随时间的变化** | 月度销售走势 |
| 柱状图 | `ax.bar` | **类别之间的比较** | 各门店销售额比较 |
| 散点图 | `ax.scatter` | **两个数值的关系** | 广告费 vs 销售额 |

用反了立刻就别扭。门店比较用折线图，会产生"从朝阳店渐变成海淀店"
的错觉；时间走势用柱状图，趋势就看不清。请记住六个字：
"时间线、类别柱、关系点"。

### 3.3 中文字体问题

Matplotlib 的默认字体里没有中文字形，中文标题变成 □□□ (豆腐块)
是常有的事。解决办法是指定操作系统里已安装的中文字体。

```python
from matplotlib import font_manager
plt.rcParams["font.family"] = "PingFang SC"   # macOS
# Windows: "Microsoft YaHei" / Linux: "Noto Sans CJK SC" (需安装)
plt.rcParams["axes.unicode_minus"] = False    # 防止负号显示成方块
```

实战代码里的 `set_chinese_font()` 会按顺序查找三大操作系统的代表字体，
找到哪个用哪个，在任何环境都能运行。这个函数可以直接拷去你自己的项目。
不过本课程为了在任何机器上都不乱码，图内文字统一用英文，
中文只出现在终端输出里。

### 3.4 保存规则 — Agg 与 outputs/

服务器和自动化脚本是没有显示器的。`matplotlib.use("Agg")` 声明
"别往屏幕上画，只生成图片文件"，而且必须在 **import pyplot 之前**调用。
保存用 `fig.savefig(路径, dpi=120)`；这门课永远存到关卡文件夹的
`outputs/` 下，然后把路径 print 出来。用
`os.makedirs(OUT_DIR, exist_ok=True)` 实现"没有就建、有就跳过"
也是标准套路。

### 3.5 颜色就是信息

颜色不是用来"好看"的，是用来"说事"的。把所有柱子涂成彩虹色，
读者会拼命给每种颜色找含义，最后筋疲力尽。基本原则是：全部用一种
沉稳的颜色，只把想强调的那根柱子换个颜色——仅此一条，图表的信息
传达力就大幅上升。实战 [3] 的柱状图就是这条原则的忠实实现。

## 4. 实战 — main.py

```bash
python3 main.py
```

用 `hjh_data.sales_table()` 的销售额画三种图。

- [1] 清洗数据，按月、按门店聚合 (复习 lecture03)。
- [2] `outputs/line_monthly.png`：月度总销售额**折线图**。加上标记点和
  数值标签，让趋势更好读。
- [3] `outputs/bar_stores.png`：各门店总销售额**柱状图**。按从大到小
  排序，只给第一名门店换色强调。
- [4] `outputs/scatter_ad.png`：每日广告费对销售额的**散点图**。
  两个值的关系就体现在点云的倾斜上 (这是 level05 的预告片)。

每个函数都严格遵循 3.1 的五步 (纸张→段落→句子→小标题→保存)，
请挑一个函数，把每一行和骨架一一对应起来。

## 5. 亲手试试

1. **(简单)** 改掉折线图的颜色 (`color="#d1495b"`)，把线型换成虚线
   (`ls="--"`) 再保存。
2. **(中等)** 把 [3] 的柱状图改成水平柱 (`ax.barh`)。体会门店名很长时
   水平柱为什么更好读。
   (提示：原来放进 x 和 y 的值要互换。)
3. **(挑战)** 用 `plt.subplots(1, 2, figsize=(12, 4))` 在一张纸上并排放
   折线图和柱状图，存成 `outputs/dashboard.png`。
   (提示：`fig, (ax1, ax2) = plt.subplots(1, 2, ...)` 拿到两个段落，
   各画各的。)

## 6. 常见错误

- **在 import pyplot 之后才调用 `matplotlib.use("Agg")`**：可能报警告
  或不生效。永远放在 import matplotlib 之后的第一行。
- **漏掉 `plt.close(fig)`**：在画几十张图的循环里，内存会不断堆积。
  养成"存完就关"的习惯。
- **省略轴名和单位**："销售额"和"销售额 (万韩元)"，图的解读能差一百倍。
  单位务必写进轴名。
- **类别比较用折线图**：门店之间没有"中间值"。用线连起来，
  就暗示了不存在的连续性。
- **不设 CJK 字体就分发**：在你电脑上正常，到同事电脑上就变豆腐块。
  把字体查找代码包含进脚本里。

## 下一关预告

学会了怎么画，接下来该学"怎么画错"了。亲手制作截断坐标轴、
套上 3D 的骗人图表，再和同一份数据的诚实版本并排比较。
