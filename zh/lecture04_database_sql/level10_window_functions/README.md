# Lecture 04 · Level 10 — 窗口函数与分析查询

> 学习不把行折起来、而是在旁边贴上汇总的窗口函数。用 ROW_NUMBER·RANK 排序号和名次，用 SUM OVER 做累计销售额和移动求和，把它和 GROUP BY 的区别刻进肌肉记忆。

**难度** ⭐⭐⭐⭐ / **先修** level05、level06 / **预计时间** 55分钟

## 1. 为什么要学 — 业务视角

做报表做久了，一定会碰到 GROUP BY 搞不定的需求。"标一下这是每位客户的
**第几次**购买"(首购分析)、"在部门**内部**排一下销售额名次"、"在月度
销售额旁边加上**累计**达成额"、"用最近 3 个月的**移动求和**看趋势"。
它们的共同点是: **行要原样保留**，序号、名次、累计这些计算结果得贴在
旁边的列上。而 GROUP BY 会把行折起来，所以做不到。窗口函数
(window function) 就是这种"不折行的聚合"，它既是数据分析岗 SQL 笔试的
常客，也是能把你月末那套"复制公式 + 反复排序"的活儿一句话搞定的工具。

## 2. 打个比方

GROUP BY 是那个**把卡片扔进筐里、只留一行汇总**的员工。200 张卡片被折
成了 6 行城市汇总。

窗口函数不一样。这位员工把卡片**全都摊在桌上**，然后在每一张卡片旁边
贴一张便利贴。便利贴上写什么，靠三条指令来定。

- **看哪个范围**(PARTITION BY — "只看同一位客户的卡片")
- **按什么顺序数**(ORDER BY — "按下单日期")
- **写点什么**(ROW_NUMBER — "这是第几张"，SUM — "到这里为止的合计")

"窗口 (window)"这个名字，你可以这样理解: 站在每张卡片的位置上，
它都有一扇**只能望见自己周围一部分卡片的窗**。有只看得见同一位客户
卡片的窗，也有只看得见到本月为止卡片的窗。把窗外的景象汇总一下，
写到自己的便利贴上 — 这就是 `函数() OVER (窗口定义)`。

## 3. 核心概念

### 3-1. 基本语法 — OVER 就是信号

```sql
SELECT name, category, price,
       AVG(price) OVER (PARTITION BY category) AS cat_avg
FROM   products;
```

一旦加上 `OVER`，AVG 就不再折行。商品 10 行还是 10 行，只是每一行旁边
多了个"自己品类的均价"。用 `price - cat_avg` 就能直接算出"比品类均价贵
多少" — 以前要把 GROUP BY 的结果再 JOIN 回原表，现在一个子句就完事。

### 3-2. ROW_NUMBER — 排序号

```sql
ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY ordered_at) AS nth
```

每位客户都从 1 重新开始数的购买序号。再用 `WHERE nth = 1` (裹在子查询
或 CTE 里) 只挑出"每位客户的首单"，这是实战中最高频的套路。
"每组取最新/最早/最大的 1 条" = ROW_NUMBER + 过滤，当公式背下来。

### 3-3. RANK / DENSE_RANK — 名次与并列处理

三个都在排顺序，区别在于怎么处理并列。

| 值 | ROW_NUMBER | RANK | DENSE_RANK |
|---|---|---|---|
| 100 | 1 | 1 | 1 |
| 90 | 2 | 2 | 2 |
| 90 | 3 | 2 | 2 |
| 80 | 4 | **4** | **3** |

ROW_NUMBER 连并列也硬排成一队 (谁先谁后是随意的)，RANK 在并列第 2 名
之后跳到第 4 名，DENSE_RANK 则接着排第 3 名。颁奖典礼用 RANK，
"前 3 个价格档"用 DENSE_RANK，"无论如何一行一个号"用 ROW_NUMBER。

### 3-4. 累计合计 — 带 ORDER BY 的 SUM OVER

```sql
SUM(revenue) OVER (ORDER BY month) AS cum_revenue
```

ORDER BY 把窗口的范围变成"从开头到当前行"。于是月度销售额旁边就贴上了
年初至今的累计销售额。再加上 PARTITION BY，就能做"各部门各自累计"这种
组内累计。

### 3-5. 移动求和/移动平均 — ROWS BETWEEN

```sql
SUM(revenue) OVER (ORDER BY month
                   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov3
```

"前 2 行 + 当前行" — 就是最近 3 个月的移动求和。它是把上下乱跳的月度
数字的趋势看得更平滑的报表常用手法，语法上是自己直接指定窗口的大小。

### 3-6. 注意执行顺序

窗口函数是在 WHERE、GROUP BY 都结束**之后**才计算的。所以窗口的结果
(nth、rank……) 不能直接写在 WHERE 里，必须先用 CTE 或子查询裹一层，
再在外面过滤。main.py 的 [2] 就是这个套路。

## 4. 动手练习 — main.py

运行:

```bash
python3 main.py
```

- **[1] GROUP BY vs 窗口**: 同一个"各品类均价"用两种方式做 —
  并排看 5 行 vs 10 行的差别 (折起来 vs 保留)。
- **[2] 每位客户的购买序号**: 用 ROW_NUMBER 排序号，再用 CTE 裹住，
  只挑出"每位客户的首单"。
- **[3] 并列处理三兄弟**: 给商品价格 (这份数据里有并列) 并排加上
  ROW_NUMBER/RANK/DENSE_RANK，用真实数据重现 3-3 节的表。
- **[4] 部门内销售额排名**: 先算出员工各自经手的业绩 (CTE)，再在部门
  内部 RANK — 这是"各门店排名"那类报表的骨架。
- **[5] 月度销售额 + 累计销售额**: SUM OVER (ORDER BY month)。
- **[6] 3 个月移动求和**: ROWS BETWEEN 2 PRECEDING AND CURRENT ROW。

每看一段输出，都确认一下"行被折起来了，还是保留着" — 这是本关最核心的
观察重点。

## 5. 自己动手试试

1. **(简单)** 改一改 [3]，按部门 (PARTITION BY dept) 排薪资名次。
   *提示: 在 OVER 的括号里加一句 PARTITION BY dept。*
2. **(中等)** 只挑出每位客户的**最后**一单。
   *提示: 把 [2] 里的 ORDER BY ordered_at DESC 反过来，第 1 号就成了最后一单。*
3. **(挑战)** 给月度销售额加一列"环比增减额"。
   *提示: LAG(revenue) OVER (ORDER BY month) 能取到'上一行的值'。写成 revenue - LAG(revenue) OVER (...)。*

## 6. 常见误区

- **把窗口结果直接写在 WHERE 里**: `WHERE ROW_NUMBER() ...` 会报错。
  用 CTE 裹住、在外面过滤 — 永远是这个两层结构。
- **把 PARTITION BY 和 GROUP BY 搞混**: PARTITION BY 不折行。
  用"行数变少就是 GROUP BY，行数保留就是窗口"来区分。
- **累计合计漏了 ORDER BY**: OVER () 里没有 ORDER BY，窗口就变成全体，
  于是每一行贴上的都是总合计。累计看着不对，先查 ORDER BY。
- **有并列时 ROW_NUMBER 的可复现性**: 并列的时候谁当第 1 号，每次运行
  可能不一样。在 ORDER BY 里再加一个唯一键 (`ORDER BY 金额 DESC,
  订单号`) 把顺序钉死，是做报表的基本礼数。
- **数据库版本太老**: 窗口函数是 SQLite 3.25 (2018) 以后才支持的。
  公司里那种非常老的数据库上可能会直接语法报错。

## 下一关预告

SQL 的功夫已经练成了。最后一关 **level11 — 连接 Python 与数据管道**
会把 SQL 和 Python·pandas 接起来，做一份"每天早上自动出来的汇总报表"，
并学会用参数绑定挡住 SQL 注入，为这门课收尾。
