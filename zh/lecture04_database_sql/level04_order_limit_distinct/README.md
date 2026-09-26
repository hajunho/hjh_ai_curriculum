# Lecture 04 · Level 04 — 排序、去重与前 N 名

> 用 ORDER BY 给结果排队，用 LIMIT 只截取前 N 个，用 DISTINCT 清掉重复。一句 SQL 做出"最贵商品 Top5"这样的排行榜。

**难度** ⭐⭐ / **先修** level03 / **预计时间** 35分钟

## 1. 为什么要学 — 业务视角

报表里的表格大多是**排好队的表格**: "销售额前 10 的门店"、
"最近 20 笔订单"、"最便宜的 3 份报价"。而会议上常有人问的
"我们的客户都分布在哪些城市？"，问的其实是**去掉重复后的清单**。
只要有 ORDER BY (排序)、LIMIT (前 N)、DISTINCT (去重) 这三件工具，
这类需求大部分一句话就能搞定。尤其是"Top N"套路，
是实战 SQL 里体感出场率最高的形态之一，值得练到完全长在手上。

## 2. 打个比方

想象柜台职员的操作顺序。他先按申请单把账本里符合条件的行挑进篮子
(FROM + WHERE)，最后做三道收尾工序。

- **ORDER BY = 排队**: 把篮子里的行按要求的标准 (价格序、日期序……)
  排好放上托盘。有并列的就按第二标准再排 — 就像运动会上
  "按个子排，个子一样再按姓名排"。
- **LIMIT = 从上面截几张**: 从排好队的托盘顶端只递出 N 张。
  不排序就截，得到的只是"随便 5 张"，所以 **Top N 必须和 ORDER BY 成套**。
- **DISTINCT = 扔掉重复的卡片**: 内容相同的卡片有好几张时只留一张。
  问"客户住在哪些城市"这类**种类**问题时用它。

重点是: 这些工序全都是**对结果篮子的操作**，原始账本一点都不会变。

## 3. 核心概念

### 3-1. ORDER BY — 升序与降序

```sql
SELECT name, price
FROM   products
ORDER BY price DESC;   -- 从贵到便宜 (DESC = descending, 降序)
```

默认是升序 (ASC，可省略)。日期列按 DESC 排就是"最新在前"。
文字列按字典序排列 (中文按字符编码顺序，不是拼音顺序)。

### 3-2. 多重排序标准

```sql
ORDER BY city ASC, grade DESC
```

先按城市排，同一城市内再按等级倒序。逗号的先后就是优先级。
别名和计算式也能当排序标准: `ORDER BY price - cost DESC` (毛利大的在前)。

### 3-3. LIMIT 与 OFFSET

`LIMIT 5` 取前 5 行，`LIMIT 5 OFFSET 5` 取第 6~10 行。
网站上的"第 2 页"就是 OFFSET。注意: **LIMIT 减少的不是计算量，
而是展示量** — 排序是对全体做完之后才截取顶端的。

### 3-4. DISTINCT — 问"有哪些种类"

```sql
SELECT DISTINCT city FROM customers;
```

从 200 位客户的行里只取城市值会满是重复，DISTINCT 让相同的值只留一次。
写两个列 (`(city, grade)`) 就是对**组合**去重 —
比如"每个城市实际存在的等级组合"。要连"有几种"一起数的
`COUNT(DISTINCT city)`，下一关 (聚合) 就会遇到。

### 3-5. 在语句里的位置 (语序规则)

SQL 子句的顺序是固定的。

```
SELECT [DISTINCT] 列清单
FROM   表
WHERE  条件
ORDER BY 标准
LIMIT  N;
```

执行顺序是 FROM → WHERE → SELECT → ORDER BY → LIMIT。
记成"先挑出 (WHERE)，再修列 (SELECT)，然后排队 (ORDER BY)，最后截取 (LIMIT)"。

## 4. 动手练习 — main.py

运行:

```bash
python3 main.py
```

- **[1] 最贵商品 Top5**: `ORDER BY price DESC LIMIT 5` — 排行榜的基本形。
- **[2] 毛利最大的商品 Top3**: 用计算式 `price - cost` 当排序标准。
- **[3] 最新 5 笔订单**: 日期列 DESC — "最近记录"套路。
- **[4] 多重排序**: 客户先按城市排，同城再按等级排。
- **[5] DISTINCT**: 客户实际居住的城市清单 — 确认 200 行缩成 6 行。
  接着还执行 (city, grade) 组合的 DISTINCT。
- **[6] 分页**: 在同一个排行查询上加 `LIMIT 5 OFFSET 5`，取出"第 2 页"。
- **[7] 陷阱演示**: 不加 ORDER BY 只用 LIMIT 5，得到的只是"最上面的 5 行"，
  根本不是 Top5 — 亲眼确认。

请务必比较输出里 [1] 和 [7] 的结果差在哪。同样是 LIMIT 5，含义完全不同。

## 5. 自己动手试试

1. **(简单)** 做一个"最便宜商品 Top3"。
   *提示: 把 DESC 换成 ASC (或者干脆省略)。*
2. **(中等)** 把员工 (employees) 按薪资从高到低排序、只取前 5 名，
   薪资相同的再按姓名排序。
   *提示: ORDER BY salary DESC, name ASC LIMIT 5。*
3. **(挑战)** 列出订单 (orders) 表里存在的所有状态 (status) **种类**，
   并猜猜每种分别代表什么。
   *提示: SELECT DISTINCT status FROM orders — 结果是 3 种。*

## 6. 常见误区

- **没有 ORDER BY 的 "Top N"**: 只用 LIMIT，数据库按自己顺手的顺序
  (等于随机) 给你 N 条而已。Top N = ORDER BY + LIMIT 成套，没有例外。
- **忘写 DESC 的排行榜**: 想拿"销售额 Top5"却拿到倒数 5 名，
  十有八九是漏了 DESC。记住默认是升序。
- **DISTINCT 随手乱放**: DISTINCT 只在 SELECT 后面写一次，
  作用于**选中的整个列组合**。`SELECT DISTINCT city, name`
  不是"只对城市去重"，而是"对 (城市, 姓名) 组合去重"。
- **长得像数字的文字排序**: 价格如果按 TEXT 存储，就会出现
  '9' > '1200000' 这种字典序排序。排序结果诡异时先查数据类型。
- **子句顺序打乱**: 像 `LIMIT 5 ORDER BY price` 这样调换顺序会报语法错误。
  守住 WHERE → ORDER BY → LIMIT 的顺序。

## 下一关预告

到目前为止我们只是把行挑出来排列，接下来终于要**汇总**了。
在 **level05 — 聚合函数与 GROUP BY** 中，"按城市数客户"、"按品类算销售额" —
用 SQL 干 Excel 数据透视表干过的活。
