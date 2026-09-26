# Lecture 04 · Level 06 — JOIN: 连接多张表

> 学习用键把拆开存放的表重新拼起来的 JOIN。覆盖 INNER JOIN 与 LEFT JOIN 的差别、怎么选连接键，以及会把汇总吹大的扇出陷阱。

**难度** ⭐⭐⭐ / **先修** level05 / **预计时间** 50分钟

## 1. 为什么要学 — 业务视角

正如 level01 所见，数据库按"一个事实只记一处"的原则把数据拆表存放。
所以业务问题几乎总是横跨好几张表: "谁 (customers) 买了什么 (products)
买了多少 (order_items) 什么时候买的 (orders)?" — 得把四张表拼起来才有答案。
不会 JOIN，SQL 就停留在单表玩具的水平；会了 JOIN，
整个公司数据库都成了你的分析对象。还有一点: 像"没下过单的客户"这种
**找不存在之物的问题** (沉睡客户、未回款单位)，没有 LEFT JOIN 根本表达不出来。

## 2. 打个比方

订单账本上只写着"会员号 7、商品号 3"这样的编号。要出报表，
职员得**读一行订单账本 → 拿着会员号去客户名册里找到那一行 →
把姓名抄过来**。这个"拿着编号去别的账本查"就是 JOIN，
手里拿的编号就是**连接键 (join key)** — 通常是一边的外键对另一边的主键。

INNER JOIN 和 LEFT JOIN 的差别在于**没配上对的行怎么处理**。
想想相亲活动的名单: 把报名名单 (客户) 和配对记录 (订单) 拼在一起时:

- **INNER JOIN**: 只留配上对的人。没有配对记录的报名者从结果里消失。
- **LEFT JOIN**: 左边名单**全员保留**。没配上的人，记录栏留成 NULL (空格)。
  → "找没配上对的人" = 找空格的行。

要找"没下过单的客户"，就把客户放左边做 LEFT JOIN，
再挑出订单栏为 NULL 的行。

## 3. 核心概念

### 3-1. INNER JOIN 基本形

```sql
SELECT o.order_id, c.name, o.ordered_at
FROM   orders AS o
JOIN   customers AS c ON c.customer_id = o.customer_id;
```

- `AS o`、`AS c`: 表别名。出现多张表时，列名前面加 `别名.`
  写明是哪张表的列。
- `ON`: 用哪个键配对。**外键 = 主键**是标准形态。
- 只写 `JOIN` 就是 INNER JOIN 的缩写。

### 3-2. 连接三张以上的表

JOIN 像链条一样一节节接上去。订单明细 → 订单 → 客户，再加订单明细 → 商品。
把 level01 的关系图放在手边，**顺着外键的箭头**写 ON 子句就不会迷路。

```sql
FROM order_items AS oi
JOIN orders    AS o ON o.order_id    = oi.order_id
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products  AS p ON p.product_id  = oi.product_id
```

### 3-3. LEFT JOIN — 左边全员生还

```sql
SELECT c.name, o.order_id
FROM   customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id;
```

没有订单的客户也留在结果里，order_id 栏是 NULL。再加一句
`WHERE o.order_id IS NULL`，就只剩"没下过单的客户" —
沉睡客户分析的正统套路。(RIGHT JOIN 只是左右互换，
实战中大家几乎都统一用 LEFT。)

### 3-4. 扇出 (fan-out) 陷阱 — 连接会让行变多

连接 1:N 关系时，**1 那边的行会被复制成 N 份**。一笔订单里有 3 种商品，
orders 一接上 order_items，这笔订单就变成 3 行。到这里都还正常，
事故发生在下一步: 在膨胀后的结果上用 `COUNT(*)` 数"订单数"，
得到的不是 1,000 笔而是**被吹大到** 2,000 笔上下。
把客户表错误地接到销售额合计上，也会发生同样的事。

两种防御:

1. 用汇总对象的**唯一键来数**: `COUNT(DISTINCT o.order_id)`。
2. **只连接需要的表**: 这次汇总用不上的表就不要接。

连接后的汇总结果"比感觉的大"，第一反应就该怀疑扇出。

### 3-5. ON 和 WHERE 的分工

ON 是"配对规则"，WHERE 是"在拼好的结果里挑选的条件"。
INNER JOIN 里写哪边结果都一样，但 LEFT JOIN 里一旦把右表条件写进 WHERE，
NULL 行就会被淘汰，**事实上变回 INNER JOIN**。
LEFT JOIN 的右表条件写在 ON 里才安全。

## 4. 动手练习 — main.py

运行:

```bash
python3 main.py
```

- **[1] 两张表**: 订单+客户 — 编号变成姓名的第一次连接。
- **[2] 四张表**: "谁买了什么、几件、多少钱" — 以订单明细为中心，
  把订单、客户、商品链式连接。
- **[3] 连接+聚合**: 客户总购买额 Top5 — JOIN 与 GROUP BY 的合体。
- **[4] LEFT JOIN**: 客户全员 + 各自的订单数。确认没有订单的客户显示为 0。
- **[5] 反连接**: 用 `WHERE o.order_id IS NULL` 拿到"一次都没下过单的客户"
  名单 — 沉睡客户报表。
- **[6] 扇出演示**: 同一个"订单数"问题用三种方式数:
  (a) orders 单表、(b) 接上 order_items 后 COUNT(*)、
  (c) COUNT(DISTINCT order_id)，用数字确认只有 (b) 被吹大。
  这张表是本关最值钱的输出。

## 5. 自己动手试试

1. **(简单)** 改一改 [1]，把订单和员工 (employees) 接起来，
   做一张"这笔订单是哪位员工处理的"表。
   *提示: ON e.employee_id = o.employee_id。*
2. **(中等)** 按商品统计销量合计并从多到少输出。
   *提示: order_items JOIN products → GROUP BY p.name → SUM(oi.quantity)。*
3. **(挑战)** 找出"只有已取消订单的客户"。只要有一笔已完成/配送中的订单
   就排除。
   *提示: 一种做法 — 在 LEFT JOIN 的 ON 里加上 o.status <> '已取消'
   条件再用 IS NULL 过滤，然后和全部订单比对。用子查询 (下一关) 解会更干净。*

## 6. 常见误区

- **漏写 ON (笛卡尔积)**: 不写 ON 就连接，会生成所有行×所有行的组合。
  1,000 行×200 行=20 万行。结果爆炸时先检查 ON。
- **用错键连接**: 用姓名这种可能重复的列连接，重名的人会缠在一起。
  连接键要用主键↔外键。
- **在扇出状态下 SUM/COUNT**: 见 3-4 节。汇总前先自问
  "这次连接让行变多了吗？"
- **LEFT JOIN + 右表条件写进 WHERE**: NULL 行被切掉，变回 INNER。
  右表条件放 ON。
- **同名列不加别名**: 两张表都有的 name 直接裸写，轻则报歧义错误，
  重则悄悄选错了表。连接语句里给所有列加别名是好习惯。

## 下一关预告

还剩一类问题: 像"买得比平均多的客户"这种**把查询结果当另一个查询的原料**
的问题。在 **level07 — 子查询** 中学习查询里的查询，
以及把它整理得好读的 WITH 语法。
