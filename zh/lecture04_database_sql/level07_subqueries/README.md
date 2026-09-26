# Lecture 04 · Level 07 — 子查询

> 学习把一个查询的结果当另一个查询的原料的子查询 (标量、IN、相关)，再用 WITH (CTE) 把复杂查询整理成好读的分步语句。

**难度** ⭐⭐⭐ / **先修** level06 / **预计时间** 50分钟

## 1. 为什么要学 — 业务视角

业务问题里有很多"标准本身也得从数据里算出来"的类型。
"比**平均**贵的商品有哪些？" — 得先算出平均是多少。
"**买过笔记本电脑**的客户名单？" — 得先拿到买了笔记本电脑的订单清单。
"薪资**高于本部门平均**的员工？" — 每一行的标准都不一样。
把这种两步走的问题写成一句话的工具，就是子查询 (subquery)。
当分析拉长到三四步时，给每一步起名字、让语句自上而下可读的
CTE (Common Table Expression, WITH 子句) 就成了协作必需品。
"写别人读得懂的查询"和"写只有自己懂的查询"，分水岭就在这里。

## 2. 打个比方

你对柜台职员说"给我比平均贵的商品清单"，职员会写**两张便条**。
第一张: "算商品价格的平均" → 答案 290,850 韩元。第二张:
"找价格超过 290,850 韩元的商品"。把第一张的**答案填进第二张的空格** ——
这就是子查询。在 SQL 里，括号就是便条的边界:
`WHERE price > (SELECT AVG(price) FROM products)`。

便条上答案的形状不同，用法也不同。

- 答案是**一个数** (平均值) → 填进比较运算 = **标量子查询**
- 答案是**一份名单** (一串订单号) → 填进 IN = **IN 子查询**
- 外层**每读一行就重写一张便条** (那位员工所在部门的平均) =
  **相关子查询**

CTE 则是**给便条贴上名签、摊在桌上**: "这张便条就叫'客户购买额'吧" ——
下一步把它当成一张表来用。括号套括号变成了像菜谱一样的
第 1 步、第 2 步，从上往下读。

## 3. 核心概念

### 3-1. 标量子查询 — 答案是一个值

```sql
SELECT name, price
FROM   products
WHERE  price > (SELECT AVG(price) FROM products);
```

读法是: 括号里先执行、变成一个值，然后外层查询再执行。
也可以放进 SELECT 清单，让"全体平均"作为一列并排显示。

### 3-2. IN 子查询 — 答案是一份名单

```sql
SELECT name FROM customers
WHERE  customer_id IN (SELECT customer_id
                       FROM orders
                       WHERE status = '已取消');
```

"下过已取消订单的客户"。括号里的结果 (客户编号名单) 填进 IN 的名单位置。
用反义 (`NOT IN`) 时有个著名陷阱: 名单里混进 NULL，结果就整个变成 0 条。
请养成给 NOT IN 子查询配一句 `WHERE ... IS NOT NULL` 的习惯。

### 3-3. 相关子查询 — 每行重新问一次

```sql
SELECT e.name, e.dept, e.salary
FROM   employees AS e
WHERE  e.salary > (SELECT AVG(e2.salary)
                   FROM   employees AS e2
                   WHERE  e2.dept = e.dept);   -- 引用了外层行的部门!
```

内层查询引用了外层行的值 (`e.dept`)，所以**外层每一行**都会让内层
重新执行一次。像"高于本部门平均的薪资"这种每行标准不同的问题，
这就是标准解法。行数非常多时它可能变慢，这一点 level09 会讲。

### 3-4. CTE (WITH 子句) — 给步骤起名字

```sql
WITH customer_totals AS (        -- 第1步: 客户购买额
    SELECT o.customer_id, SUM(oi.quantity * p.price) AS total
    FROM   orders o
    JOIN   order_items oi ON oi.order_id = o.order_id
    JOIN   products p     ON p.product_id = oi.product_id
    GROUP BY o.customer_id
)
SELECT c.name, t.total            -- 第2步: 只挑平均以上的
FROM   customer_totals t
JOIN   customers c ON c.customer_id = t.customer_id
WHERE  t.total > (SELECT AVG(total) FROM customer_totals);
```

关键在于: 同一个中间结果 (customer_totals) 被复用了两次，
而语句依然自上而下可读。WITH 里还可以用逗号接上多个步骤。
**"子查询被用到两次以上、或者括号叠到两层，就升格成 CTE"** —
这是实战里很好用的标准。

### 3-5. FROM 子句里的子查询

把带括号的查询放在 FROM 的位置、加个别名，就能当临时表用 (派生表)。
功能和 CTE 重叠，但可读性几乎总是 CTE 更好。
在别人的查询里遇到它，理解成"没有名字的 CTE"就行。

## 4. 动手练习 — main.py

运行:

```bash
python3 main.py
```

- **[1] 标量**: 比平均贵的商品 — 先展示标准值 (便条 1)，
  再执行填进了它的正式查询 (便条 2)，让两步在眼前连起来。
- **[2] IN**: 买过"笔记本电脑"的客户名单 — order_items → orders →
  customers 一路接力的双层 IN 子查询。
- **[3] 相关**: 薪资高于本部门平均的员工 — 先亮出部门平均表，
  再和相关子查询的结果对照。
- **[4] CTE**: "购买额高于平均的客户" — 给第 1 步起名 customer_totals，
  第 2 步和平均比较。注意同一个 CTE 被复用了两次。
- **[5] 多步 CTE**: 先算月度销售额 → 再找销售额最高的月份，
  用 WITH 分两步 — 报表查询的典型骨架。

## 5. 自己动手试试

1. **(简单)** 用标量子查询求"薪资低于平均薪资的员工"。
   *提示: 在 [1] 的套路上换掉表和不等号即可。*
2. **(中等)** 求"买过'巧克力'的客户数"。
   *提示: 把 [2] 的查询包进 COUNT(*)，或者改 SELECT 清单。*
3. **(挑战)** 扩展 [4] 的 CTE，求"购买额达到前 10% 分界线以上的客户"。
   *提示: 分界线可以用 ORDER BY total DESC LIMIT 1 OFFSET (人数的10%)
   取出来。用下一关的窗口函数解会更优雅。*

## 6. 常见误区

- **标量位置返回了多行**: `price > (SELECT price FROM ...)` 的括号
  返回多行时，有的数据库报错，SQLite 则悄悄只用第一行。
  标量位置请用聚合或 LIMIT 1 保证"只有一个值"。
- **NOT IN + NULL**: 名单里哪怕只有一个 NULL，结果就整个 0 条。
  NOT IN 子查询要配 IS NOT NULL 过滤。
- **相关子查询省略别名**: 内外是同一张表时，不写别名 (e, e2)
  就分不清是哪边的列。相关子查询别名必写。
- **括号叠三层**: 能读懂的就只剩你自己了。叠到两层就换 CTE。
- **把 CTE 误当性能工具**: CTE 本质是"起名字让人好读"，
  并不会因此变快 (速度是 level09 的主题)。

## 下一关预告

到现在为止我们一直在"读"。在 **level08 — 数据修改与事务** 中，
终于要用 INSERT/UPDATE/DELETE 修改数据，
并学习就算失手也能回退的安全装置 (COMMIT/ROLLBACK)。
