# Lecture 04 · Level 05 — 聚合函数与 GROUP BY

> 用 COUNT·SUM·AVG·MIN·MAX 做汇总，用 GROUP BY 做"按城市、按品类"的小计，再用 HAVING 给小计设条件。用 SQL 干 Excel 数据透视表干过的活。

**难度** ⭐⭐ / **先修** level04 / **预计时间** 45分钟

## 1. 为什么要学 — 业务视角

管理层不看行清单，他们看**汇总数字**: "客户一共多少人"、
"各城市分别多少人"、"各品类销售额是多少"。在 Excel 里做数据透视表的活，
在 SQL 里就是聚合 (aggregate) 和 GROUP BY。差别在于规模和可复现性:
几百万行也是在数据库里汇总后只拿回结果，文件永远不会臃肿；
查询存下来，每周同样的报表几秒钟就能重做一遍。
"会用 GROUP BY"意味着"报表里任何一个小计都能自己做出来"。

## 2. 打个比方

这次这样拜托柜台职员: "请把客户卡片**按城市分堆**，
每一堆**数一数有几张**。"

职员的操作分两步: 先把卡片分进各城市的**篮子** (GROUP BY city)，
再对每个篮子做**一次汇总计算** (COUNT)。
结果不是卡片，而是**每个篮子一行** — 广州 37 张、上海 36 张……

大原则由此而来: **GROUP BY 结果的一行 = 一个篮子。**
所以 SELECT 里只能放 (1) 篮子的标签 (分组标准列) 和
(2) 篮子的汇总值 (聚合函数)。要是问篮子里某张卡片的姓名，
职员只能反问"哪张卡片？"

HAVING 是汇总完之后**筛选篮子的筛子**: "只报告超过 35 张的篮子。"
WHERE 是在装篮子之前一张一张地筛卡片，HAVING 是数完之后筛篮子。

## 3. 核心概念

### 3-1. 聚合函数五兄弟

| 函数 | 含义 | 例子 |
|---|---|---|
| COUNT(*) | 行数 | 客户数 |
| SUM(x) | 合计 | 销售额合计 |
| AVG(x) | 平均 | 平均单价 |
| MIN(x) / MAX(x) | 最小/最大 | 最低价/最高价 |

聚合函数把多行折叠成**一个值**。不带 GROUP BY 时，
整张表就是一个篮子，结果也只有一行。

### 3-2. COUNT 的三副面孔

- `COUNT(*)` — 数行本身 (含 NULL)。
- `COUNT(列)` — 只数该列**不是 NULL** 的行。
- `COUNT(DISTINCT 列)` — 非 NULL 的**不同值**的个数。

拿 employees 的 manager_id 比较这三副面孔，差异一目了然。
总经理 (manager_id 为 NULL) 不会被 `COUNT(manager_id)` 数进去。

### 3-3. GROUP BY — 做小计

```sql
SELECT city, COUNT(*) AS customer_cnt
FROM   customers
GROUP BY city
ORDER BY customer_cnt DESC;
```

聚合可以并排放好几个: `COUNT(*), AVG(price), MAX(price)`。
分组标准写两个 (`GROUP BY city, grade`)，就是每个"城市×等级"组合一行 —
相当于往数据透视表的行/列区域各拖一个字段。

### 3-4. HAVING — 给小计设条件

```sql
SELECT city, COUNT(*) AS cnt
FROM   customers
GROUP BY city
HAVING COUNT(*) >= 35;
```

它和 WHERE 的区分是本关的核心考点。

- **WHERE**: 分组**之前**，针对单行的条件。不能用聚合函数。
- **HAVING**: 分组**之后**，针对篮子 (组) 的条件。聚合函数写在这里。

"剔除已取消的订单 (WHERE)，只看订单 100 笔以上的城市 (HAVING)" —
两者搭配才是实战标准形。

### 3-5. 执行顺序升级版

```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

挑出来 → 分堆 → 筛篮子 → 修整汇总 → 排队 → 截取。
记住这个顺序，"为什么 WHERE 里不能用聚合" (因为还没分组、没法数)
就自然想通了。

## 4. 动手练习 — main.py

运行:

```bash
python3 main.py
```

- **[1] 整体汇总**: 商品总数，加上价格的 SUM/AVG/MIN/MAX 一行输出 —
  不带 GROUP BY 的聚合就是"整张表 = 一个篮子"。
- **[2] COUNT 的三副面孔**: 用 employees 比较 `COUNT(*)` vs
  `COUNT(manager_id)` vs `COUNT(DISTINCT dept)`。
- **[3] 各城市客户数**: GROUP BY 基本形 + 按多到少排序。
- **[4] 城市×等级客户数**: 两个分组标准。
- **[5] 各品类销售额**: 把订单明细 (order_items) 和商品 (products) 连起来
  算 `SUM(quantity * price)`。连接两张表的 JOIN 是下一关的主角，
  这里先尝个"原来是这样接的"的鲜。
- **[6] HAVING**: 只要客户 35 人以上的城市。注释里还说明了
  用 WHERE 做同样的事会报错。
- **[7] WHERE + HAVING 联手**: 剔除已取消订单 (WHERE) 后按月数订单，
  只留 80 笔以上的月份 (HAVING) — 实战标准形。

## 5. 自己动手试试

1. **(简单)** 按等级 (grade) 统计客户数并从多到少输出。
   *提示: 把 [3] 的 city 换成 grade。*
2. **(中等)** 求各部门 (dept) 的平均薪资，只显示平均 5,500 以上的部门。
   *提示: GROUP BY dept HAVING AVG(salary) >= 5500。*
3. **(挑战)** 按月份 (`SUBSTR(ordered_at, 1, 7)`) 统计"已完成"订单数，
   只输出最忙的 3 个月。
   *提示: WHERE status='已完成' → GROUP BY 月份 → ORDER BY 笔数 DESC LIMIT 3。*

## 6. 常见误区

- **把不在分组标准里的列放进 SELECT**: `GROUP BY city` 却写
  `SELECT city, name` — "每个篮子一行"里放不下单张卡片的姓名。
  (SQLite 不报错，而是随手挑一行给你，反而更危险。多数数据库直接报错。)
- **在 WHERE 里用聚合函数**: `WHERE COUNT(*) > 10` 报错。
  分组之前没法数数。组条件请交给 HAVING。
- **AVG 与 NULL**: AVG 会**跳过 NULL** 再取平均。想"按 0 计入平均"，
  必须写明 `AVG(COALESCE(x, 0))`。
- **整数除法**: SQLite 里 `SUM(a)/COUNT(*)` 是整数相除，小数会被截掉。
  乘个 `1.0 *` 或者直接用 AVG。
- **省略汇总检算**: 养成习惯核对一次"小计之和是否等于总计"
  (`SUM(小计) = 总计`)，能挡住报表事故。

## 下一关预告

正如 [5] 里浅尝的那样，真正的分析要从拼接多张表开始。
在 **level06 — JOIN** 中，把订单+客户+商品连起来分析"谁买了什么"，
还会讲连接的陷阱 (扇出)。
