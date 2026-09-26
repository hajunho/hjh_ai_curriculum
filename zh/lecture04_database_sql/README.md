# Lecture 04 — 数据库与 SQL

> "数据是公司的账本，SQL 就是和这本账本对话的语言。"

这是一门帮你完成转变的课：从靠传 Excel 文件干活的人，成长为能直接向公司
数据库 (Database) 提问的人。所有练习都使用 Python 内置的 SQLite，
所以**什么都不用安装**。在每个关卡文件夹里运行 `main.py`，
就会自动生成练习用的网店数据库 `hjh_shop.db`，
并把 SQL 语句和运行结果并排打印出来。

这门课的主角是 SQL。Python 只是负责执行 SQL 的"执行器"。
请把注意力放在**SQL 语句本身**上，多读、多跟着写。

## 这门课你将学到

- 为什么数据库比共享 Excel 更靠谱 (并发修改、数据一致性、权限)
- 表、主键、外键是如何把数据连接起来的
- SELECT / WHERE / ORDER BY / GROUP BY / JOIN / 子查询 — 把业务问题翻译成 SQL 的方法
- INSERT / UPDATE / DELETE 与事务 — 安全地修改数据的方法
- 索引与执行计划 — 让慢查询变快的原理
- 窗口函数 — 排名、累计、移动求和这类分析查询
- 用 Python + pandas 把 SQL 结果一路自动化到报表的迷你数据管道

## 先修课程

- **lecture02 (Python 编程基础)** — 能读懂并运行 main.py 就足够了。
- **lecture03 (NumPy·Pandas)** — level11 会用到一点 pandas，不会也能继续学。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_why_databases/README.md) | 为什么需要数据库 | ⭐ |
| [level01](level01_tables_keys/README.md) | 表、行、列与主键 | ⭐ |
| [level02](level02_select_basics/README.md) | SELECT 基础 | ⭐ |
| [level03](level03_where_filtering/README.md) | WHERE — 条件查询 | ⭐⭐ |
| [level04](level04_order_limit_distinct/README.md) | 排序、去重与前 N 名 | ⭐⭐ |
| [level05](level05_aggregate_groupby/README.md) | 聚合函数与 GROUP BY | ⭐⭐ |
| [level06](level06_joins/README.md) | JOIN — 连接多张表 | ⭐⭐⭐ |
| [level07](level07_subqueries/README.md) | 子查询 | ⭐⭐⭐ |
| [level08](level08_dml_transactions/README.md) | 数据修改与事务 | ⭐⭐⭐ |
| [level09](level09_indexes_performance/README.md) | 索引与查询性能 | ⭐⭐⭐⭐ |
| [level10](level10_window_functions/README.md) | 窗口函数与分析查询 | ⭐⭐⭐⭐ |
| [level11](level11_python_db_pipeline/README.md) | 连接 Python 与数据管道 | ⭐⭐⭐⭐ |

## 快速通道 (时间紧就先学这 5 个)

1. **level02 — SELECT 基础**: 所有 SQL 的出发点。
2. **level03 — WHERE**: 只挑出"符合条件的那些"。业务问题的 80%。
3. **level05 — GROUP BY**: "按城市"、"按品类" — 报表的语言。
4. **level06 — JOIN**: 把多本账本连起来，才是真正的分析。
5. **level11 — 连接 Python**: 把 SQL 结果做成自动报表的收官之作。

## 练习方法

```bash
cd lecture04_database_sql/level02_select_basics
python3 main.py
```

每个 main.py 每次运行时都会在关卡文件夹里重新生成 `hjh_shop.db`。
就算不小心把数据弄坏了，重跑一次就会原样恢复，尽管放心大胆地实验。
输出里永远是**SQL 语句在前**，结果表格在后。
把 SQL 语句念出声来，是最好的复习方式。

## 这门课在实际工作中的应用场景

- **市场营销**: "把上季度 VIP 客户里没有复购的人的名单给我" → WHERE + 子查询
- **销售管理**: "各门店、各月份的销售额排名和累计达成率" → GROUP BY + 窗口函数
- **库存/运营**: "订单进来就扣减库存，库存不够就整单取消" → 事务
- **数据分析**: 用 SQL 从公司数据库取数，pandas 整理，每周自动发送周报
- **与开发协作**: 不再问开发"这个页面怎么这么慢"，而是说"这条查询好像缺个索引"

以前只会用 Excel 时，你是那个开口求人"帮我导一下数据"的人；
学完这门课，你就成了**自己提问、自己把答案取出来的人**。
