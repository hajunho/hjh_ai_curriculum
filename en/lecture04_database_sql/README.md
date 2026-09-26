# Lecture 04 — Databases and SQL

> "Data is the company's ledger. SQL is the language you use to talk to that ledger."

This lecture takes you from someone who trades Excel files back and forth to
someone who can ask the company database (Database) questions directly. Every
exercise uses SQLite, which ships inside Python, so **there is nothing to
install.** Run `main.py` in any level folder and a practice online-store
database, `hjh_shop.db`, is built automatically, with each SQL statement and
its result printed side by side.

The star of this lecture is SQL. Python only plays the role of a "runner" that
executes the SQL for you. Focus on reading and re-typing the **SQL statements
themselves** rather than the Python code.

## What you will learn in this lecture

- Why a database beats sharing Excel files (concurrent edits, consistency, permissions)
- How tables, primary keys, and foreign keys connect your data
- SELECT / WHERE / ORDER BY / GROUP BY / JOIN / subqueries — turning real business questions into SQL
- INSERT / UPDATE / DELETE and transactions — changing data safely
- Indexes and execution plans — why slow queries are slow and how to fix them
- Window functions — analytical queries like rankings, running totals, and moving sums
- A mini pipeline that automates SQL results all the way to a report with Python + pandas

## Prerequisites

- **lecture02 (Python basics)** — being able to read and run main.py is enough.
- **lecture03 (NumPy · Pandas)** — level11 briefly uses pandas, but you can follow without it.

## Level index

| Level | Title | Difficulty |
|---|---|---|
| [level00](level00_why_databases/README.md) | Why Do We Need Databases? | ⭐ |
| [level01](level01_tables_keys/README.md) | Tables, Rows, Columns, and Primary Keys | ⭐ |
| [level02](level02_select_basics/README.md) | SELECT Basics | ⭐ |
| [level03](level03_where_filtering/README.md) | WHERE — Searching with Conditions | ⭐⭐ |
| [level04](level04_order_limit_distinct/README.md) | Sorting, Removing Duplicates, and Top N | ⭐⭐ |
| [level05](level05_aggregate_groupby/README.md) | Aggregate Functions and GROUP BY | ⭐⭐ |
| [level06](level06_joins/README.md) | JOIN — Connecting Multiple Tables | ⭐⭐⭐ |
| [level07](level07_subqueries/README.md) | Subqueries | ⭐⭐⭐ |
| [level08](level08_dml_transactions/README.md) | Modifying Data and Transactions | ⭐⭐⭐ |
| [level09](level09_indexes_performance/README.md) | Indexes and Query Performance | ⭐⭐⭐⭐ |
| [level10](level10_window_functions/README.md) | Window Functions and Analytical Queries | ⭐⭐⭐⭐ |
| [level11](level11_python_db_pipeline/README.md) | Python Integration and Data Pipelines | ⭐⭐⭐⭐ |

## Fast track (short on time? just these 5)

1. **level02 — SELECT basics**: where all SQL begins.
2. **level03 — WHERE**: picking out "only the rows that match." 80% of real-world questions.
3. **level05 — GROUP BY**: "by city," "by category" — the language of reports.
4. **level06 — JOIN**: real analysis starts when you connect multiple ledgers.
5. **level11 — Python integration**: the finishing move that turns SQL results into automated reports.

## How to practice

```bash
cd lecture04_database_sql/level02_select_basics
python3 main.py
```

Each main.py rebuilds `hjh_shop.db` inside its level folder every time it runs.
Even if you accidentally wreck the data, one more run restores everything —
so experiment freely. The output always shows the **SQL statement first**,
followed by the result table. Reading the SQL out loud is the best possible review.

## Where this lecture shows up at work

- **Marketing**: "Give me a list of last quarter's VIP customers who never bought again" → WHERE + subquery
- **Sales management**: "Revenue rankings by store and month, plus cumulative attainment" → GROUP BY + window functions
- **Inventory/operations**: "When an order comes in, deduct stock — but cancel the whole thing if stock runs short" → transactions
- **Data analysis**: pull data from the company DB with SQL, tidy it with pandas, and auto-send a weekly report
- **Working with engineers**: instead of "why is this screen slow?", saying "I think this query is missing an index"

Back when you only had Excel, you were the one asking someone to "pull the data."
By the end of this lecture, you become **the one who asks the question and pulls
the answer yourself.**
