# Lecture 04 · Level 05 — Aggregate Functions and GROUP BY

> Summarize with COUNT · SUM · AVG · MIN · MAX, build "per city / per category" subtotals with GROUP BY, and put conditions on the subtotals with HAVING. Everything the Excel pivot table did, now in SQL.

**Difficulty** ⭐⭐ / **Prerequisites** level04 / **Estimated time** 45 min

## 1. Why this matters — the business view

Executives don't look at row listings. They look at **summary numbers**: "how
many customers in total," "how many per city," "how much revenue per
category." The very thing you built pivot tables for in Excel is SQL's
aggregation (aggregate) and GROUP BY. The differences are scale and
reproducibility. Millions of rows get summarized inside the DB and only the
result comes back, so no file ever gets heavy — and a saved query rebuilds the
same weekly report in seconds. "I can GROUP BY" means "I can produce any
subtotal in any report, myself."

## 2. Understanding by analogy

This time you ask the clerk: "please **bundle the customer cards by city**,
and **count how many** are in each bundle."

The clerk works in two steps. First the cards get split into per-city
**baskets** (GROUP BY city); then **one summary calculation** runs per basket
(COUNT). The result isn't cards — it's **one line per basket**: New York 41
cards, Boston 33…

Which gives us the golden rule: **one row of a GROUP BY result = one basket.**
So the SELECT list may contain only (1) the basket's label (the grouping
column) and (2) basket summary values (aggregate functions). Ask for the name
of an individual card inside a basket and the clerk can only reply, "which
card?"

HAVING is the **sieve applied to the baskets after the counting is done**:
"only report the baskets with more than 35 cards." Where WHERE filters cards
one by one before they enter a basket, HAVING filters whole baskets after the
count.

## 3. Core concepts

### 3-1. The five aggregate siblings

| Function | Meaning | Example |
|---|---|---|
| COUNT(*) | row count | number of customers |
| SUM(x) | total | revenue total |
| AVG(x) | average | average unit price |
| MIN(x) / MAX(x) | smallest/largest | lowest/highest price |

An aggregate function takes many rows and folds them into **a single value**.
Used without GROUP BY, the whole table becomes one basket and the result is
one row.

### 3-2. The three faces of COUNT

- `COUNT(*)` — counts the rows themselves (NULLs included).
- `COUNT(column)` — counts only rows where that column is **not NULL**.
- `COUNT(DISTINCT column)` — the number of **different** non-NULL values.

Compare the three faces on employees' manager_id and the difference jumps out:
the CEO (whose manager_id is NULL) drops out of `COUNT(manager_id)`.

### 3-3. GROUP BY — building subtotals

```sql
SELECT city, COUNT(*) AS customer_cnt
FROM   customers
GROUP BY city
ORDER BY customer_cnt DESC;
```

You can line up several aggregates side by side: `COUNT(*), AVG(price),
MAX(price)`. Two grouping keys (`GROUP BY city, grade`) produce one row per
"city × grade" combination — like dropping two fields into a pivot table's
row/column areas.

### 3-4. HAVING — conditions on subtotals

```sql
SELECT city, COUNT(*) AS cnt
FROM   customers
GROUP BY city
HAVING COUNT(*) >= 35;
```

Telling this apart from WHERE is the exam question of this level.

- **WHERE**: **before** grouping, a condition on individual rows. Aggregate
  functions not allowed.
- **HAVING**: **after** grouping, a condition on baskets (groups). This is
  where aggregate functions go.

The real-world standard shape uses both together: "exclude cancelled orders
(WHERE), then only cities with 100+ orders (HAVING)."

### 3-5. Execution order, updated

```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

Filter → bundle → sieve the baskets → shape the summary → line up → cut.
Once you know this order, "why can't WHERE hold an aggregate" (nothing is
bundled yet) explains itself.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

- **[1] Whole-table summary**: total customers, then SUM/AVG/MIN/MAX of
  product prices in one row — aggregation without GROUP BY means "whole table
  = one basket."
- **[2] The three faces of COUNT**: `COUNT(*)` vs `COUNT(manager_id)` vs
  `COUNT(DISTINCT dept)`, compared on employees.
- **[3] Customers per city**: the GROUP BY basic form + sorted by count.
- **[4] Customers per city × grade**: two grouping keys.
- **[5] Revenue per category**: connects order line items (order_items) to
  products and computes `SUM(quantity * price)`. Joining two tables is the
  next level's star, so this is just an early taste of "so that's how they
  connect."
- **[6] HAVING**: only cities with 35+ customers. A comment also notes that
  trying the same thing with WHERE raises an error.
- **[7] WHERE + HAVING together**: exclude cancelled orders (WHERE), count
  orders per month, then keep only months with 80+ (HAVING) — the workplace
  standard shape.

## 5. Try it yourself

1. **(Easy)** Print the customer count per grade, largest first.
   *Hint: change city to grade in [3].*
2. **(Medium)** Compute the average salary per department, but show only
   departments averaging 5,500 or more.
   *Hint: GROUP BY dept HAVING AVG(salary) >= 5500.*
3. **(Challenge)** Count "completed" orders per month
   (`SUBSTR(ordered_at, 1, 7)`) and print only the top 3 busiest months.
   *Hint: WHERE status='completed' → GROUP BY month → ORDER BY count DESC LIMIT 3.*

## 6. Common mistakes

- **Selecting a column that isn't a grouping key**: `GROUP BY city` with
  `SELECT city, name` — "one row per basket" has no room for an individual
  card's name. (SQLite sometimes silently picks an arbitrary row instead of
  erroring, which is even more dangerous. Most other DBs raise an error.)
- **Aggregates in WHERE**: `WHERE COUNT(*) > 10` is an error. You can't count
  before bundling. Group conditions go in HAVING.
- **AVG and NULL**: AVG **ignores** NULLs when averaging. If you want "treat
  as zero and average," you must say so: `AVG(COALESCE(x, 0))`.
- **Integer division**: in SQLite, `SUM(a)/COUNT(*)` is integer-on-integer, so
  the decimals get chopped. Multiply by `1.0 *` or use AVG.
- **Skipping the sanity check**: the habit of once verifying that subtotals
  add up to the total (`SUM(subtotals) = total`) prevents report disasters.

## Next level preview

As you glimpsed in [5], real analysis begins when tables get connected. In
**level06 — JOIN** we link orders + customers + products to analyze "who
bought what," and cover the join trap (fan-out) too.
