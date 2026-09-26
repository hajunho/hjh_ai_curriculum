# Lecture 04 · Level 07 — Subqueries and CTEs

> Learn subqueries (scalar · IN · correlated), which use one query's result as another query's ingredient, and tidy complex queries into readable step-by-step sentences with WITH (CTE).

**Difficulty** ⭐⭐⭐ / **Prerequisites** level06 / **Estimated time** 50 min

## 1. Why this matters — the business view

Many real questions have "a threshold that must itself be computed from the
data." "Which products cost **more than average**?" — first you need the
average. "The list of customers **who bought a Laptop**?" — first you need the
list of laptop orders. "Employees paid **above their own department's
average**?" — the threshold differs per row. The tool that expresses these
two-step questions in one sentence is the subquery (subquery). And when an
analysis stretches to 3–4 steps, the CTE (Common Table Expression, the WITH
clause) — which names each step so the whole thing reads top to bottom —
becomes essential for collaboration. This is exactly where people who write
queries others can read part ways with people who don't.

## 2. Understanding by analogy

Ask the clerk for "products priced above average" and the clerk writes **two
sticky notes**. Note 1: "compute the average product price" → answer:
KRW 290,850. Note 2: "find products priced above 290,850." **Slotting note 1's
answer into note 2's blank** — that is a subquery. In SQL, parentheses are the
note's boundary: `WHERE price > (SELECT AVG(price) FROM products)`.

The shape of the note's answer determines the use:

- Answer is **a single number** (the average) → slots into a comparison =
  **scalar subquery**
- Answer is **a list** (order IDs) → slots into IN = **IN subquery**
- A fresh note gets written **for every outer row read** (that employee's
  department average) = **correlated subquery**

A CTE is **labeling the notes and laying them out on the desk**: "let's call
this note 'customer_totals'" — the next step uses it like a table. Instead of
parentheses nesting inside parentheses, it reads like a recipe: step 1, step 2.

## 3. Core concepts

### 3-1. Scalar subquery — the answer is one value

```sql
SELECT name, price
FROM   products
WHERE  price > (SELECT AVG(price) FROM products);
```

Read it as: the parentheses run first and collapse into a single value, then
the outer query runs. You can also put one in the SELECT list to show an
"overall average" column alongside each row.

### 3-2. IN subquery — the answer is a list

```sql
SELECT name FROM customers
WHERE  customer_id IN (SELECT customer_id
                       FROM orders
                       WHERE status = 'cancelled');
```

"Customers who have ever placed a cancelled order." The result inside the
parentheses (a list of customer IDs) fills IN's list slot. When negating
(`NOT IN`), there's a famous trap: one NULL in the list and the whole result
becomes 0 rows — so make it a habit to add `WHERE ... IS NOT NULL` to any
NOT IN subquery.

### 3-3. Correlated subquery — asking again for every row

```sql
SELECT e.name, e.dept, e.salary
FROM   employees AS e
WHERE  e.salary > (SELECT AVG(e2.salary)
                   FROM   employees AS e2
                   WHERE  e2.dept = e.dept);   -- references the outer row's dept!
```

Because the inner query references a value from the outer row (`e.dept`), the
inner query re-runs **for every outer row**. It's the standard solution for
questions where the threshold changes per row, like "salary above one's own
department average." It can get slow on very large row counts — that story is
level09.

### 3-4. CTE (the WITH clause) — naming the steps

```sql
WITH customer_totals AS (        -- step 1: spend per customer
    SELECT o.customer_id, SUM(oi.quantity * p.price) AS total
    FROM   orders o
    JOIN   order_items oi ON oi.order_id = o.order_id
    JOIN   products p     ON p.product_id = oi.product_id
    GROUP BY o.customer_id
)
SELECT c.name, t.total            -- step 2: keep only above-average
FROM   customer_totals t
JOIN   customers c ON c.customer_id = t.customer_id
WHERE  t.total > (SELECT AVG(total) FROM customer_totals);
```

The point is that the same intermediate result (customer_totals) gets reused
twice while the whole thing still reads top to bottom. You can chain several
steps inside WITH with commas. **"The moment a subquery is used twice or the
parentheses go two levels deep, promote it to a CTE"** — a rule of thumb that
holds up well at work.

### 3-5. Subqueries in the FROM clause

Put a parenthesized query in the FROM slot, give it an alias, and it acts like
a temporary table (a derived table). It overlaps with CTEs in capability, but
the CTE almost always reads better. When you meet one in someone else's query,
just interpret it as "a CTE without a name."

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

- **[1] Scalar**: products above the average price — the threshold is shown
  first (note 1), then the main query it slots into (note 2) runs, so you see
  the two steps connect.
- **[2] IN**: customers who have ever bought a 'Laptop' — a double IN subquery
  chaining order_items → orders → customers.
- **[3] Correlated**: employees paid above their own department's average —
  the per-department average table is shown first for comparison.
- **[4] CTE**: "customers who spend above average" — step 1 gets the name
  customer_totals, step 2 compares against its average. Notice the same CTE is
  reused twice.
- **[5] Multi-step CTE**: monthly revenue → find the best month, in two WITH
  steps — the classic skeleton of a report query.

## 5. Try it yourself

1. **(Easy)** Find "employees paid below the average salary" with a scalar
   subquery.
   *Hint: take [1]'s pattern and change only the table and the inequality.*
2. **(Medium)** Compute "the number of customers who have ever bought
   'Chocolate'."
   *Hint: wrap [2]'s query in COUNT(*) or change the SELECT list.*
3. **(Challenge)** Extend the CTE in [4] to find "customers at or above the
   top-10% spend line."
   *Hint: one way to get the line is ORDER BY total DESC LIMIT 1 OFFSET (10%
   of the headcount). Next level's window functions solve it more elegantly.*

## 6. Common mistakes

- **Multiple rows in a scalar slot**: if the parentheses of
  `price > (SELECT price FROM ...)` return several rows, it's an error (other
  DBs) or the first row silently gets used (SQLite). Guarantee "one value" in
  scalar slots with an aggregate or LIMIT 1.
- **NOT IN + NULL**: one NULL in the list and the whole result is 0 rows. Add
  an IS NOT NULL filter to NOT IN subqueries.
- **Skipping aliases in correlated subqueries**: when inner and outer use the
  same table, without aliases (e, e2) the columns tangle. Correlated
  subqueries require aliases, full stop.
- **Three levels of nested parentheses**: the only person who can read it is
  you. From two levels on, use a CTE.
- **Mistaking CTEs for a performance tool**: a CTE is fundamentally 'naming
  things for readability.' It does not make queries faster (speed is level09's
  topic).

## Next level preview

So far we've only read. In **level08 — Modifying Data and Transactions** we
finally change data with INSERT/UPDATE/DELETE — and learn the safety net
(COMMIT/ROLLBACK) that lets you undo mistakes.
