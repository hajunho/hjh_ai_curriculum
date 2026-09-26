# Lecture 04 · Level 10 — Window Functions and Analytical Queries

> Learn window functions, which attach a summary beside each row without folding the rows away. Build sequence numbers and ranks with ROW_NUMBER · RANK and running totals and moving sums with SUM OVER — and burn the difference from GROUP BY into muscle memory.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level05, level06 / **Estimated time** 55 min

## 1. Why this matters — the business view

Build enough reports and you'll hit requests GROUP BY can't serve: "mark
**which purchase** this is for each customer" (first-purchase analysis), "rank
revenue **within** each department," "add a **running** total next to monthly
revenue," "show the trend as a **3-month moving sum**." The common thread:
**leave the rows as they are** and attach computed values — sequence, rank,
cumulative — as an extra column. GROUP BY folds rows away, so it can't. The
window function (window function) is exactly this "aggregation that doesn't
fold," a fixture of analyst hiring tests, and the tool that turns the
end-of-month ritual of copy-the-formula-and-re-sort in Excel into one
sentence.

## 2. Understanding by analogy

GROUP BY was the clerk who **puts cards in baskets and keeps only one summary
line each**. 200 cards folded into 6 city lines.

A window function is different. The clerk keeps every card **spread out on the
desk** and sticks a post-it beside each one. What goes on the post-it is
specified three ways:

- **which range to look at** (PARTITION BY — "only cards of the same customer")
- **counting in which order** (ORDER BY — "by order date")
- **what to write** (ROW_NUMBER — "which one this is", SUM — "the total so far")

The name "window" makes sense if you imagine each card having its own **window
looking out at just some of the surrounding cards**: a window showing only
this customer's cards, a window showing only cards up to this month.
Summarizing the view and writing it on your own post-it — that's
`function() OVER (window definition)`.

## 3. Core concepts

### 3-1. Basic syntax — OVER is the signal

```sql
SELECT name, category, price,
       AVG(price) OVER (PARTITION BY category) AS cat_avg
FROM   products;
```

The moment `OVER` appears, AVG stops folding rows. All 10 product rows come
back, each with "its own category's average price" attached. With
`price - cat_avg` you immediately get "how far above the category average is
this?" — the old routine of joining a GROUP BY result back to the original is
done in one clause.

### 3-2. ROW_NUMBER — sequence numbers

```sql
ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY ordered_at) AS nth
```

A purchase counter restarting at 1 for every customer. Filtering to
`WHERE nth = 1` (wrapped in a subquery or CTE) to pull "each customer's first
order" is the most common pattern in practice. "Latest/first/largest 1 per
group" = ROW_NUMBER + filter — memorize it like a formula.

### 3-3. RANK / DENSE_RANK — ranks and ties

All three order rows, but they treat ties differently.

| Value | ROW_NUMBER | RANK | DENSE_RANK |
|---|---|---|---|
| 100 | 1 | 1 | 1 |
| 90 | 2 | 2 | 2 |
| 90 | 3 | 2 | 2 |
| 80 | 4 | **4** | **3** |

ROW_NUMBER lines up ties too (in arbitrary order), RANK skips to 4 after a
shared 2nd place, DENSE_RANK continues at 3. Award ceremonies use RANK, "top
3 tiers" uses DENSE_RANK, and "one per line, no matter what" uses ROW_NUMBER.

### 3-4. Running totals — SUM OVER with an ORDER BY

```sql
SUM(revenue) OVER (ORDER BY month) AS cum_revenue
```

The ORDER BY makes the window "from the start up to the current row." Monthly
revenue gets a year-to-date cumulative column beside it. Add a PARTITION BY
and you get per-group running totals, like "each department accumulating its
own."

### 3-5. Moving sums/averages — ROWS BETWEEN

```sql
SUM(revenue) OVER (ORDER BY month
                   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov3
```

"The previous 2 rows + the current row" — a 3-month moving sum. A reporting
staple for smoothing jagged monthly figures into a visible trend, using the
syntax that sets the window's size explicitly.

### 3-6. Watch the execution order

Window functions are computed **after** WHERE and GROUP BY finish. So a window
result (nth, rank…) can't go straight into WHERE — wrap it in a CTE or
subquery and filter outside. main.py's [2] is exactly this pattern.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

- **[1] GROUP BY vs window**: the same "category average price" computed both
  ways — compare the 5-row vs 10-row difference (folded vs kept) side by side.
- **[2] Purchase sequence per customer**: ROW_NUMBER assigns the sequence,
  then a CTE wrapper pulls "each customer's first order."
- **[3] The tie-handling trio**: ROW_NUMBER/RANK/DENSE_RANK side by side on
  product prices (data with ties), reproducing section 3-3's table with real
  data.
- **[4] Revenue rank within department**: compute revenue handled per employee
  (CTE), then RANK within each department — the skeleton of every "rank by
  branch" report.
- **[5] Monthly revenue + running total**: SUM OVER (ORDER BY month).
- **[6] 3-month moving sum**: ROWS BETWEEN 2 PRECEDING AND CURRENT ROW.

For every output, the key thing to watch is: "did the rows fold, or stay?"

## 5. Try it yourself

1. **(Easy)** Modify [3] to rank salaries within each department
   (PARTITION BY dept).
   *Hint: add PARTITION BY dept inside the OVER parentheses.*
2. **(Medium)** Pull only each customer's **last** order.
   *Hint: flip [2] to ORDER BY ordered_at DESC and number 1 becomes the last
   order.*
3. **(Challenge)** Add a "change vs previous month" column to monthly revenue.
   *Hint: LAG(revenue) OVER (ORDER BY month) fetches 'the previous row's
   value'. revenue - LAG(revenue) OVER (...).*

## 6. Common mistakes

- **Window results directly in WHERE**: `WHERE ROW_NUMBER() ...` is an error.
  Wrap in a CTE and filter outside — always this two-stage structure.
- **Confusing PARTITION BY with GROUP BY**: PARTITION BY doesn't fold.
  Distinguish them as "row count shrinks → GROUP BY, row count kept → window."
- **Running totals missing their ORDER BY**: without ORDER BY inside OVER (),
  the window becomes everything and every row gets the grand total. When a
  cumulative looks wrong, check the ORDER BY first.
- **ROW_NUMBER reproducibility with ties**: which tied row gets #1 can differ
  between runs. Adding a unique key to the ORDER BY (`ORDER BY amount DESC,
  order_id`) to pin the order down is good reporting manners.
- **Old DB versions**: window functions require SQLite 3.25 (2018) or later.
  On a very old company DB you may get syntax errors.

## Next level preview

Your SQL toolkit is complete. In the final level, **level11 — Python
Integration and Data Pipelines**, you'll connect SQL with Python and pandas to
build "the summary report that appears automatically every morning," learn the
parameter binding that blocks SQL injection, and close out the lecture.
