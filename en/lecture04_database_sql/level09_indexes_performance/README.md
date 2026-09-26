# Lecture 04 · Level 09 — Indexes and Query Performance

> Understand why an index speeds up search through the book-index analogy, read the DB's game plan with EXPLAIN QUERY PLAN, and compare measured times before and after adding an index on a 300,000-row table.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level08 / **Estimated time** 50 min

## 1. Why this matters — the business view

"Why is this screen so slow?" — a large share of complaints about internal
systems come down to **searches without an index**. While the data is small,
every query is fast and the problem stays hidden; as data piles up, things
suddenly get tens of times slower. Knowing how an index (index) works gives
you two abilities. First, when your own query is slow, you can diagnose the
cause instead of vaguely waiting. Second, you can talk to the engineering team
at the level of "I think this customer_id lookup is running as a full scan —
could you check the index?" Non-engineers who can say that sentence are
genuinely rare, and they get treated accordingly.

## 2. Understanding by analogy

**An index is the index at the back of a book.** Two ways to find where
'depreciation' appears in a 600-page book: (1) skim from page 1 to the end — a
full scan (full scan). (2) look up 'depreciation → pages 152, 380' in the back
index and open straight to them — an index lookup. The index is alphabetized,
so a few glances land you on the entry. If the data grows 10×, a full scan
gets 10× slower, but the cost of looking something up in a sorted index barely
grows at all.

It isn't free. Insert new content into a book that has an index and **the
index must be updated too**. The more indexes a table carries, the slower its
INSERT/UPDATE gets, and the more storage it uses. Hence the principle: indexes
go "only on frequently searched columns, only as many as needed."

One more thing — an index is sorted **from the first letter**. You can't look
up a word by its middle, like '…ciation'. That's why a middle-match search
such as `LIKE '%Coffee%'` becomes a full scan even when an index exists.

## 3. Core concepts

### 3-1. Creating an index

```sql
CREATE INDEX idx_big_customer ON big_orders (customer_id);
```

Specify `table(column)` and a separate sorted data structure (a B-tree) is
built for that column's values; afterwards the DB uses the index **on its
own** for queries that search by that column. Not one character of your query
changes — only the execution path gets faster.

### 3-2. EXPLAIN QUERY PLAN — reading the DB's game plan

Prefix a query with `EXPLAIN QUERY PLAN` and it shows **only the plan**,
without executing.

- `SCAN big_orders` — "I will skim every row" (full scan). A warning sign on a
  big table.
- `SEARCH big_orders USING INDEX idx_... (customer_id=?)` — "I will jump
  straight there via the index." Relax.

When you hit a slow query: (1) check with EXPLAIN whether it's a SCAN → (2)
check whether the column in the WHERE has an index. Those two steps are the
fundamentals of performance diagnosis.

### 3-3. The primary key is already an index

Primary keys get an index automatically. That's why `WHERE order_id = 5` is
always fast. The problem children are foreign keys and ordinary columns used
often as search conditions (customer_id, dates, status…) — those are your
index candidates.

### 3-4. When an index is useless

- **Middle-match LIKE**: `LIKE '%Coffee%'` — an index is sorted from the first
  letter, so no. (`LIKE 'Coffee%'` works.)
- **Conditions that transform the column**: `WHERE price * 1.1 > 1000` — the
  transformed value isn't in the index. Rewrite as `price > 1000 / 1.1` and
  the index gets used.
- **Conditions matching most of the table**: if you're fetching half the rows,
  hopping through an index can be slower than just skimming — the DB may
  choose a full scan on its own.

### 3-5. Write cost — the index's invoice

Every added index means extra index-maintenance work on each
INSERT/UPDATE/DELETE. An analysis table that's overwhelmingly read-heavy can
carry generous indexes; a write-heavy order-intake table should carry only the
essential ones — it's a trade-off.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

main.py builds a 300,000-row `big_orders` table in the practice DB for the
experiment (creation takes a few seconds).

- **[1] Set up the test bench**: generate 300,000 rows (fixed seed — same data
  every run).
- **[2] Search without an index**: run `WHERE customer_id = ?` 200 times and
  measure the total elapsed time. EXPLAIN says `SCAN` — every search skims all
  300,000 rows.
- **[3] Create the index**: one `CREATE INDEX` line. Its creation time is
  measured too.
- **[4] The same search after the index**: the same 200 queries re-measured.
  EXPLAIN flips to `SEARCH ... USING INDEX`, and the time drops by a factor of
  tens to hundreds — see the numbers.
- **[5] Searches an index can't help**: a middle-match condition like
  `LIKE '%pp%'` stays a SCAN even with the index — confirmed via EXPLAIN.
- **[6] Write cost**: the same 10,000-row INSERT runs against a table without
  indexes and one with, measuring the write-speed difference.

Absolute times differ by machine, but the **ratio** is always dramatic. Look
for the "x faster" number in the output.

## 5. Try it yourself

1. **(Easy)** Change [2]'s condition to `ordered_at = '2025-06-15'`, confirm
   date search is also a SCAN, then create an index on the date column and
   measure again.
   *Hint: CREATE INDEX idx_big_date ON big_orders (ordered_at).*
2. **(Medium)** Put EXPLAIN QUERY PLAN in front of
   `SELECT * FROM orders WHERE order_id=10` (a primary-key search). You never
   created an index — why is it a SEARCH?
   *Hint: section 3-3.*
3. **(Challenge)** For the two-condition search
   `WHERE customer_id = ? AND status = ?`, compare EXPLAIN output and timings
   between two indexes: (customer_id) and (customer_id, status).
   *Hint: a composite index is CREATE INDEX idx ON big_orders (customer_id, status).*

## 6. Common mistakes

- **Indexing every column**: writes slow down and space gets wasted. Only
  "columns used often in WHERE/JOIN" are candidates.
- **Slow → blame the server**: EXPLAIN first. One full-scan line is the cause
  most of the time.
- **Expecting an effect on tiny tables**: a few hundred rows full-scan in an
  instant, so you'll see no difference. Indexes shine as data grows.
- **Conditions wrapped in functions**: `WHERE SUBSTR(ordered_at,1,7)='2025-06'`
  defeats the index. Rewrite as a range —
  `ordered_at BETWEEN '2025-06-01' AND '2025-06-30'` — and the index gets used.
- **Arguing without measuring**: not "it feels slow" but repeated, measured
  numbers, like this level does.

## Next level preview

Speed handled — time for analysis's final weapon. In **level10 — Window
Functions** you'll learn the analytical queries GROUP BY can't do: "each
customer's nth purchase," "revenue rank within a category," "running totals."
