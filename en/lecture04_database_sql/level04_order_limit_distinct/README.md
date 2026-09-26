# Lecture 04 · Level 04 — Sorting, Removing Duplicates, and Top N

> Line results up with ORDER BY, cut off the top N with LIMIT, and sweep out duplicates with DISTINCT. Build ranking tables like "the 5 most expensive products" in a single SQL sentence.

**Difficulty** ⭐⭐ / **Prerequisites** level03 / **Estimated time** 35 min

## 1. Why this matters — the business view

Most tables in reports are **ranked tables**: "top 10 stores by revenue,"
"the 20 most recent orders," "the 3 cheapest quotes." And a question that comes
up in every meeting — "which cities are our customers in, anyway?" — is really
asking for a **de-duplicated list**. With just three tools — ORDER BY
(sorting), LIMIT (top N), DISTINCT (duplicate removal) — most such requests
end in one sentence. The "top N" pattern in particular is one of the shapes
you'll type most often in day-to-day SQL, so it's worth getting it fully into
your fingers.

## 2. Understanding by analogy

Picture the clerk's workflow. Following the request form, they pick the
matching rows out of the book and put them in a basket (FROM + WHERE), then
apply three finishing touches at the end:

- **ORDER BY = lining up**: the rows in the basket get sorted by the requested
  criterion (by price, by date…) and laid on a tray. Ties get re-sorted by the
  second criterion — like a school lineup "by height, and alphabetically when
  heights match."
- **LIMIT = taking only the top few**: from the sorted tray, only the top N
  cards get handed over. Cut without sorting and you get "just any 5 cards,"
  which is why **top N always pairs with ORDER BY**.
- **DISTINCT = discarding duplicate cards**: when several cards say the same
  thing, only one is kept. Use it when the question is about **kinds** — like
  "the list of cities our customers live in."

The important part: all of this is **work on the result basket**. The original
book never changes.

## 3. Core concepts

### 3-1. ORDER BY — ascending and descending

```sql
SELECT name, price
FROM   products
ORDER BY price DESC;   -- most expensive first (DESC = descending)
```

The default is ascending (ASC, can be omitted). Sorting a date column DESC
gives you "newest first." Text columns sort alphabetically.

### 3-2. Multiple sort criteria

```sql
ORDER BY city ASC, grade DESC
```

First alphabetically by city, then within the same city by grade in reverse.
The comma order is the priority order. Aliases and expressions work as
criteria too: `ORDER BY price - cost DESC` (biggest margin first).

### 3-3. LIMIT and OFFSET

`LIMIT 5` returns only the first 5 rows; `LIMIT 5 OFFSET 5` returns rows 6–10.
A website's "page 2" is exactly OFFSET. Note: **LIMIT doesn't reduce the
computation, only how much is shown.** The sort happens over everything, and
then the top slice is cut off.

### 3-4. DISTINCT — questions about kinds

```sql
SELECT DISTINCT city FROM customers;
```

Pull just the city value from 200 customer rows and it's full of repeats;
DISTINCT keeps each value once. With two columns it de-duplicates the
**combination** `(city, grade)` — as in "which grade combinations exist per
city." Counting "how many kinds?" with `COUNT(DISTINCT city)` arrives in the
next level (aggregation).

### 3-5. Where each clause goes (word-order rules)

SQL clauses have a fixed order:

```
SELECT [DISTINCT] column_list
FROM   table_name
WHERE  condition
ORDER BY criterion
LIMIT  N;
```

The execution order is FROM → WHERE → SELECT → ORDER BY → LIMIT. Remember it
as "filter (WHERE), shape the columns (SELECT), line up (ORDER BY), cut
(LIMIT)."

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

- **[1] Top 5 most expensive products**: `ORDER BY price DESC LIMIT 5` — the
  basic ranking table.
- **[2] Top 3 by margin**: the expression `price - cost` as the sort key.
- **[3] 5 most recent orders**: date column DESC — the "recent activity"
  pattern.
- **[4] Multi-key sort**: customers alphabetically by city → then by grade
  within the same city.
- **[5] DISTINCT**: the list of cities customers actually live in — watch 200
  rows collapse to 6. Then a (city, grade) combination DISTINCT runs as well.
- **[6] Paging**: the same ranking query with `LIMIT 5 OFFSET 5` pulls out
  "page 2."
- **[7] Trap demo**: with LIMIT 5 but no ORDER BY, you get "just the top 5 rows
  as stored," not a top 5 — see it with your own eyes.

Be sure to compare the outputs of [1] and [7]. Same LIMIT 5, completely
different meaning.

## 5. Try it yourself

1. **(Easy)** Build a "3 cheapest products" table.
   *Hint: change DESC to ASC (or drop it entirely).*
2. **(Medium)** Sort employees by salary, highest first, top 5 only, breaking
   salary ties alphabetically by name.
   *Hint: ORDER BY salary DESC, name ASC LIMIT 5.*
3. **(Challenge)** List every **kind** of status that exists in the orders
   table, and guess what each one means.
   *Hint: SELECT DISTINCT status FROM orders — there are 3 kinds.*

## 6. Common mistakes

- **"Top N" without ORDER BY**: LIMIT alone just returns N rows in whatever
  order the DB finds convenient (effectively random). Top N = ORDER BY +
  LIMIT, no exceptions.
- **Rankings missing DESC**: pulled "top 5 by revenue" and got the bottom 5?
  Nine times out of ten you dropped DESC. Remember the default is ascending.
- **Sticking DISTINCT anywhere**: DISTINCT appears once, right after SELECT,
  and applies to **the whole combination of selected columns**.
  `SELECT DISTINCT city, name` doesn't mean "de-duplicate city only" — it
  de-duplicates (city, name) pairs.
- **Sorting numbers stored as text**: if prices are stored as TEXT you get
  dictionary order like '9' > '1200000'. When a sort looks wrong, check the
  data type first.
- **Shuffling clause order**: `LIMIT 5 ORDER BY price` is a syntax error. Keep
  the order WHERE → ORDER BY → LIMIT.

## Next level preview

So far we've been picking rows and listing them. Next comes, at last,
**summarizing**. In **level05 — Aggregate Functions and GROUP BY** you'll do
what Excel pivot tables did — "customers per city," "revenue per category" —
in SQL.
