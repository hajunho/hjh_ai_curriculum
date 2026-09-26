# Lecture 04 · Level 03 — WHERE: Searching with Conditions

> Learn to pull out "only the rows that match" instead of "everything." Comparison operators, AND/OR, IN, LIKE, BETWEEN, and NULL handling — six real business questions translated into SQL.

**Difficulty** ⭐⭐ / **Prerequisites** level02 / **Estimated time** 40 min

## 1. Why this matters — the business view

Almost every data question at work comes with a condition attached: "**VIP**
customers in New York," "**cancelled** orders," "**Q3** revenue," "account
owners whose **last name is Smith**." What used to be clicking the filter
button in Excel and ticking checkboxes is a single WHERE line in SQL. The
difference: an Excel filter has to be clicked again by hand, but a WHERE
statement can be **saved and rerun with the same condition any time**, and
instead of describing the condition in words you can hand a colleague the
statement itself. Answering "what filter did you use?" with one line of query —
that's what precise data communication looks like.

## 2. Understanding by analogy

WHERE is the **third box** on the request form you hand to the clerk.

> From which book (FROM) / which fields (SELECT) / **which rows only** (WHERE)

The clerk flips through the book line by line, asking each row a question:
"Is this customer in New York? And are they a VIP?" — only rows whose answer is
'true' go into the result basket. In other words, what follows WHERE is **a
sentence that evaluates to true/false for every row**.

It plays exactly the same role as the Python if-condition from lecture02.
`if city == "New York" and grade == "VIP":` in SQL becomes
`WHERE city = 'New York' AND grade = 'VIP'` — the only differences are a
single equals sign (=) and single quotes around strings.

## 3. Core concepts

### 3-1. Comparison operators

`=` (equals), `<>` or `!=` (not equal), `>`, `>=`, `<`, `<=`.
Unlike Python, "equals" is a single `=`, not `==`.

```sql
WHERE price >= 100000            -- KRW 100,000 or more
WHERE status <> 'cancelled'      -- everything except cancelled
```

### 3-2. AND / OR / NOT — combining conditions

```sql
WHERE city = 'New York' AND grade = 'VIP'      -- both must hold
WHERE grade = 'VIP' OR grade = 'GOLD'          -- either one is enough
```

AND is evaluated before OR. `A OR B AND C` is read as `A OR (B AND C)`, which
is rarely what you meant — so build the habit of **always adding parentheses
the moment an OR appears**.

### 3-3. IN — one of a list

`grade IN ('VIP', 'GOLD')` is shorthand for `grade='VIP' OR grade='GOLD'`.
The longer the list, the more readable IN becomes. The opposite is `NOT IN`.

### 3-4. LIKE — pattern search

Searching text for "starts with / ends with / contains." `%` is a wildcard
meaning "any 0 or more characters," and `_` means "exactly 1 character."

```sql
WHERE name LIKE '% Smith'     -- ends with " Smith" (last name Smith)
WHERE name LIKE '%Coffee%'    -- product names containing "Coffee"
```

### 3-5. BETWEEN — ranges

`price BETWEEN 10000 AND 100000` is the same as
`price >= 10000 AND price <= 100000`. Remember it **includes both endpoints**.
It works on date strings too, making it the go-to for period queries:
`ordered_at BETWEEN '2025-07-01' AND '2025-09-30'` (Q3).

### 3-6. NULL — handling "no value"

NULL is not zero and not an empty string — it means **"never recorded."**
Ordinary comparisons therefore don't work on it. `manager_id = NULL` is always
false and returns no rows. You must use the dedicated syntax:

```sql
WHERE manager_id IS NULL       -- people with no manager (= the person at the top)
WHERE manager_id IS NOT NULL   -- people who have a manager
```

"Why doesn't = work?" — because NULL means 'unknown,' and the answer to
"unknown = unknown" is also 'unknown.' This is the #1 beginner trap in SQL, so
if you memorize one thing, memorize IS NULL.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

main.py shows six questions the way they'd be asked at work, then prints the
SQL translation and the result of running it.

- **[Q1] "Give me the list of VIP customers in New York"** — `=` and AND.
- **[Q2] "Which products cost KRW 100,000+ or have a margin above 50%?"** —
  OR with parentheses, and a calculation used inside a condition.
- **[Q3] "Customers in New York/Chicago or Boston, but only VIP/GOLD"** — two
  INs combined with AND/OR parentheses.
- **[Q4] "Find customers whose last name is Smith"** — LIKE '% Smith'.
- **[Q5] "Cancelled orders in Q3 (July–September)"** — BETWEEN + AND.
- **[Q6] "Which employee has no manager (top of the org chart)?"** — IS NULL.
  For comparison, the same query mistakenly written with `= NULL` also runs,
  demonstrating **the trap that returns 0 rows**.

For each question, trace how "plain-English question → WHERE condition" lines
up, as if underlining the mapping. The best practice is reading each condition
and judging for yourself, row by row: "is this row true?"

## 5. Try it yourself

1. **(Easy)** Query "BASIC-grade customers living in Boston."
   *Hint: just change the city and grade in Q1.*
2. **(Medium)** Query "products whose name contains 'Coffee' or whose category
   is Stationery."
   *Hint: LIKE '%Coffee%' OR category = 'Stationery'.*
3. **(Challenge)** Query "orders placed in the first half of 2025
   (January–June) whose status is **not** cancelled." Combine BETWEEN with
   `<>`.
   *Hint: ordered_at BETWEEN '2025-01-01' AND '2025-06-30' AND status <> 'cancelled'.*

## 6. Common mistakes

- **`= NULL`**: forever 0 rows. Remember only `IS NULL` / `IS NOT NULL`.
- **Skipping parentheses with OR**: `WHERE city='New York' OR city='Chicago'
  AND grade='VIP'` means "all of New York + Chicago VIPs." If you meant "VIPs
  in New York or Chicago," write
  `(city='New York' OR city='Chicago') AND grade='VIP'`.
- **Comparing numbers as strings**: quoted comparisons like `'100' > 20` mix
  data types and produce nonsense. Compare numeric columns without quotes.
- **LIKE and letter case**: SQLite's LIKE ignores upper/lower case for ASCII
  letters, but other DBs (e.g., PostgreSQL) do not. When results differ on the
  company DB, suspect this first.
- **BETWEEN's end date**: `BETWEEN '2025-07-01' AND '2025-09-30'` includes
  September 30. But on a DB where dates carry times, that can mean "up to Sept
  30, 00:00" — so also keep the `< '2025-10-01'` style in your toolkit.

## Next level preview

You've filtered the rows — now line them up nicely. In **level04 — Sorting,
Removing Duplicates, and Top N** you'll use ORDER BY, DISTINCT, and LIMIT to
build ranking tables like "the 5 most expensive products."
