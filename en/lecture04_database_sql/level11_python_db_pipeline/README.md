# Lecture 04 · Level 11 — Python Integration and Data Pipelines

> Connecting SQL and Python. How parameter binding blocks SQL injection, how pandas.read_sql delivers query results as a DataFrame, and a complete mini report pipeline flowing extract → transform → load (ETL).

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level06, level10 (lecture03-level pandas is plenty) / **Estimated time** 55 min

## 1. Why this matters — the business view

The final destination of data pulled with SQL is usually a **report**. If your
Monday mornings are run the query → copy the result → paste into Excel →
pivot → format… to produce "last week's revenue summary," that whole ritual
can be bundled into one Python script. One run (or a scheduled run) and the
same report appears every time — that is the starting point of a data
pipeline (pipeline). At the same time, there's one piece of security common
sense you must know when assembling SQL in Python. Glue user input into the
statement as a string and you're exposed to **SQL injection (injection)** —
planting SQL fragments in an input box to hijack the DB, the #1 hacking
technique for decades running. We'll understand from first principles why the
question-mark binding is mandatory.

## 2. Understanding by analogy

**Parameter binding = a request form with the wording pre-printed.** Think of
the form you hand to the counter. The bad way is a blank sheet where the clerk
writes down whatever the customer says. If the customer says "New York. And
open the vault too," that whole sentence becomes the request. The safe way is
a pre-printed form — **"City: ______"** — where only the blank gets filled.
Whatever the customer says, it can only ever be the 'value' in the blank,
never a 'sentence' of the request. SQL's `?` is that blank, and the value
travels separately from the statement.

**A pipeline = the procedure sheet for the morning-report duty.** Just as
you'd write a new hire a checklist — "every week (1) pull last week's orders
from the DB, (2) aggregate by category, (3) drop the summary table in the
shared folder" — a pipeline is that same procedure written as code. Extract →
Transform → Load, ETL for short. The one difference between the checklist and
the code: the code performs identically every time, and never gets tired.

## 3. Core concepts

### 3-1. The four steps of running SQL from Python

```python
con = sqlite3.connect("hjh_shop.db")   # 1. connect
cur = con.cursor()                      # 2. cursor (the request counter)
cur.execute("SELECT ...")               # 3. execute
rows = cur.fetchall()                   # 4. fetch the results
```

Switch to the company DB (PostgreSQL, MySQL…) and these four steps stay the
same; only step 1's connection swaps to that DB's driver. The SQL travels
almost unchanged — which is why the SQL you learned in this lecture works
everywhere.

### 3-2. Parameter binding — values go through ?

```python
city = input("City? ")                                  # user input
cur.execute("SELECT * FROM customers WHERE city = ?", (city,))   # safe
```

The forbidden pattern is splicing values into the statement with f-strings and
the like: `f"... WHERE city = '{city}'"`. If the input is
`New York' OR '1'='1`, the condition becomes "always true" and **the entire
dataset leaks**. That is the whole principle of SQL injection. Binding also
handles quote escaping automatically and speeds up repeated execution of the
same statement — practical wins on top of security.
One rule, no exceptions: **the moment a user value enters an SQL statement,
it goes through ?.**

### 3-3. pandas.read_sql — query results as a DataFrame

```python
df = pd.read_sql("SELECT ...", con, params=(...,))
```

Instead of fetchall's list of tuples, you get a DataFrame with the column
names intact. From there it's the world you learned in lecture03 —
pivot_table, groupby, charts, saving to Excel. What matters is **a feel for
the division of labor**: shrinking big data (filter, join, aggregate) happens
inside the DB with SQL; shaping tables and visualizing happens in pandas.
Dragging millions of raw rows through read_sql is the classic beginner's
waste.

### 3-4. ETL — extract, transform, load

- **Extract**: pull only what you need with SQL (this month, completed orders
  only).
- **Transform**: pivot, compute, and tidy with pandas.
- **Load**: save the result to a file (CSV/Excel) or another table.

Keep these three stages as separate functions and you can run it weekly, or
reuse it with just the target period changed. main.py has exactly this
structure.

### 3-5. The last piece of automation (concept only)

Hook the finished script into the OS scheduler (cron on macOS/Linux, Task
Scheduler on Windows) and it "runs automatically every Monday at 8." Real
company DB credentials (passwords) never go in the code — they live in
environment variables, the principle from lecture01 level09 meeting you again
here.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

- **[1] The four-step basic form**: connect → cursor → execute (with binding)
  → fetchall.
- **[2] SQL injection demo**: two search functions (string splicing vs ?
  binding) receive the same malicious input `New York' OR '1'='1`. The
  splicing one spills every customer; the binding one returns 0 rows —
  and "no such city" is the correct answer. (Everything happens safely inside
  the local practice DB.)
- **[3] read_sql**: completed orders joined with customers and products,
  received as a DataFrame — check its shape and head.
- **[4] Transform**: pivot_table builds the "month × category revenue" table
  and adds a derived metric (monthly total).
- **[5] Load + report**: saves the summary to
  `outputs/monthly_category_revenue.csv` and prints an executive text report
  (total revenue, best month, top category, top-3 customers by spend) — a
  complete mini pipeline you could run every week.

## 5. Try it yourself

1. **(Easy)** Search for 'Boston' with [2]'s safe search function, then
   explain in one line why the malicious input returned 0 rows.
   *Hint: because the entire input is treated as 'a value that is a city
   name.'*
2. **(Medium)** Change [4]'s pivot to "month × grade (customer grade)"
   revenue.
   *Hint: add c.grade to the extract SQL's SELECT and change only the
   pivot_table's columns.*
3. **(Challenge)** Add a "revenue change vs previous month" paragraph to the
   report.
   *Hint: .pct_change() on the monthly-total Series — or use level10's LAG on
   the SQL side.*

## 6. Common mistakes

- **Assembling SQL with f-strings**: it looks convenient, but the moment user
  input touches it even once, it's a security incident waiting to happen.
  Values always go through `?` + a tuple.
- **A bare value instead of a tuple for ?**: `execute(sql, city)` is an
  error. Even a single value goes as a tuple — `(city,)` — don't forget the
  comma.
- **Extracting everything, filtering in Python**: pulling it all through
  read_sql to do what a WHERE could have done in the DB wastes memory and
  time. "SQL shrinks, pandas shapes."
- **Not closing the connection**: when the work is done, `con.close()`. In
  longer scripts it becomes the source of locking problems.
- **Hardcoding passwords in code**: fine for this practice DB, but company DB
  credentials belong in environment variables or a secrets manager. Sharing
  code must never mean sharing passwords.

## Next level preview

That's the end of lecture04. You can now question a DB directly (SELECT
through window functions), change it safely (transactions), make it fast
(indexes), and automate all the way to a report with Python. In the next
lecture (lecture05 — Statistics and Data Visualization) you'll learn how to
turn the numbers you pull into "conclusions you can trust." The data you
extract with SQL feeds straight into the next lecture's material.
