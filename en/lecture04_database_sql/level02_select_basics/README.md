# Lecture 04 · Level 02 — SELECT Basics

> Learn SELECT — SQL's first sentence and the start of everything. Pick just the columns you want from a table, give columns friendlier names with aliases, and create calculated columns.

**Difficulty** ⭐ / **Prerequisites** level01 / **Estimated time** 35 min

## 1. Why this matters — the business view

More than 90% of real-world SQL is **reading**. Systems do the writing; people
mostly ask "show me." That "show me" is SELECT. Get comfortable with just this
one sentence and the task that used to mean asking another team to "pull the
customer list" and waiting two days becomes a 30-second self-service job.
Better yet, SELECT only reads data and never changes it, so it is a perfectly
safe sentence for a beginner to practice with as much as they like.

## 2. Understanding by analogy

A SELECT statement is **the request form you hand to the clerk at the
counter**. Continuing the stationery-shop ledger analogy, the form has two
boxes:

> **From which book** (FROM customers — from the customer register)
> **Which fields** (SELECT name, city — just the name and city fields)

The one thing to watch is that the reading order is backwards from how you'd
say it. SQL is written "SELECT name, city FROM customers," but the clerk
processes it as "from the customer register (FROM) → give me name and city
(SELECT)." Build the habit of **finding the FROM first** when you read SQL,
and even long queries stop being scary.

An alias (alias) is the form's "please label it this way on the report" box.
Write `price AS unit_price` and the column header in the result changes to
unit_price. The data itself is untouched — only the **displayed name** changes.

## 3. Core concepts

### 3-1. The basic shape

```sql
SELECT column1, column2   -- what (the column list)
FROM   table_name;        -- from where
```

- Case doesn't matter, but by convention keywords (SELECT, FROM…) are
  uppercase and table/column names are lowercase. It makes them easy to spot.
- The semicolon (;) at the end means "end of request form."
- Anything after `--` is a comment and is not executed.

### 3-2. All columns: `SELECT *`

`*` means "everything." Handy when you first explore a table, but real
production queries should list only the columns they need. On a table with
dozens of columns, `*` wastes screen space, and when a column gets added later
the shape of your result silently changes. "Skim with *, write the real query
with column names" — that's the working rhythm.

### 3-3. Aliases (AS)

```sql
SELECT name AS product_name, price AS unit_price
FROM   products;
```

AS can be omitted (`price unit_price`), but while you're learning it's easier
to read when spelled out. An alias changes only the column header in the
result; the table's actual column name never changes.

### 3-4. Calculated columns (expressions)

The SELECT list can hold not just column names but **calculations**.

```sql
SELECT name,
       price,
       cost,
       price - cost AS margin
FROM   products;
```

`price - cost` is computed for every row and shows up like a new column. It's
the same as filling column D of a spreadsheet with `=B2-C2`, except one
sentence applies to every row — no copy-pasting formulas. String
concatenation with `||` and functions like `ROUND()` go in the same spot.

### 3-5. An early hello to LIMIT

Results with hundreds of rows overflow the screen, so from this level on we
append `LIMIT 5` (first 5 rows only) to keep the output short. The formal
explanation comes in level04; for now just think of it as a "preview knob."

### 3-6. SELECT never changes the original

A SELECT result is a **display copy** of the original table. Whether you add
aliases or build calculated columns, not one byte of data inside hjh_shop.db
changes. That's why practicing queries is safe no matter how badly you fumble.
The statements that really change data (INSERT, UPDATE, DELETE) are taught
carefully in level08, together with transactions.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

main.py rebuilds `hjh_shop.db` each time, prints the SQL statement first, and
shows the result table right below it. **The lines starting with SQL> are the
main event**; the Python code is just the runner that delivers them.

- **[1] Skim all columns**: `SELECT * FROM customers LIMIT 5` — a first
  hello to the table. Check the columns and what the data looks like.
- **[2] Only the columns you need**: `SELECT name, city FROM customers LIMIT 5`
  — the basic form of naming fields on the request.
- **[3] Adding aliases**: prints products as a report-style table with
  headers like product_name / product_category / unit_price.
- **[4] Calculated columns**: `price - cost AS margin`, and
  `ROUND(100.0*(price-cost)/price, 1) AS margin_pct`, computed per row.
- **[5] String building**: `name || ' (' || grade || ')'` builds a display
  column like "James Smith (VIP)". A pattern you'll use constantly for things
  like mail merges.

At each step, practice reading the SQL by finding FROM first → then the
SELECT list. Also confirm that the result table's headers change to exactly
your aliases.

## 5. Try it yourself

1. **(Easy)** Mimic step [2] and add a query that selects only name and dept
   from the employees table.
   *Hint: change only the table name in FROM and the column names in SELECT.
   Column names are in level01's step [2] output.*
2. **(Medium)** In products, build a "price for a bundle of 10" column titled
   `bundle_price`.
   *Hint: `price * 10 AS bundle_price`.*
3. **(Challenge)** From customers, build a one-column table shaped like
   `Emma Johnson from New York`.
   *Hint: `name || ' from ' || city` — inside single quotes is literal text,
   outside is a column name.*

## 6. Common mistakes

- **Double quotes around text**: in SQL, text data is wrapped in single quotes
  (`'New York'`). Double quotes are for column/table names, and mixing them up
  produces confusing errors.
- **Column name typos**: `SELECT nmae` gives a "no such column" error. The
  error message contains all the hints you need — read it, don't panic.
- **Missing/stray commas after SELECT**: `SELECT name city` is not an error —
  city gets interpreted as an **alias** for name and you get a bizarre result.
  Always double-check the commas.
- **Using `SELECT *` in a real query**: skimming only. Any query you'll save
  and reuse should spell out its columns.

## Next level preview

Right now every row of the table comes back. Most real questions are "only the
ones **that match a condition**." In **level03 — WHERE** you'll learn
conditional searches like "only VIP customers in New York" and "only cancelled
orders."
