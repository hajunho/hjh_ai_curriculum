# Lecture 03 · Level 00 — What Is Data? — The Structure of a Table

> We dissect the "table", the starting point of all data analysis. Once you grasp that rows are cases and columns are attributes, Excel and Python start speaking the same language.

**Difficulty** ⭐ / **Prerequisites** none (finishing lecture02 Python basics is plenty) / **Estimated time** 40 min

## 1. Why learn this — the business view

Most of the information that moves through a company ends up organized as a table. Sales reports, customer lists, inventory status, attendance records — the formats differ, but they are all "many cases recorded under the same set of fields". Whether your analysis tool is Excel, Python, SQL, or a BI dashboard, every one of them runs on top of this table structure.

That is why understanding the structure of a table precisely comes before learning any particular tool. If you can answer the question "What does one row in this table mean?", you are already halfway there with any tool. Conversely, if you memorize functions without being able to answer it, you end up producing numbers without knowing what those numbers actually counted. A large share of bad reports at work start exactly here.

In this level we do not use Pandas or NumPy yet. Using only Python's basic ingredients — dictionaries and lists — we handle a table with our bare hands and verify the essence of a table that fancy tools tend to hide.

## 2. Understanding through an analogy

A table is like the **wall of mailboxes in an apartment building**. Each individual mailbox (a value) is pinpointed by two coordinates: "which building, which unit". A table works the same way — with just two coordinates, "which row, which column", you can retrieve any value.

An analogy closer to office life is a **binder of employee record cards**.

- One card = a **row**. It holds one complete "case" — one employee.
- The blanks printed on the card (name, department, hire date...) = the **columns**. Every card has the same blanks.
- The list of blank titles = the **schema**. It is the agreement about "which fields our cards record, and in what format".

What matters is the binder's rule. If some cards have a "hire date" blank and others do not, you cannot sort the whole binder by hire date. The power of a table comes from the promise that **every row shares the same column structure**. Data that keeps this promise is called structured data; free-form records without it (say, a review sentence a customer left) are called unstructured data.

## 3. Core concepts

### 3-1. Row = case, column = attribute

- **Row (record)**: one observed case. One thing that happened in the world, like "the daily revenue of the Coffee category at the Downtown store on January 3, 2025".
- **Column (attribute/variable)**: a field recorded for every case — date, store, category, revenue amount. In statistics it is also called a variable.

Whenever you receive a table, there is one question to ask first: **"What is one row of this table, exactly?"** Even for the same sales data, a table where a row is "one transaction" must be treated completely differently from one where a row is "a daily total per store and category". This unit is sometimes called the grain.

### 3-2. Schema — the agreement on column names and types

A schema fixes two things.

1. The **names** of the columns: which fields exist, like `date`, `store`, `revenue`.
2. The **type** of each column: is the field a string, an integer, a date?

Types matter because operations behave differently depending on them. `"100" + "200"` is string concatenation and gives `"100200"`, while `100 + 200` gives `300`. This is a classic cause of real-world incidents.

### 3-3. Structured vs unstructured

| Aspect | Structured data | Unstructured data |
|---|---|---|
| Shape | A table with fixed rows/columns | Free text, images, audio |
| Examples | Sales ledger, member list | Customer reviews, call recordings, photos |
| Aggregation | `SUM` and averages work immediately | Structure must be extracted first |

Unstructured is not inferior — if anything it carries more information. It just needs **to be converted into a table before you can aggregate it** (e.g., review → a positive/negative column). A later lecture (natural language processing) covers that conversion.

### 3-4. The humblest way to represent a table in Python

A list of dictionaries.

```python
table = [
    {"store": "Downtown", "revenue": 500000},
    {"store": "Riverside", "revenue": 420000},
]
```

- One element of the list = one row
- The dictionary's keys = the column names

Pandas' DataFrame is ultimately this very structure, made far faster and more convenient. If you handle the raw form with your own hands first, the Pandas you meet from level03 onward looks like a "convenience feature" rather than "magic".

### 3-5. Missing values — an empty cell is information too

Real data always has cells where a value never got recorded. Python represents this with `None`. "0" and "empty" are completely different things. Revenue of 0 is the fact "nothing was sold"; `None` is the state "we don't know how much was sold". Level06 treats this distinction in earnest, but from the very first time you look at a table, it pays to build the habit of counting how many cells are empty.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level00_what_is_data
python3 main.py
```

The script builds 30 days of sales for a fictional cafe chain (750 rows) and performs the following.

- **[1] Table size**: prints the number of rows (cases) and the column names (schema). `rows[0].keys()` is exactly the column list.
- **[2] Preview**: looks at the top of the table with `hjh_data.head`. When you get a table, eyeballing it always comes first.
- **[3] Dissecting one row**: pulls out the first row and confirms that "one row of this table = one day's revenue for a specific date, store, and category".
- **[4] Extracting a column**: uses a list comprehension to pull only the `revenue` column, showing that a column is "a collection of values of the same attribute".
- **[5] Finding rows by condition**: picks out and counts only Downtown's Coffee rows. This is the prototype of a filter.
- **[6] Mini schema summary**: counts, by hand, the types that appear in each column plus the missing (`None`) count. It is essentially a hand-made version of what Pandas' `info()` does for you.
- **[7] Structured vs unstructured**: compares against the same information written as a free sentence (unstructured) and confirms you cannot run `SUM` on a sentence.

In the output, pay special attention to how many `None` values are in the `revenue` column, and how many negative revenues are mixed in. They are deliberately planted contamination, and you will keep running into them in later levels.

## 5. Try it yourself

1. **(Easy)** Using [5] as a reference, count the rows for "Dessert at the Riverside store". Hint: connect two conditions with `and`.
2. **(Medium)** Keep only the rows where `revenue` is not `None`, and compute the total revenue. Hint: think of the skeleton `sum(r["revenue"] for r in rows if r["revenue"] is not None)` — but decide for yourself what to do about negatives.
3. **(Challenge)** Add "min/max" entries to the schema summary in [6]. Hint: it should apply only to numeric columns, and you can use `min()`/`max()` after filtering out `None`.

## 6. Common mistakes

- **Computing totals before checking what a row means**: in this table, `sum(revenue)` is "the total of per-category daily revenues". If you mistakenly assume a row is a transaction, you produce the wrong report "750 transactions".
- **Mixing `None` and 0**: calling `sum()` directly on a list containing `None` raises a `TypeError`. That error is a blessing. Silently replacing `None` with 0 distorts the average.
- **Typos in column names**: writing the key wrong, like `rows[0]["revenu"]`, raises a `KeyError`. Always check column names with `keys()` first.
- **Judging types by eye**: what looks like `123` on screen may actually be the string `"123"`. Build the habit of checking with `type()`.

## Next level preview

Now that you know what a table is, next we **move "the things you did in Excel" into code**. While reproducing sums, filters, and sorting in Python, you will see how code overcomes the limits Excel hits: reproducibility, scale, and automation.
