# Lecture 03 · Level 03 — Pandas — Series and DataFrame

> We dissect the DataFrame — the tool that puts an entire Excel sheet into a single Python variable.

**Difficulty** ⭐⭐ / **Prerequisites** level02_numpy_basics / **Estimated time** 40 min

## 1. Why learn this — the business view

The standard tool of practical data analysis is Pandas. Opening the sales CSV, applying filters, aggregating by store, building the table for the report — the DataFrame is there at the start and the end of all of it. Every remaining level of this lecture (loading, cleaning, aggregation, merging, time series, pivoting) plays out on top of a DataFrame. In other words, this level is "the session where we assemble the workbench you will use from now on".

If you look at the next levels' code without knowing the DataFrame, expressions like `df.loc`, `df.dtypes`, and `df["revenue"]` read like cipher. Understand the structure properly once, and even a Pandas feature you have never seen becomes guessable: "ah, in the end this is just handling a table whose rows and columns wear name tags". When a job posting says "Pandas proficiency preferred", the minimum bar it refers to is exactly the content of this level.

## 2. Understanding through an analogy

Picture an Excel workbook.

- A **Series** is **one column** of the sheet. Imagine tearing off just the "revenue" column — with one important difference: every value carries a **row label (index)** along with it. Copy column D in Excel and the positional information "this was row 3" is lost; a Series holds onto its labels to the very end.
- A **DataFrame** is **one full sheet** made by placing several Series side by side. All columns share the same row labels (index), and each column wears a column name (columns).

Another analogy is a **filing cabinet**. A DataFrame is a set of folders with labeled tabs (column names), and each document (row) is stamped with a reference number (index). Being able to say "show me document number 100 from the revenue folder" is what the labeling system buys you. Excel points with the eyes and the mouse; Pandas speaks in labels. That difference is what makes automation possible.

## 3. Core concepts

### 3.1 Series — a labeled 1-D bundle of values

```python
s = pd.Series([310, 250, 480], index=["Downtown", "Riverside", "Airport"])
s["Airport"]   # 480
```

A Series is a NumPy array dressed in an index of labels. So vectorized operations like `s.mean()` and `s * 1.1` work exactly as in NumPy, while at the same time you can call values by name, like `s["Airport"]`.

### 3.2 DataFrame — a bundle of Series

The three parts of a DataFrame:

| Part | Excel counterpart | How to inspect |
|---|---|---|
| values | the cell contents | `df.values` |
| row labels (index) | row numbers 1, 2, 3... | `df.index` |
| column names (columns) | the header row | `df.columns` |

Pull a column out by name, as in `df["revenue"]`, and you get that column as a Series. Remember the round trip: "take one column out of a DataFrame → a Series; gather several Series → a DataFrame".

### 3.3 dtype — one data type per column

Each column carries exactly one data type (dtype). Numeric columns show `int64` or `float64`; text columns show `str` in recent Pandas versions and `object` in older ones. One practically important fact: **mix missing values (NaN) into a numeric column and even integers become float64.** When revenue shows up with a decimal point, like `758123.0`, get in the habit of suspecting "there must be an empty cell somewhere".

### 3.4 The four-part greeting — head / info / describe / shape

Whenever you meet new data, always say hello in this order.

1. `df.shape` — how many rows and columns (grasp the scale)
2. `df.head()` — preview the first 5 rows (grasp the look)
3. `df.info()` — dtype and non-null count per column (the health check)
4. `df.describe()` — mean, std, min, max of numeric columns (skim the distribution)

In particular, a negative number in `describe()`'s min, or an abnormally large max, is a signal of outliers.

### 3.5 Creating a new column

```python
df["roas"] = df["revenue"] / df["ad_cost"]
```

The Excel routine of typing `=C2/D2` into column E and dragging down finishes in one line. It is a vectorized operation, so even 100,000 rows take an instant.

### 3.6 Counting categories — value_counts

To see the composition of a text column, use `df["store"].value_counts()`. What used to be one COUNTIF per store in Excel becomes a single line.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level03_pandas_dataframe
python3 main.py
```

main.py turns 90 days of cafe-chain sales (`hjh_data.sales_table`) into a DataFrame and dissects the structure piece by piece.

- **[1]** Converts the list of dictionaries into a table with the single line `pd.DataFrame(rows)`. The table you wrestled with using lists and loops early in the lecture materializes instantly.
- **[2]~[3]** Prints `shape`, `index`, `columns`, and `dtypes`. See with your own eyes why revenue is float64 (it contains missing values).
- **[4]~[6]** Performs the head / info / describe greeting. You should spot negative revenue in describe's min — the first clue that the data is contaminated.
- **[7]** Pulls out `df["revenue"]`, confirms its type is Series, and computes summaries like `.mean()` and `.max()` directly.
- **[8]** Builds the derived column `roas` (revenue over ad spend) and looks at the top rows.
- **[9]** Checks store and category composition with `value_counts()`. Since the data is synthetic, the counts are exactly uniform — another point worth confirming.
- **[10]** Counts missing values per column with `isna().sum()`. Handling comes in earnest in level06; here we stop at "knowing how many there are".

What to look for in the output: any column whose `Non-Null Count` in info() is smaller than the total row count is a column with missing values.

## 5. Try it yourself

1. **(Basic)** Print the mean and max of `df["ad_cost"]`. Hint: a Series also has `.mean()` and `.max()`.
2. **(Applied)** Add a `revenue_manwon` column with revenue in units of 10,000 KRW and check it with `head()`. Hint: divide the whole column by 10000.
3. **(Challenge)** Print `df["weekday"].value_counts()` and explain why the counts look the way they do, connecting it to how the data is generated (90 days × 5 stores × 5 categories). Hint: 90 is not divisible by 7 — compute `90 % 7` and you will find the number of days actually differs by weekday. Compare with the real output.

## 6. Common mistakes

- **Confusing `df["revenue"]` with `df[["revenue"]]`** — single brackets give a Series, double brackets give a one-column DataFrame. The methods you can chain afterward differ.
- **Relaxing after only describe()** — describe summarizes numeric columns only. Typos in text columns (like `"Downtown "` with a trailing space) must be checked separately with value_counts.
- **Computing without checking dtype** — if something looks numeric but its dtype is object (text), a sum becomes string concatenation or an error. Checking `df.dtypes` before computing is non-negotiable.
- **Trying to store the result of info() in a variable** — `info()` only prints to the screen and returns None. If you need the values, use `df.dtypes` and `df.isna().sum()`.

## Next level preview

So far we generated the data inside Python code, but real-world data arrives as CSV, Excel, and JSON files. In the next level we learn the key options of `read_csv` — and the fix for the encoding problems (like legacy cp949 files from Korean systems) that torment office workers everywhere.
