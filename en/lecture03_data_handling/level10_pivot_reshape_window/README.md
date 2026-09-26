# Lecture 03 · Level 10 — Pivoting, Reshaping, and Window Operations

> Reproduce the Excel pivot table in code, reshape tables at will, and finish the report with ranks, growth rates, and running totals.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level07 (level09 recommended) / **Estimated time** 60 min

## 1. Why learn this — the business view

Most tables that end up in a report are two-dimensional summaries of the form "something × something": store × category revenue, month × product sales, team × quarter results. If you are an Excel user, that is the table you have built countless times with a pivot table.

Pandas' `pivot_table` does exactly what Excel's pivot table does, with three differences. First, even a million rows finish in seconds. Second, it lives on as code, so next month is one re-run instead of repeated button clicks. Third, it comes as a set with `melt`, which turns a finished pivot back into an analyzable shape. Add window operations like rank, pct_change, and cumsum on top, and every staple number of a management report — "store rankings", "running progress against target" — falls out.

## 2. Understanding through an analogy

**Reshaping data is like organizing a closet.** The same clothes serve different purposes "folded one per drawer" versus "hung in a row on the rail".

- **Long format** is everything hung on the rail. One row records one event: "January 1, Downtown, Coffee, 450 thousand KRW". It is the machine-friendly shape — filtering, grouping, and merging all happen here.
- **Wide format** is the clothes folded into a dresser. Rows are stores, columns are categories, cells hold the totals — a summary table. It is the human-friendly shape, used for reports and slide decks.
- `pivot_table` is folding from the rail (long) into the dresser (wide); `melt` is taking them back out of the dresser (wide) onto the rail (long). Analyze in long, report in wide — that round trip is the working rhythm.

Ranking is a race podium. The times (revenue) already exist; `rank` is the judge who converts them into places. There is the overall ceremony (company-wide ranks) and the heat-by-heat ceremony (ranks within each category group).

## 3. Core concepts

### 3-1. The four ingredients of pivot_table

```python
pd.pivot_table(df, index="store", columns="category",
               values="revenue", aggfunc="sum", margins=True)
```

| Argument | Name in Excel's pivot table | Meaning |
|---|---|---|
| `index` | Rows area | what goes on the vertical axis |
| `columns` | Columns area | what goes on the horizontal axis |
| `values` | Values area | the numbers that fill the cells |
| `aggfunc` | value summary method | sum, mean, count, ... |

Pass `margins=True` and you get Excel's "Grand Total" row and column. When several rows share the same (index, columns) combination, `aggfunc` folds them together — so duplicates never cause an error (unlike `pivot`, which errors on duplicates).

### 3-2. melt — melting the pivot back down

`melt` turns a wide table back into long. Specify `id_vars` (the identifier columns to leave alone) and the remaining columns melt down into two: "variable name (variable) / value (value)". Why go back? Because wide tables please the human eye, but filters, groupby, and most charting libraries demand long. When you "received a report-style table but need to analyze it again", melt is the relief pitcher.

### 3-3. rank — assigning places

`s.rank(ascending=False)` ranks so the largest value takes first place. Tie handling is set with `method` (the default is average rank; `method="min"` is the shared-first-place style). What you really use constantly at work is the **within-group rank**.

```python
df.groupby("category")["revenue"].rank(ascending=False)
```

This yields heat-level places, like "where does Downtown rank within the Coffee segment?". It is the signature pattern combining groupby (level07) with a window operation.

### 3-4. pct_change and cumsum — reading the flow

- `monthly.pct_change()`: the rate of change versus the previous row. The level09 operation applied to a monthly table becomes "month-over-month growth".
- `monthly.cumsum()`: the running total. Revenue accumulated from January to this month — the numerator when computing "how far along are we against the annual target?".

Operations like these are collectively called window operations. Where aggregation (groupby) "reduces" many rows to one, a window operation keeps the row count and attaches to each row "a value informed by its neighbors". That difference matters.

### 3-5. Wide vs long — when to use which

| Situation | Format |
|---|---|
| Analysis work: filtering, grouping, merging | long |
| Input to charting libraries | mostly long |
| Report/dashboard tables for humans | wide |
| Handing a CSV to another team | long recommended (re-analyzable) |

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level10_pivot_reshape_window
python3 main.py
```

The output flows as follows.

- **[1] Store × category revenue pivot** — builds the annual report with `pivot_table` and `margins=True` for the grand totals. Values print rounded in 100M-KRW units, so you can see at a glance which category earns its keep at which store.
- **[2] Weekday × store mean revenue pivot** — switch to `aggfunc="mean"` and the same function answers a completely different question ("does the weekday effect differ by store?"). Also note how the weekday rows, scrambled into alphabetical order, get restored to Mon–Sun with `reindex`.
- **[3] Back with melt** — melts the pivot from [1] into long format, confirming that 25 wide cells become 25 long rows, and compares what each format is for.
- **[4] Rankings** — computes the company-wide store ranking (`rank`) and the within-category store ranks (`groupby` + `rank`). Check whether the overall #1 store is #1 in every category.
- **[5] The monthly flow** — builds the monthly management summary with `pct_change` (month-over-month growth) and `cumsum` (running total) attached.
- **[6] Saving the report** — saves the finished pivot report to `outputs/pivot_report.csv`. Open it in Excel and you will see the table you used to craft by hand, produced by a single run of code.

## 5. Try it yourself

1. **(Basic)** In the pivot of [1], change `aggfunc="sum"` to `"count"`. What does a cell now mean instead of "total revenue"? Hint: it becomes how many rows went in — the number of sales records.
2. **(Applied)** Build a two-level pivot by passing a list to index, like `index=["store", "weekday"]`. Hint: the result is a MultiIndex table — the same as "two fields in the Rows area" of an Excel pivot.
3. **(Challenge)** For each category, compute the "revenue gap between stores" (within-group max − min) and find the category with the widest gap. Hint: apply `groupby("category")["revenue"].agg(lambda s: s.max() - s.min())` to the category-store totals built in [4].

## 6. Common mistakes

- **Confusing pivot with pivot_table** — `pivot` errors when a (row, column) combination repeats. If the source has duplicate combinations, as daily data does, always use `pivot_table` (aggregation built in).
- **Treating the margins row as data** — the "All"/"Total" row and column added by `margins=True` are totals, not data. Before melting a pivot or ranking it, use the margins-free version.
- **Leaving weekdays/months in alphabetical order** — a pivot's row order defaults to lexicographic, giving you "Fri Mon Sat Sun Thu Tue Wed". Restore the business order with `reindex(["Mon","Tue",...])`.
- **Getting rank's direction backwards** — the default `ascending=True` makes the smallest value #1. Revenue rankings almost always need `ascending=False`.
- **Reading NaN cells as 0** — in a pivot, NaN means "no data for that combination", not "zero revenue". Only `fillna(0)` after confirming the meaning.

## Next level preview

Everything so far assumes the data fits comfortably in memory. In the final level11, we learn the survival strategies for when data grows to millions of rows — dtype dieting, chunked processing, categoricals, Parquet — and settle the judgment call of "when to leave pandas for a database or Spark".
