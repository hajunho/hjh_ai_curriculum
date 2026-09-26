# Lecture 03 · Level 07 — Grouping and Aggregation (groupby)

> "Revenue by store? Average by weekday?" — we learn groupby, the standard answer to real-world questions.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level06_missing_outliers / **Estimated time** 40 min

## 1. Why learn this — the business view

The questions a boss asks while looking at data almost always have the same shape: "How does it look **by** ...?" Revenue by store, average by weekday, share by category, performance by rep. That little word "by" maps in Pandas to exactly one thing: `groupby`.

In Excel this job is done with pivot tables or SUMIF. But once the conditions multiply ("by store — no, only weekends — and within that, by category"), the report needs weekly refreshes, and the aggregation criteria keep changing, manual work hits its limit fast. groupby turns every one of these questions into a one-line piece of code, and next week you simply run the same code again.

Finish this level and you will be able to go from raw data to "the summary table you show the executives" on your own.

## 2. Understanding through an analogy

Imagine tidying a box of 500 shuffled business cards. If you want to see them by company, you do this:

1. **Split** — sort the cards into piles by company name.
2. **Apply** — do the same task to each pile: count the cards, or average the job levels.
3. **Combine** — collect the per-pile results into one summary sheet.

These three steps are groupby's official name: the **split-apply-combine** pattern. The code `df.groupby("store")["revenue"].sum()` is a literal transcription of "split into piles by store → sum each pile's revenue → combine into one table".

One more analogy: `agg` is assigning each pile several tasks **at once** ("count the cards AND compute the average"), while `transform` **pastes the pile-level result back onto every single card**. We revisit this difference in section 3-4.

## 3. Core concepts

### 3-1. The basic form — group key, target column, aggregation function

```python
df.groupby("store")["revenue"].sum()
#          ─┬────   ─┬───────  ─┬─
#       group key  target col  aggregation
```

The result is a Series whose index is the group key. In place of `sum` you can use `mean`, `count`, `median`, `max`/`min`, and more.

### 3-2. agg — several aggregations at once

A report usually needs total, average, and count together. Use named aggregation with `agg` and you fix the result's column names in the same stroke.

```python
df.groupby("store").agg(
    total=("revenue", "sum"),
    avg=("revenue", "mean"),
    n=("revenue", "count"),
)
```

### 3-3. Multiple keys — "by store, and within each store by category"

Give the group key as a list and you get hierarchical aggregation: `df.groupby(["store", "category"])["revenue"].sum()`. The result carries a two-level index (MultiIndex), and `reset_index()` turns it back into a plain table. This answers questions like "which store × category combinations make the money".

### 3-4. agg vs transform — a summary, or the original size preserved?

| | Returned size | Used for |
|---|---|---|
| `agg` | one row per group (summary table) | reports, rankings |
| `transform` | exactly the original row count | derived columns (share within group, group-wise filling) |

For example, the derived column "what % of its store's total revenue is this row" is awkward to build without `transform`.

```python
store_total = df.groupby("store")["revenue"].transform("sum")
df["share_in_store"] = df["revenue"] / store_total
```

Level06's group-median filling ran on the same principle.

### 3-5. Sorting and ordering — the finishing touch of a report

- Aggregation results come back sorted alphabetically by group key. For a report you usually line them up largest first with `sort_values(ascending=False)`.
- For keys with a **meaningful order**, like weekdays, alphabetical order actively hurts. Force the order you want with `reindex(["Mon", "Tue", ...])`.

### 3-6. Never forget: clean before you aggregate

groupby's aggregation functions skip NaN silently. Convenient — but if negative contamination is still in there, your totals are distorted just the same. **Level06's cleaning → this level's aggregation** is the standard order of the real-world pipeline.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level07_groupby_aggregation
python3 main.py
```

main.py cleans a year of sales and then builds the "cafe chain revenue analysis report" step by step.

- **[1] Per-store summary (agg)** — total, mean, and count in one table. You can see that the later-numbered stores like Lakeside and University do more revenue (the store-size effect planted in the data).
- **[2] Weekday averages and the weekend effect** — fixes the Mon–Sun order with `reindex`, then computes the "weekend multiple" as weekend mean ÷ weekday mean. It comes out around 1.2x. The generator plants a 1.25x effect, but the ad-spend contribution is added regardless of weekday, which dilutes the multiple slightly — when interpreting aggregates, you have to think through structure like this.
- **[3] Category shares** — divides the totals by the grand total to get shares (%) and sorts descending. Check whether Coffee is #1.
- **[4] Store × category multi-key** — pulls the top 5 of the 25 combinations by revenue.
- **[5] transform derived column** — builds each row's "share within its store" and verifies the per-store shares sum to 1.
- **[6] Saving the report** — saves the per-store summary to `outputs/store_report.csv`. That CSV is exactly "the report you attach to the email".

The single most instructive part of the code is the weekday ordering in [2]. The alphabetical-order trap is dispatched with one line of `reindex`.

```python
weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_weekday = df.groupby("weekday")["revenue"].mean().reindex(weekday_order)
```

## 5. Try it yourself

1. **Add min and max** — add `best=("revenue", "max")` and `worst=("revenue", "min")` to the agg in [1] to pull each store's best and worst revenue too. (Hint: named aggregation takes as many entries as you like.)
2. **A weekday × category cross question** — answer "which weekday has the highest Beverage revenue?" in code. (Hint: first filter with `df[df["category"] == "Beverage"]`, then groupby weekday, finish with `idxmax()`.)
3. **Ad-spend efficiency** — compute `sum of revenue ÷ sum of ad_cost` per store and find the store with the highest "revenue per 1 KRW of ad spend". (Hint: get both sums with agg, then divide.)

## 6. Common mistakes

- **Writing groupby with no aggregation attached** — `df.groupby("store")` is only a "grouping blueprint", not yet a computation. Attach an aggregation like `.sum()` to get a result.
- **Forgetting the result is a Series** — the column name disappears and it throws people off. If you want a table, append `reset_index()`.
- **Leaving weekdays in alphabetical order** — a table ordered "Fri, Mon, Sat, Sun, Thu, Tue, Wed" is a reporting accident. Fix the order with `reindex`.
- **Using agg where transform belongs** — if you are building a derived column and get a size-mismatch error, nine times out of ten this is the confusion.
- **Aggregating without cleaning** — leftover negative revenue contaminates the per-store totals directly. Always run the level06 procedure before aggregating.

## Next level preview

So far we handled a single table. In practice, the answer only appears when you **combine** the sales table with the store info table and the targets table. In level08 we learn to join tables with merge and concat — and the traps that joins create.
