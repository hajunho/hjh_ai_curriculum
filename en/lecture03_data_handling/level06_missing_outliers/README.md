# Lecture 03 · Level 06 — Handling Missing Values and Outliers

> Learn to find the empty values and the values that make no sense, and to choose a handling strategy that fits the situation.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level05_filter_sort_select / **Estimated time** 40 min

## 1. Why learn this — the business view

Real-world data is almost never clean. A POS terminal went down for a while and revenue never got recorded, a refund came through as negative revenue, someone fat-fingered an extra zero — these things happen every day.

The problem is that this contamination wrecks your analysis **silently**. Compute a mean with missing values mixed in and the sample shrinks without telling you; mix in negative revenue and the total comes out smaller than reality. Report numbers that are wrong while no error is ever raised — that is the scariest scenario there is.

That is why data analysts say "more than half of analysis time goes into cleaning". In this level you will **diagnose** the contamination, compare several **handling strategies**, and develop the judgment for which strategy fits which situation.

## 2. Understanding through an analogy

Imagine doing the month-end close, holding the stack of sales slips sent up from five stores.

- On some slips the **amount box is blank**. → Missing values.
- One slip reads **−350,000 KRW**. Revenue cannot be negative, so something went wrong. → A domain-rule violation.
- One slip, from a store that normally does around 300 thousand, stands alone at **90 million**. Not impossible, but deeply suspicious. → A statistical outlier.

An experienced bookkeeper does not simply toss these slips. First they **count how many look wrong** (diagnosis), then they call the store and **ask why** (investigation), and only when the cause cannot be determined do they **correct by rule** (handling). For example, a slip with a blank amount gets provisionally booked at "that store's usual figure for that item".

Handling missing values and outliers in Pandas follows exactly the same order. Count with `isna()`, infer the cause, then handle with `dropna()` and `fillna()`.

## 3. Core concepts

### 3-1. Where do missing values come from?

Missing values have three broad causes.

| Type | Example | Handling hint |
|---|---|---|
| Collection failure | terminal connectivity error dropped a record | if it is missing at random, filling (imputation) is relatively safe |
| Value never existed | "last year's revenue" for a brand-new store | often must NOT be filled |
| Entry mistake | someone skipped the field | checking the source is best |

Because the correct handling depends on the cause, the first principle is: **ask why before you delete.**

### 3-2. Diagnostic tools — isna, notna, describe

- `df["revenue"].isna()` : tells you True/False whether each value is missing. Append `.sum()` for the count of missing values, `.mean()` for the missing ratio.
- `df.describe()` : shows count, mean, min, and max at once. **A negative min** makes no sense for revenue data, so you spot it immediately.

### 3-3. Domain-rule outliers — business knowledge comes first

Before any statistical technique, apply domain rules — plain business common sense. "Revenue is at least 0", "age cannot exceed 150", and so on. A value that breaks the rule is a measurement error, and the standard move is to convert it to a missing value.

```python
df.loc[df["revenue"] < 0, "revenue"] = np.nan
```

Careful: simply **deleting** the negatives also erases the fact that "that store did sell that day". Marking only the value as unknown (NaN) loses less information.

### 3-4. Statistical outliers — the IQR method

The "suspiciously large or small values" that domain rules cannot catch are found with the IQR (Interquartile Range).

- Q1 = the 25th percentile, Q3 = the 75th percentile, IQR = Q3 − Q1
- Normal range: **[Q1 − 1.5×IQR, Q3 + 1.5×IQR]** — values outside it are outlier candidates.

The word "candidates" matters. **Genuine phenomena** — like a blockbuster weekend — get flagged as outliers too. IQR is not an auto-delete tool; it is a tool for **building a review list**.

### 3-5. Three handling strategies

| Strategy | Code | Pros | Cons |
|---|---|---|---|
| (a) drop the rows | `dropna()` | simple, no fabricated values | sample shrinks, totals come out low |
| (b) fill with the overall median | `fillna(med)` | rows preserved | ignores store/product characteristics |
| (c) fill with per-group medians | `groupby().transform("median")` | context-aware, most precise | slightly more code |

The reason to use the **median** rather than the mean: the mean gets dragged around by any outliers still present, while the median holds its ground. Strategy (c) is the idea "fill a blank in Downtown-Coffee with Downtown-Coffee's usual value", and it is the approach used most in practice.

The key fact: **every strategy changes the statistics.** If the report's headline is a total, deletion is dangerous; if it is an average, the choice of fill matters less. That is why you need the habit of switching strategies and comparing the results.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level06_missing_outliers
python3 main.py
```

The output flow and what to watch for:

- **[1] Diagnosis** — missing count and ratio, negative count, and the negative min in `describe()`. About 1% missing and about 0.5% negatives are planted among the 9,125 rows.
- **[2] Applying the domain rule** — converts negative revenue to NaN. Note that this marks "value unknown" rather than deleting.
- **[3] IQR boundaries** — computes Q1, Q3, the normal range, and the count of upper outliers. It also prints the caveat that weekend/peak-season sales may be among them.
- **[4] Strategy comparison** — handles the data with all three strategies (dropna / overall median / group median) and compares row counts, total revenue, and means in one table. The highlight: dropna's total revenue comes out visibly smaller.
- **[5] Before/after summary and boxplot** — prints the before/after summary and saves a boxplot PNG into `outputs/`.

The heart of the code is strategy (c).

```python
group_median = df.groupby(["store", "category"])["revenue"].transform("median")
df["revenue_filled"] = df["revenue"].fillna(group_median)
```

`transform` returns the per-group results **at the original row count**, so each row gets filled with its own group's median.

## 5. Try it yourself

1. **Check the missing ratio per store** — see whether missing values cluster at a particular store with `df[df["revenue"].isna()]["store"].value_counts()`. (Hint: this data is contaminated at random, so it comes out even. At work, if one store dominates, suspect that store's equipment.)
2. **Mean fill vs median fill** — switch strategy (b) to `mean()` and compare how the total and the mean change. (Hint: filling with the mean computed **before** converting negatives to NaN propagates the contamination. Order matters.)
3. **IQR multiplier experiment** — what happens to the outlier count with 3.0 instead of 1.5? For what kind of report would 3.0 (conservative) be more appropriate? (Hint: an automated alerting system that wants fewer false positives.)

## 6. Common mistakes

- **Trying to filter missing values with `df[df["revenue"] != None]`** — Pandas missing values are `NaN`, and a `!= None` comparison does not catch them. Always use `isna()`/`notna()`.
- **Not knowing that NaN == NaN is False** — NaN is not even equal to itself. That is exactly why you need the dedicated functions instead of comparisons.
- **Not assigning the result of fillna** — `df.fillna(0)` returns a new DataFrame and does not change the original. Reassign, as in `df = df.fillna(0)`.
- **Damaging the original** — clean on a copy (`df.copy()`) and leave the source file alone. "The original is evidence."
- **Auto-deleting outliers** — mechanically deleting whatever IQR flags throws away real signals like the holiday rush. Always review by eye before deciding.

## Next level preview

The data is clean, so now the real analysis begins. In level07 we build aggregation reports with `groupby`, answering questions like "revenue by store, by weekday, by category?".
