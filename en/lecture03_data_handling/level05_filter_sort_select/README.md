# Lecture 03 · Level 05 — Filtering, Sorting, and Selecting

> We answer business questions like "the top 10 weekend Coffee sales at Downtown", one line of code at a time.

**Difficulty** ⭐⭐ / **Prerequisites** level03_pandas_dataframe / **Estimated time** 45 min

## 1. Why learn this — the business view

Most questions from the boss look like this: "Pull last month's weekend sales for the Downtown store." "Which category gets the best bang for the ad budget?" The shared structure of these questions is **pick the rows (filter), line them up (sort), and take only what you need (select)**. In truth, 80% of data-analysis time is combinations of these three moves.

You can do the same things in Excel with the filter and sort buttons. But stack just three conditions and you are at dozens of clicks — and when the same question comes next month, you click through it all again from scratch. Code writes the conditions down in words, so even the most complex condition is one line, and next month is a single re-run. Finish this level and you will have the muscle to translate "a business question in plain words" into "a condition expression".

## 2. Understanding through an analogy

Filtering is the **security guard at the entrance**. Every row of the DataFrame lines up to pass through the gate, and the guard inspects each row against the condition, sticking a pass/block tag (True/False) on it. The list of these tags is the boolean mask. `df[mask]` means "only rows with a True tag may enter".

With several conditions, you post several guards. "Only people with a badge and (&) on the reservation list", "executives or (|) VIPs". Wrapping each guard's verdict in parentheses and combining them is the Pandas syntax.

loc and iloc are **two ways of finding an apartment**. loc finds it by **name tag (label)** — "Building 101, Unit 502" — while iloc finds it by **order (position)** — "the third building from the main gate, fifth floor up". The same home can be addressed both ways, and if the buildings get renamed in a redevelopment (if the index changes), loc's answer and iloc's answer can diverge.

## 3. Core concepts

### 3.1 Boolean masks — how filtering works

```python
mask = df["store"] == "Downtown"   # a Series filled with True/False
gangnam = df[mask]                 # keep only the True rows
```

Apply a comparison (`==`, `>`, `<=`, ...) to a column and you get one True/False per row; put that inside square brackets and you have a filter. By convention the two lines are fused: `df[df["store"] == "Downtown"]`.

### 3.2 Combining conditions — &, |, ~ and parentheses

```python
df[(df["store"] == "Downtown") & (df["weekday"].isin(["Sat", "Sun"]))]
```

- Use `&` (and), `|` (or), `~` (not). Python's own `and/or/not` **cannot be used.**
- Every condition must be wrapped in parentheses. Leave them out and operator precedence produces an error.
- To test membership in several values, `isin(["Sat", "Sun"])` is the clean way — shorter and less error-prone than chaining `==` with `|`.

### 3.3 loc vs iloc

| | loc | iloc |
|---|---|---|
| Keyed by | labels | integer positions |
| Example | `df.loc[3, "revenue"]` | `df.iloc[3, 6]` |
| Slices | end **inclusive** | end **exclusive** (the Python convention) |

A filtered DataFrame does not have index 0,1,2... — it carries the original row numbers along. So `filtered.loc[0]` can raise an error, while `filtered.iloc[0]` always returns "the first row". Not knowing this difference means a long, confused stare at a KeyError.

The pattern used most at work is **row condition and column selection at once**: `df.loc[mask, ["store", "revenue"]]`.

### 3.4 Sorting and top N

```python
df.sort_values("revenue", ascending=False).head(10)   # top 10 by revenue, descending
df.nlargest(10, "revenue")                            # same result, faster and shorter
df.sort_values(["store", "revenue"], ascending=[True, False])  # multiple keys
```

When "top N" is the goal, `nlargest`/`nsmallest` states the intent more clearly.

### 3.5 query — conditions as a sentence

```python
df.query("store == 'Downtown' and revenue >= 500000")
```

You can also write conditions as a string, without parentheses and `&`. It reads nicely but has limitations — column names with spaces become awkward, for example — so build your fundamentals on boolean masks and keep query as an option.

### 3.6 SettingWithCopyWarning — graffiti on a photocopy

Assigning values into a filtered result sometimes triggers a `SettingWithCopyWarning`. It appears because it is ambiguous whether the filter result is a copy of the original or a view into it. Two habits solve it: if your goal is modification, write it as the single statement `df.loc[mask, "col"] = value`; if you will keep using the subset, make an explicit copy with `sub = df[mask].copy()`.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level05_filter_sort_select
python3 main.py
```

main.py strips the missing rows out of 180 days of cafe-chain sales, then translates five business questions into code one by one.

- **[Q1]** Downtown & weekend & Coffee, top 10 by revenue — three conditions combined + `nlargest`. Most real questions are variations of this pattern.
- **[Q2]** Revenue at least KRW 500k & ad spend at most KRW 200k — combining numeric comparisons. The "find the efficient days" question.
- **[Q3]** Dessert and Bakery at the Airport and University stores — `isin` twice. Compare how long it would get written with `==` and `|`.
- **[Q4]** loc vs iloc — demonstrates how `iloc[0]` (the first row) and `loc[<original index>]` behave differently on a filtered table.
- **[Q5]** Top rows by revenue-per-ad-spend (roas) — a synthesis exercise: build a derived column, then sort.

Each question is printed as a "business question" before the code runs, so read along matching question → code → answer.

## 5. Try it yourself

1. **(Basic)** Select only Riverside's weekday rows (Mon–Fri) and count them. Hint: negation like `~df["weekday"].isin(["Sat", "Sun"])` gives you weekdays.
2. **(Applied)** Select the rows where the store is Lakeside and revenue is above the Lakeside average. Hint: first filter to Lakeside, store its mean in a variable, then use it in the second condition.
3. **(Challenge)** Extract "each store's single highest-revenue row" using loc and `idxmax()`. Hint: `df.loc[df.groupby("store")["revenue"].idxmax()]` — groupby is two levels ahead, so just running it as a taste is fine.

## 6. Common mistakes

- **Using `and`/`or`** — `(a) & (b)` is correct. `and` produces the famous "truth value is ambiguous" error.
- **Dropping the parentheses** — in `df[df["a"] > 1 & df["b"] < 2]`, `&` binds before the comparisons, giving nonsense or an error. Parentheses around every condition is the iron rule.
- **Calling loc[0] after a filter** — a filtered table's index may not start at 0. First row: `iloc[0]`; a specific label: `loc[label]`. Keep them apart.
- **Chained double brackets when assigning** — `df[mask]["col"] = value` warns and may not change the original. Write it in one shot: `df.loc[mask, "col"] = value`.
- **Whitespace in string comparisons** — `"Downtown "` (trailing space) is a different value from `"Downtown"`. If a filter returns 0 rows, check the actual values with `unique()`.

## Next level preview

Trying to answer the questions, we found empty values and negative revenue hiding in the data. In the next level we diagnose that contamination (isna, IQR), compare strategies for dropping or filling it (dropna, fillna), and learn "how to clean real-world data".
