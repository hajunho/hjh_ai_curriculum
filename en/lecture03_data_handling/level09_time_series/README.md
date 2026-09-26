# Lecture 03 · Level 09 — Working with Time Series Data

> Learn to read trend, seasonality, and growth by putting revenue "up on the time axis".

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level07 (grouping and aggregation) / **Estimated time** 60 min

## 1. Why learn this — the business view

Half of the questions executives ask are about time. "How much are we up on last week?", "Has this month's trend turned, or is this just the usual season?", "Strip out the weekend effect — are we actually growing?" Every one of these can only be answered once the data is lined up in date order.

Data where the time order carries meaning is called a time series. Revenue, visitor counts, inventory, stock prices, server traffic — most real-world data is, in fact, time series. But time series demand slightly special handling. You must not shuffle the row order casually, you need operations that change the time unit ("daily → weekly"), and the core computations compare the data **with its own past** ("versus last week"). Pandas provides dedicated tools for this: `to_datetime`, `resample`, `rolling`, and `shift`. Master just these four and you can produce every headline number of a weekly report in code.

## 2. Understanding through an analogy

**Time series analysis is like keeping a weight log.** Say you record your weight every morning.

- The daily numbers bounce around. A team dinner last night and you look 1 kg heavier today. So we look at the "average of the last 7 days" instead. That is exactly the moving average (rolling mean) — a magnifying glass that erases one-day noise and leaves only the true trend.
- "Daily entries are too much — let's summarize by week" is turning the diary into a weekly digest. That is resampling (resample).
- Asking "how many % did this week change versus last week's average?" is `shift` (slide by one slot) and `pct_change` (rate of change) — standing last week's you next to this week's you and comparing.

One more thing. What if the weight notebook says "February 31st"? A person laughs it off; a computer halts. Real-world data genuinely contains malformed date strings like this. The first button of any time series analysis is always checking "are these dates actually dates?".

## 3. Core concepts

### 3-1. A string is not a date — to_datetime

The `"2025-03-15"` you read from a CSV is just characters. Sort it as characters and it shuffles in dictionary order; subtraction doesn't work either. Only after converting to the date type (datetime64) with `pd.to_datetime()` do computations like "how many days apart" and "which weekday" become possible.

The `errors` option matters here.

| Option | Behavior | When to use |
|---|---|---|
| `errors="raise"` (default) | error immediately on a bad date | when you are sure there is no contamination |
| `errors="coerce"` | turn bad dates into NaT (Not a Time — the missing value of dates) | most real-world data |

After converting with `coerce`, count the NaT values with `isna().sum()` and the diagnosis "how many contaminated dates" is done. It is the level06 missing-value habit, applied to dates.

### 3-2. DatetimeIndex and the dt accessor

Promote the date column to the index (a DatetimeIndex) and time-series-only features like `resample` unlock. On a date column that is not the index, you pull parts out with the `dt` accessor: `df["date"].dt.month` gives the month, `.dt.dayofweek` the weekday number.

### 3-3. resample — changing the time unit

`daily.resample("W").sum()` pours the daily data into week-sized boxes and sums each box. `"W"` is weekly; `"ME"` is monthly, anchored at month end. Think of it as groupby's time-flavored twin — internally it really is "split into time buckets → apply the aggregation to each group", the same split-apply-combine as level07.

### 3-4. rolling — noise removal with a moving average

`daily.rolling(7).mean()` computes "the average of the last 7 days including today", every day. Because the window slides forward one day at a time, it is called a moving average. For data like our cafe's, where sales spike every weekend, a 7-day window contains exactly one full cycle of the weekday effect — the chop disappears and only the seasonal trend remains. The first 6 days cannot fill the window and come out NaN; that is normal.

### 3-5. shift and pct_change — comparing with your past self

`weekly.shift(1)` slides the series down one slot, placing "last week's value" beside this week's row. Growth is then `(this week − last week) / last week`. The one-liner for this computation is `weekly.pct_change()`. The first row has no past to compare against, so it is NaN.

### 3-6. Two time series traps

First, unsorted dates. Data stitched together from several files often has dates out of order, and running `shift` on it compares against "whatever row is above" rather than "last week". `sort_index()` comes first. Second, missing dates. If closure days are absent as rows, the moving-average window actually spans more than 7 calendar days. Build the habit of filling in the gaps with `resample("D")` to check.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level09_time_series
python3 main.py
```

The output flows as follows.

- **[1] Date contamination diagnosis** — converts `sales_table`'s `date` column with `to_datetime(errors="coerce")`. The dates in this data were stamped with the broken rule "12 months of 31 days each", so calendar-impossible dates like `2025-02-31` are mixed in. We count how many values coerce turned into NaT to size the contamination, then rebuild correct dates by adding `pd.to_timedelta` to the trustworthy `day_index` (days since opening). Deciding "which column of the source can I trust" is itself the real-world skill.
- **[2] Building the daily series** — sums all stores and categories per date into a daily series with a DatetimeIndex, and pulls the month out via the `dt` accessor.
- **[3] resample** — computes weekly and monthly totals. Also confirms that the first and last weeks are "partial weeks" that could not fill all 7 days, so their values run small.
- **[4] rolling(7)** — lays the 7-day moving average over daily revenue and watches the weekend chop vanish while the seasonal trend emerges.
- **[5] Week-over-week growth** — applies `pct_change` to weekly revenue and pulls the top 3 and bottom 3 weeks by growth. Reproduces the same computation by hand with `shift` to verify the two methods agree.
- **[6] Saving the chart** — saves daily revenue (thin line) and the 7-day moving average (thick line) to `outputs/daily_trend.png`. See for yourself why the moving average is "the reading glasses for trends".

## 5. Try it yourself

1. **(Basic)** Use `.mean()` instead of `resample("ME").sum()` to get each month's "average daily revenue". Which is fairer for month-to-month comparison, the total or the mean? Hint: think about short months like February.
2. **(Applied)** Filter to the Coffee category only, build its daily series, and plot a 14-day moving average. Hint: filter first with `df[df["category"] == "Coffee"]`, then repeat the steps of [2].
3. **(Challenge)** Let's imitate "year-over-year" on weekly revenue. Since we only have one year of data, compute "growth versus 4 weeks ago" with `shift(4)` and confirm it matches `pct_change(periods=4)`.

## 6. Common mistakes

- **Sorting and comparing while still strings** — `"2025-9-2"` sorts after `"2025-10-1"` as a string. Converting with `to_datetime` is unconditionally step one of any time series work.
- **Not checking the result of coerce** — `errors="coerce"` is convenient but silently erases data into NaT. If you don't count the NaT right after converting, you won't even know data disappeared.
- **Resampling without building the index** — `resample` needs a DatetimeIndex (or an `on=` argument). If you see "Only valid with DatetimeIndex", check the index first.
- **Mistaking pct_change's first NaN for a bug** — NaN where there is no past to compare against is correct. Force-filling it with 0 creates the false information "0% growth".
- **Reporting partial weeks/months as-is** — the boxes at both ends of the period may not have all their days. Trim them or add a footnote before they go into a report.

## Next level preview

The time axis is conquered; next comes the crown jewel of reporting, the **pivot table**. In level10 we build two-dimensional summaries like store × category (pivot_table), reshape tables at will (melt), and finish reports with window operations like ranks and running totals.
