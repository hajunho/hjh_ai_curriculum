# Lecture 07 · Level 08 — Case Study 1 — Sales Demand Forecasting

> Build a full pipeline that forecasts "next week's daily revenue" from one year of cafe-chain sales. Day-of-week, seasonality, and moving-average features plus time-based splitting play the lead roles.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level03, level06 / **Estimated time** 50 min

## 1. Why this matters — the business view

"How much will we sell next week?" is the universal question across every industry. The answer drives order quantities, staff schedules, and cash-flow plans. Miss the forecast by 10% and it turns straight into cost: spoiled inventory or stockouts, overstaffing or queues.

Demand forecasting is **regression**, not classification, and the data has a **time axis** — which makes it a different animal from everything so far. A time axis changes two things. First, the main features become time-derived ones: "day of week, season, recent trend." Second, the validation rules change — you must train on the past and test on the future (the trap we saw in level06 [4]). This level is the **first case study you run all the way to the finish line** with both pieces in place.

## 2. An analogy to hold onto

Let's transcribe the mind of an experienced cafe manager. Deciding next week's order quantities, the manager thinks roughly like this:

1. "How much are we selling per day lately?" → **moving average** (recent trend)
2. "Next Saturday will sell about like a normal Saturday" → **day-of-week effect**
3. "Summer's coming, beverages will pick up" → **seasonal effect**
4. "We sold a lot yesterday out of nowhere — will some of that carry into tomorrow?" → **previous-day revenue (lag)**

Machine-learning demand forecasting turns each of these four intuitions into a numeric column and hands them to the model. The model doesn't see the future by magic — it **translates the manager's intuition into features and applies it consistently, at scale, automatically, every day**. That's the essence. Which is why most of this level's code goes into feature building, not the model.

## 3. Core concepts

### 3.1 The big three time-series features

- **Calendar features**: day of week (one-hot or a weekend flag), month, holiday flags. They capture repeating patterns.
- **Lag features**: `lag_1` (yesterday), `lag_7` (same weekday last week). They capture "how things have been lately."
- **Rolling means**: last-7-day average, 28-day average. They capture the trend with the noise filtered out.

Caution: lags and rolling means must use **only values from before the prediction moment** — compute them after `shift(1)`. Use a rolling mean that includes today in today's forecast and you're copying the answer: leakage.

### 3.2 Seasonality as numbers — the sin/cos transform

Feed in "day of year (1–365)" raw and December 31 (365) and January 1 (1) look like polar opposites. Convert to the two columns `sin(2π·day/365)`, `cos(2π·day/365)` and you get a circular calendar where year-end and new-year are natural neighbors. The same trick works for weekdays, but with only 7 of them one-hot is more common.

### 3.3 Time-based splitting and baselines

- **Split**: last few weeks as test (reproducing the upcoming weeks), everything before as training. Random splitting is forbidden. If the evaluation window is too short (say, 7 days) the ranking is decided by luck, so in practice you evaluate with a backtest spanning several weeks.
- **Baseline**: the minimum bar the model must clear. Demand forecasting's conventional baselines are ① "same as yesterday" (naive) and ② "same as the same weekday last week" (seasonal naive). A model that can't beat these simple rules has no reason to be deployed. In real reports too, always speak in improvement relative to the baseline.

### 3.4 Metrics — MAE and MAPE

- **MAE** (mean absolute error): "off by N KRW per day on average." In currency units, so it's intuitive.
- **MAPE** (mean absolute percentage error): "off by N% on average." Useful for comparing locations of different sizes, but it explodes when actual values approach 0 — handle with care.

## 4. Hands-on — main.py

```bash
python3 main.py
```

- [1] Cleans one year of `sales_table` (drops missing/negative) and aggregates to daily total revenue.
- [2] Feature building: weekday one-hots, weekend flag, seasonal sin/cos, lag_7 (same weekday last week), 7-day/28-day moving averages (all as of 7 days ago — only values knowable at prediction time).
- [3] Time-based split: seals the last 4 weeks (28 days) as the evaluation window. Seven days alone mixes in too much luck, so we evaluate with a 4-week backtest assuming a weekly forecast-refresh operation.
- [4] Compares the two baselines (same as yesterday / same weekday last week) against a RandomForest regressor on MAE and MAPE over the 4-week backtest.
- [5] Prints "next week" (the last 7 days) forecast vs actual day by day, and saves a PNG of the recent 8-week trend with the forecast window to `outputs/`.
- [6] Uses feature importances to see "what the model looked at to predict."

Things to observe: the baselines (especially seasonal naive) are stronger than you'd expect, and where the model's improvement comes from (the combination of weekday + trend). In the PNG, check whether the forecast line follows the actuals' weekday rhythm.

## 5. Try it yourself

1. **(Easy)** Change the evaluation window (`BACKTEST_DAYS = 7`, `56`). In the 7-day evaluation the baseline sometimes wins — reconfirm level06's lesson that the shorter the window, the more luck decides the ranking.
2. **(Medium)** Add lag_14 (same weekday two weeks ago) and lag_28 to the features. Does MAE improve? How does the feature-importance ranking shift?
3. **(Challenge)** Run the same pipeline on the daily revenue of "the Beverage category only" instead of total revenue. The seasonal sin/cos importance should shoot up — check that it matches the domain knowledge of the summer beverage peak.

## 6. Common mistakes

- **Validating with a random split**: time series' first taboo. The score looks great and then collapses in production. Split by time, always.
- **Including today in the moving average**: use `rolling(7).mean()` without `shift(1)` and today's value enters today's feature. A regular source of silent leakage.
- **Reporting "MAE 500,000 KRW" with no baseline**: nobody knows whether that number is good or bad. Report relative improvement, like "−23% vs same-as-yesterday."
- **Forecasting far ahead in one shot**: lag features are for one-day-ahead forecasting. To predict 7 days out you must explicitly design either a recursive scheme (feeding predictions back as inputs) or a direct scheme using only lags ≥ 7 days (main.py uses the direct scheme — only information up to 7 days back, via lag_7 and the moving averages).
- **Ignoring holidays and promotions**: in real data, the biggest errors happen on special days. A special-day calendar as a feature is half of real-world accuracy.

## Next level preview

Case study two is classification. In level09 we take `churn_table` through a complete customer-churn project — EDA → features → cross-validation → interpretation → action list — and produce the practical deliverable: "top-10 at-risk customers + a recommended action for each."
