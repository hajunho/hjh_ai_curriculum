# Lecture 07 · Level 02 — The Exploratory Data Analysis (EDA) Process

> Instead of "let's just draw some charts," internalize an EDA procedure that runs on the hypothesis→verify cycle and the univariate → bivariate → time order.
**Difficulty** ⭐⭐ / **Prerequisites** level00, lecture05 / **Estimated time** 40 min

## 1. Why this matters — the business view

A large share of modeling failures start with EDA (Exploratory Data Analysis) that was skipped or done carelessly. Evaluating with accuracy without knowing the churn rate is 18%, averaging over a range where missing values are concentrated, feeding an obviously leaky variable in as a feature — all of those accidents begin here.

In practice EDA plays two roles. First, **risk removal**: it finds the data's traps (missing values, outliers, imbalance, leakage) before modeling. Second, **hypothesis production**: a discovery like "customers with many support calls tend to churn" is worth reporting to the business on its own, and becomes raw material for the feature engineering in the next level. Many projects end with an EDA report and no model at all — that is not a failure, it's a solid win.

## 2. An analogy to hold onto

EDA is a **health checkup**. A hospital examines the patient in a fixed order before putting them on the operating table (the model).

1. **Basic intake (data overview)**: height, weight, age — row counts, column types, missing-value ratios.
2. **Item-by-item tests (univariate)**: blood pressure, blood sugar, one at a time — look at each variable's distribution individually.
3. **Linked tests (bivariate)**: "the relationship between blood pressure and weight" — look at variable-vs-target and variable-vs-variable relationships.
4. **Follow-up over time (temporal)**: compare with last year's numbers — look at change and trends over time.

And a doctor forms a **working hypothesis** before ordering tests. Just as "abdominal pain → suspect the stomach first," you write down "low usage days probably means more churn" first, then check it against the data. Drawing dozens of charts with no hypothesis is like ordering a full-body MRI before hearing the symptoms: expensive, and easy to be fooled by signals that appeared by chance.

## 3. Core concepts

### 3.1 The hypothesis → verify cycle

The unit of EDA work is: "① write the hypothesis as a sentence → ② build the minimal table/chart that can check it → ③ record the verdict (supported / rejected / undecided) → ④ repeat when new hypotheses appear." The point: **the sentence comes before the chart**. The sentence determines which chart to draw, and it transfers straight into the report.

### 3.2 The univariate → bivariate → time order

- **Univariate**: one variable's distribution. Mean, median, standard deviation, min, max, histogram. This is where you catch missing values, outliers, and skew. The target variable's ratio (churn rate 18%) must also be checked at this stage.
- **Bivariate**: variable × target. For classification, "compare group means by target" is the single most powerful move. If churned and retained customers differ sharply in mean `usage_days_30d`, that variable is a promising feature. Between numeric variables, look at the correlation coefficient.
- **Temporal**: if the data has a time axis, look at trends, seasonality, and sudden shifts. churn_table has no time axis, so we skip it in this exercise, but it takes the lead role in level08 (sales forecasting).

### 3.3 The four traps EDA must catch

| Trap | Symptom | Follow-up |
|---|---|---|
| Missing values | blanks concentrated in certain columns/ranges | decide a removal/imputation strategy (lecture03) |
| Outliers | negative revenue, absurd maximums | verify cause, then remove/fix |
| Imbalance | target ratios like 18% or 1.5% | redesign metrics and weights (level07) |
| Suspected leakage | a suspicious variable with 0.95 correlation to the target | check creation timing, then exclude (level03) |

### 3.4 How to read correlation coefficients

Read +0.3 as roughly "a weak positive relationship" and |0.7|+ as "a strong relationship" — but remember two things. First, correlation is not causation (ice cream sales and drowning accidents). Second, a suspiciously high correlation is something to **distrust**, not celebrate. A variable with 0.9 correlation to the target is usually one that "could only be built if you already knew the answer" (leakage).

### 3.5 The shape of an EDA report

The deliverable of EDA is not a pile of charts but a **list of sentences with verdicts**. A good format: a 4-column table of "hypothesis → evidence (table/figure number) → verdict → implication for modeling." Example: "less usage → more churn → Fig. 2 → supported → key feature candidate." Organized this way, your EDA becomes the feature list (level03) and the first draft of the executive report at the same time.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Generates an automated EDA report for the 2,000 customers in `churn_table`.

- [1] Prints 3 hypotheses as sentences first (before any verification).
- [2] Basic intake: rows/columns, types, missing values, target ratio.
- [3] Univariate: a summary-statistics table per numeric variable.
- [4] Bivariate: a churned-vs-retained group-mean comparison table with a "difference ratio," plus a ranking of correlations with the target.
- [5] Hypothesis verdicts: prints supported/rejected for each hypothesis from [1].
- [6] Saves 2 figures to `outputs/`: ① distribution comparison of key variables by churn group, ② a correlation-matrix heatmap.

The heart of the code is the single line `df.groupby("churned")[cols].mean()`. See with your own eyes that 80% of bivariate analysis is this one groupby comparison. The heatmap is nothing more than `df.corr()` drawn with `imshow`.

## 5. Try it yourself

1. **(Easy)** Add your own fourth hypothesis to [1] (e.g., "auto-pay customers churn less") and print its verdict in [5]. Hint: read the `auto_pay` row from the group-mean comparison table (`grp`) in [4].
2. **(Medium)** Rerun with `n=200`. The group-mean differences get jumpy. Using the statistics from lecture05, explain why you shouldn't state firm EDA conclusions from a small sample.
3. **(Challenge)** Build the same report for `fraud_table`. You only need to change the target column name (`is_fraud`) and the list of numeric columns. Observe how the `amount` and `hour` distributions of fraudulent transactions differ from normal ones.

## 6. Common mistakes

- **Producing dozens of charts with no purpose**: charts without a hypothesis sentence all get cut from the report. Sentence first; the chart is the evidence.
- **Starting to model without checking the target ratio**: if you don't know about the imbalance, every subsequent evaluation is distorted. This is EDA check item number one.
- **Looking only at means, never at distributions**: two datasets with the same mean but different shapes are completely different data. The summary table and the histogram are a set.
- **Tuning rules against what you saw in the test data during EDA**: the test set stays sealed even during EDA, as a rule. If you did EDA on the full dataset, state that explicitly in the report.

## Next level preview

Insights from EDA like "churners use the service less and call support more" are the raw material for features. In level03 we combine raw columns into derived features and prove by experiment how much they lift model performance.
