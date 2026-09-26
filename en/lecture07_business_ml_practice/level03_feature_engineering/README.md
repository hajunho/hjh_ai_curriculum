# Lecture 07 · Level 03 — Feature Engineering

> Same data, same model — performance still changes with how you shape the features. We pit raw features against derived features on the same model and prove the effect.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level02, lecture06 level05 / **Estimated time** 40 min

## 1. Why this matters — the business view

On Kaggle and in companies alike, what separates good predictions from bad is usually not the choice of model but the **quality of the features (input variables)**. And the raw material for good features is not mathematics but **domain knowledge** — the instincts of someone who knows the business.

"A card swiped repeatedly overseas in the small hours is suspicious" is common sense to any card-fraud analyst. But the data only holds the isolated numbers `hour=2`, `is_foreign=1`, `tx_count_1h=9` — there is no column called "suspicious pattern." Translating that common sense into numeric columns a model can digest is feature engineering, and it is the single biggest place a non-engineer can contribute to a data project. The data team can do the coding, but only the business side knows *which features to build*.

## 2. An analogy to hold onto

A model is a **picky underwriter who eats only what you serve**. Imagine handing documents to a credit underwriter.

- Raw as-is: "monthly income 4,000,000 KRW", "monthly loan payment 1,800,000 KRW" — looking at the two numbers separately, judgment is hard.
- Derived feature: "debt-service ratio 45%" — one division later, the judgment is instant.

A human underwriter does that division automatically in their head, but a linear model cannot invent divisions on its own (it can only add and multiply by constants). So a human has to prepare ratio, difference, and bucket columns in advance — serving the data in "digestible form." In cooking terms, feature engineering is prepping the ingredients. Hand even a great chef (the model) a whole unscaled fish and they can't show their skill.

## 3. Core concepts

### 3.1 The classic recipes for derived features

| Recipe | Example | When to use |
|---|---|---|
| Ratio | payment-to-income, usage days / months of tenure | when the relative relationship of two numbers matters |
| Difference | this month − last month's usage | when change/trend is the signal |
| Log transform | log(transaction amount) | heavily skewed distributions, like money |
| Binning / flags | small-hours flag (is_night) | when a "range" carries meaning |
| Interaction | foreign × night | patterns risky only when conditions coincide |
| Aggregate | per-customer average payment over the last 30 days | summarizing transaction-level data to the customer level |

Of these, the log transform is the regular for money data. It makes the gap between 10,000 and 20,000 KRW (2×) the same size as the gap between 100M and 200M KRW (2×), letting a linear model think in the language of "multiples."

### 3.2 Where the ideas come from — EDA and business interviews

The bivariate analysis from level02 is literally your ingredient list. Saw "fraud clusters in the small hours"? Build an `is_night` flag. Saw "the amount distribution is heavily skewed"? Build a log transform. Add business interviews ("what makes something feel suspicious to you?") and your feature-candidate list is complete.

### 3.3 Data leakage — feature engineering's biggest trap

While building derived features, it's easy to accidentally mix **information from the future** or **a shadow of the answer** into a feature.

- Predicting churn from "was refunded": refunds happen **after** churn — the value is unknowable at prediction time.
- Computing a derived feature from the mean of the full dataset (test included): test information leaks into training.
- Predicting fraud from an "investigation result code": investigations happen **after** something is ruled fraud.

There is exactly one screening question: **"At the moment I have to make the prediction, can I know this value?"** If the answer is no, throw the feature away no matter how good the numbers look. A leaky feature is the nastiest kind of bug: fantastic validation scores, useless in production.

### 3.4 Compare performance fairly

To claim an effect for derived features you must **freeze the model, the data split, and the metrics completely**, changing only the feature list. main.py runs exactly this controlled experiment.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Trains a fraud-detection model twice on the 5,000 transactions of `fraud_table`. "Raw" means the 3 fields that arrive with the payment authorization itself (amount, hour, is_foreign).

- [1] Loads the data and inspects the 3 raw features.
- [2] Builds 3 derived features from domain knowledge: `log_amount` (log transform), `is_night` (small-hours flag), `foreign_night` (foreign × night interaction).
- [3] Trains "raw only" vs "raw + derived" with the same logistic regression and the same split, comparing AUC and PR-AUC (average precision).
- [4] Prints the derived-set model's coefficients to see which features do the work.
- [5] Leakage demo: adds one fake feature secretly mixed with the answer, shows performance rocketing to unrealistic heights, and prints the warning.

The output to watch is the PR-AUC comparison in [3]. Not one character of the model changed, yet performance jumps — that is the power of features. Take away the instinct from [5] too: "performance that's too good is a bug."

## 5. Try it yourself

1. **(Easy)** Remove the derived features one at a time and rerun to see which contributes most. Hint: delete entries from the `DERIVED` list one by one.
2. **(Medium)** Bring an aggregate feature into the fight. Add `tx_count_1h` from the table (payments in the last hour — a value you only get by aggregating transaction history) and the ratio feature `amount_per_tx = amount / (tx_count_1h + 1)` to DERIVED. How high does PR-AUC go? Feel for yourself why "aggregate features are the perennial ace."
3. **(Challenge)** Design the same experiment on churn_table. Candidates: `usage_days_30d / tenure_months` (activity relative to tenure), `support_calls_30d - plan_changes`, etc. Check which features get coefficients whose direction matches domain common sense.

## 6. Common mistakes

- **Skipping the prediction-time check**: if you don't ask "can I know this at prediction time?" for every feature, leakage *will* happen. Keep a "created when" column next to your feature list.
- **Growing features without limit**: more features are not better. Too many features relative to samples brings overfitting (lecture06 level09). Build → validate → discard what doesn't contribute.
- **Dividing by zero**: the classic ratio-feature accident. Add +1 to the denominator (main.py's approach) or handle it conditionally.
- **Computing statistical features over data that includes the test set**: features like "ratio to the overall mean" must be built from training-set statistics only. The pipelines in level05 solve this structurally.

## Next level preview

Every feature so far has been a number. But half of real-world data is text: "store name," "category," "day of week." level04 covers the three encodings that turn categorical variables into numbers — and each one's trap.
