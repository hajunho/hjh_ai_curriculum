# Lecture 06 · Level 05 — Logistic Regression — Where Classification Begins

> We build a model that goes beyond "churns / doesn't churn" to answer "73% probability of churn," and translate its coefficients into business language.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03, level04 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

For real-world classification problems, a **probability** is far more useful than a yes/no. Why "73% churn probability" beats "this customer will churn": with probabilities you can **prioritize**. If your retention team can only call 50 customers a day, call the 50 with the highest risk first. Probabilities also enable **expected-value math**: "30% churn probability × 400,000 KRW customer lifetime value = 120,000 KRW expected loss > 10,000 KRW coupon cost → sending the coupon pays for itself."

Logistic regression is the standard among probability-producing classifiers, and to this day it is the first choice wherever the burden of explanation is heavy — such as credit scoring in finance. Every one of its coefficients can be interpreted.

## 2. Understanding by Analogy

Picture **an insurance underwriter with a risk scorecard**. The underwriter scores each item: few usage days, +risk points; many support calls, +risk points; auto-pay enabled, −risk points… then adds them into a total. Up to here it's exactly the "weighted sum" of linear regression.

The problem: the total might be −5 points or +12 points, but the answer we want is a probability between 0% and 100%. So we pass the total through a **conversion funnel** that turns scores into probabilities. That funnel is the sigmoid function. A very low total gets squeezed toward 0%, a total of exactly 0 maps to exactly 50%, and a very high total lands near 100%. In other words, **logistic regression = a linear scorecard + a probability funnel**. If you understood level03, the funnel is the only new thing to learn.

## 3. Core Concepts

### 3.1 The Sigmoid: Score → Probability

```
z = a₁x₁ + a₂x₂ + … + b        (the linear score, same shape as level03)
p = 1 / (1 + e^(−z))            (the sigmoid: squeezes z into 0–1)
```

The sigmoid is an S-shaped curve. At z=0, p=0.5; by z=±4, p is already pinned at 0.98/0.02. Training means finding the coefficients a that "maximize the probability of the training data's true answers" (maximum likelihood). Unlike linear regression there's no closed-form formula, so it's solved by iterative optimization — sklearn handles that for you.

### 3.2 Interpreting Coefficients: Odds Ratios

If a coefficient aᵢ is positive, a larger value of that feature raises the churn probability; negative lowers it. To interpret the *size*, you need odds: odds = p/(1−p), the "churn-versus-stay betting line." **When feature xᵢ rises by one unit, the odds multiply by e^(aᵢ).** Example: if the support-calls coefficient is 0.5, then e^0.5 ≈ 1.65 — "each additional call multiplies the churn odds by 1.65." Reports conventionally quote this multiplier (the odds ratio).

Caution: if features come in different units (months vs KRW), coefficient sizes can't be compared directly. After standardizing (mean 0, standard deviation 1) as in today's exercise, each coefficient measures the effect of "a one-standard-deviation move in that feature," making them mutually comparable.

### 3.3 The Threshold: Probability → Decision

The model gives probabilities; the human makes decisions via a threshold. The default 0.5 is nothing but an arbitrary convention. Lower the threshold (say 0.3) and you classify more customers as at-risk — **fewer churners slip through, but more wasted outreach calls**; raise it and the reverse. The threshold is not a modeling issue but a **cost-vs-benefit business decision** — it should be set by whoever knows the cost of one call and the cost of losing one customer.

### 3.4 Why It's Called "Regression" When It Classifies

The name comes from "doing linear regression on the log-odds." A perennial exam gotcha: logistic regression is a **classification model**.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

We build a churn-probability model from churn_table (2,000 subscription customers).

- **[1]** Load the data, split into 6 features / 1 answer, and separate train/test the way level04 taught. `customer_id` is deliberately excluded (a meaningless column).
- **[2]** Check the sigmoid funnel's shape in numbers: what probability comes out at z = −4, −2, 0, 2, 4.
- **[3]** Train standardization + logistic regression and report test scores (accuracy, recall).
- **[4]** Print the coefficient table and translate to odds ratios. Sentences like "1 standard deviation more support calls → churn odds ×N" are generated automatically. Check whether the directions match the true signals planted in the data generator (usage days ↓, calls ↑, auto-pay −).
- **[5]** Pull predicted probabilities for 3 customers, and show how recall and precision trade places when the threshold moves from 0.5 → 0.3.

The heart of the code: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])`. With a pipeline, the scaler is fit on the training set only, so level04's leakage rule is honored automatically. Coefficients come out of `model.named_steps["clf"].coef_`.

## 5. Try It Yourself

1. **(Easy)** Raise the threshold in [5] to 0.7 and rerun. Which way do precision and recall move? In what business situation would 0.7 be the right choice? (Hint: when outreach is very expensive.)
2. **(Medium)** Drop `usage_days_30d` from the features and retrain. How much does performance fall, and how do the remaining coefficients change? (Hint: other features partially absorb the missing signal.)
3. **(Challenge)** Bolt on an expected-value calculator: assuming customer lifetime value 400,000 KRW, outreach cost 10,000 KRW, and a 30% save rate per call, find "the minimum churn probability at which calling pays off" and use it as the threshold. (Hint: p × 400,000 × 0.3 > 10,000.)

## 6. Common Mistakes

- **Throwing away the probability and reporting only 0/1**: that discards logistic regression's greatest strength. Prioritization and expected value come from the probability.
- **Comparing coefficient sizes without standardizing**: "the monthly-fee coefficient is smallest, so it doesn't matter" — it may only look small because the unit is KRW (in the tens of thousands).
- **Reading coefficients as causation**: "make customers enable auto-pay and churn drops" is a causal claim. Customers inclined to enable auto-pay may simply have been more loyal to begin with (level03's correlation ≠ causation, revisited).
- **Treating threshold 0.5 as sacred**: 0.5 is just the default; adjusting it to your cost structure is normal.
- **Reporting performance with accuracy alone**: on data with 15% churn, accuracy is an optical illusion. That's exactly the next level's topic.

## Next Level Preview

At the end of this level, two unfamiliar metrics — precision and recall — slipped into view. In level06, using extremely imbalanced data with a 1.5% fraud rate, we build a "98.5% accurate model that catches nothing" with our own hands, and fully sort out the confusion matrix, precision, recall, F1, and ROC-AUC.
