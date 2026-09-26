# Lecture 07 · Level 01 — Connecting Business Metrics to ML Metrics

> Convert the four cells of the confusion matrix into KRW amounts, and answer "how much money is a 1-percentage-point recall improvement worth?" with an actual number.
**Difficulty** ⭐⭐ / **Prerequisites** level00, lecture06 level06 (evaluation metrics) / **Estimated time** 35 min

## 1. Why this matters — the business view

When the data team reports "we raised recall from 62% to 71%," the executive response is usually silence. Say instead "converted to money, the same improvement roughly doubles our annual churn-defense profit," and the mood of the meeting changes. Same fact, different language.

ML metrics (recall, precision, AUC) and business metrics (revenue, cost, churn rate) are mutually translatable — yet people who can actually do the translation are surprisingly rare in most organizations. The translator has exactly one core move: **attach a unit price to each cell of the confusion matrix**. Finish this level and you'll be able to compute "so how much is this model worth?" for any classifier. Business cases for model adoption, vendor acceptance reviews, target setting — they all rest on this calculation.

## 2. An analogy to hold onto

Think of an airport security checkpoint. Saying a screener's performance is "99% decision accuracy" tells you nothing. What matters is the **price tag** on each of the four possible events.

- Catching a dangerous item (TP): an incident prevented — big win
- Missing a dangerous item (FN): the worst cost — one incident can shake the whole company
- Opening a harmless bag (FP): passenger annoyance and screening labor — small but accumulating cost
- Waving a harmless bag through (TN): normal operations — no cost

Since the four cells have completely different prices, you must evaluate by "how much did we earn and lose" (expected profit), not "how many times were we right" (accuracy). Churn prediction for a subscription service works exactly the same way: missing a churner (FN) and giving a coupon to a perfectly happy customer (FP) carry different prices.

## 3. Core concepts

### 3.1 Confusion matrix refresher and the two metrics

With positive = churn in churn prediction:

- **Recall** = TP / (TP+FN): the share of actual churners the model found in advance. The "don't-miss-anyone" ability
- **Precision** = TP / (TP+FP): the share of customers the model flagged as risky who really churn. The "no-false-alarms" ability

### 3.2 Putting a unit price on each cell

Three business numbers are usually enough for the translation.

| Item | Example value | Source |
|---|---|---|
| Customer retention value V (revenue kept if saved) | monthly fee × remaining months | Finance/CRM |
| Intervention cost C (coupon + outreach cost) | 12,000 KRW per person | Marketing |
| Intervention success rate s (probability the coupon makes them stay) | 30% | Past campaigns |

Then the profit of each cell is:

- TP: `+V×s − C` (you save some of them; the cost is spent for sure)
- FP: `−C` (money spent on a customer who was fine)
- FN: `0` (no intervention, so no extra spend — but you lost the chance to save them: an opportunity cost)
- TN: `0`

The model's total expected profit = `TP×(V×s−C) − FP×C`. This formula runs through this level and all of level11.

### 3.3 The "1 pp of recall = how much money" formula

With N total customers and churn rate r, the actual churners number N×r. If recall rises 1 percentage point, the extra churners you catch are `N × r × 0.01`. Since the net gain per catch is `V×s − C`:

**Value of 1 pp of recall ≈ N × r × 0.01 × (V×s − C)**

Example: with 100,000 customers, an 18% churn rate, V=179,000 KRW, s=30%, C=12,000 KRW → 100,000 × 0.18 × 0.01 × (53,700−12,000) ≈ **about 7.5 million KRW**. Suddenly "1 pp of recall" is a concrete budget-language number.

Caveat: when precision drops, FP costs rise — so pushing recall ever higher is not always a win. Where to stop is calculated in level11 (the optimal threshold).

### 3.4 Build a "metric dictionary" for your organization

This translation isn't a one-off calculation — it should become the organization's shared vocabulary. Practical tip: pin a conversion table into the project wiki, like "1 pp of recall = about 7.5M KRW/year; 1 pp drop in precision = about N more FPs = about M KRW." From then on, every performance report and target negotiation happens on top of that table, and for the first time the data team and the business team speak in the same units.

## 4. Hands-on — main.py

```bash
python3 main.py
```

- [1] Splits the 2,000 customers of `churn_table` into train/test and trains a logistic regression.
- [2] Prints the test confusion matrix (TP/FP/FN/TN).
- [3] Defines the business unit prices (V, C, s), converts the four cells into money, and computes the model's expected profit.
- [4] Puts two comparison strategies side by side ("do nothing", "coupon everyone") to see what the model is actually worth.
- [5] Computes the value of 1 pp of recall with the formula, scaled up to a 100,000-customer service and an annual amount.

The heart of the code is the `profit_of()` function: it takes the confusion-matrix counts and the three unit prices and returns profit with one line of arithmetic. Model evaluation is, in the end, arithmetic — that's the message of this level.

## 5. Try it yourself

1. **(Easy)** Raise the intervention cost `COST_C` from 12,000 to 40,000 KRW and rerun. How does the gap between the "coupon everyone" strategy and the model strategy change?
2. **(Medium)** Lower the success rate `SUCCESS_S` to 10% — using the model can then turn a loss. Compute the break-even success rate (the s where profit = 0) by hand. Hint: solve TP×(V×s−C) = FP×C for s.
3. **(Challenge)** Pick one classification problem from your own work (defect detection, delinquency prediction, ...) and fill in the table from 3.2 with real values. If you don't know a value, just writing down "who I'd have to ask" already counts as half the win in practice.

## 6. Common mistakes

- **Feeling safe because FN costs 0**: there's no accounting expense, but the opportunity cost (lost revenue) makes it the most expensive cell. In reports, list "missed churners × V×s" alongside as the opportunity loss.
- **Giving up on the translation because you don't know the unit prices**: estimates of V, C, s are plenty. Report a range with a conservative and an optimistic scenario.
- **Reporting accuracy**: at an 18% churn rate, predicting "everyone stays" already scores 82% accuracy. "Accuracy is high" carries no information.
- **Using test-set profit directly as an annual forecast**: don't forget to scale sample size and time window to the real customer count and a full year (see main.py step [5]).

## Next level preview

To build a model worth putting into the money calculation, you first have to know the data. In level02 we learn an EDA workflow done as a procedure rather than a gut feeling (univariate → bivariate → time), and build an automated EDA report.
