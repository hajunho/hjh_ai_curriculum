# Lecture 07 · Level 11 — From Model to Decision — Cost-Benefit Optimization

> Lay expected-value math on top of churn probabilities to find the profit-maximizing threshold for "who gets the coupon," and close out the lecture with a campaign-ROI simulation.
**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level01, level09 / **Estimated time** 60 min

## 1. Why this matters — the business view

The model's output is a probability; what the company wants is profit. The last bridge between the two is **decision optimization**. With the same model, the annual profit difference between "coupon everyone above 50% probability" and "coupon everyone above 22%" can run to tens of millions of KRW. One threshold can be worth more money than a respectable model improvement.

The surprising fact: the optimal threshold has a theoretical value that comes **from the business unit prices alone**, independent of model performance. Given intervention cost C, customer value V, and intervention success rate s, the break-even probability is `p* = C/(V×s)` — one line of arithmetic. Finish this level and you'll be able to take a probability file from the data team, compute "how far down to intervene" yourself, and write the expected ROI into the campaign plan as a number. This is the destination of all 12 levels of this lecture.

## 2. An analogy to hold onto

It's the umbrella vendor's dilemma. The weather forecast (the model) says "30% chance of rain tomorrow." Do you set up the stall? The answer can't come from the probability alone. It depends on **the money lost if you open and it doesn't rain (cost C), and the money earned if you open and it rains (revenue R)**. If the pitch fee is 10,000 KRW and a rainy day nets 100,000 KRW, opening pays whenever the probability exceeds just 10% (expected value 0.1×100,000 = 10,000 > the pitch fee). If the pitch fee is 50,000 KRW, you need better than 50%.

A coupon campaign works the same way. The expected profit of couponing a customer with churn probability p is `p×(V×s) − C`. Give coupons only to customers for whom that's positive. The forecaster's (model's) job ends at making the probability accurate; the merchant's (decision-maker's) job is multiplying that probability by the price tags. Fail to connect the two roles, and even a perfect forecast makes no money.

## 3. Core concepts

### 3.1 The expected-value calculation

Intervene (coupon + outreach, cost C) on a customer with churn probability p:

- With probability p they really were going to churn → you save V at success rate s: expected revenue `p × V × s`
- With probability (1−p) they weren't leaving → the coupon is spent for nothing
- The cost goes out for certain: `−C`

Therefore **expected profit of intervening = p×V×s − C**. Not intervening earns 0. So intervening only on customers with `p×V×s − C > 0`, i.e. **p > C/(V×s) ≡ p\***, maximizes profit. With V=179,000 KRW, s=30%, C=12,000 KRW: p\* = 12,000/53,700 ≈ **22.3%**. The 0.5 threshold is just a convention with no justification behind it.

### 3.2 Cross-checking theory against experiment

main.py finds the optimal threshold from two directions. ① Theory: p\* = C/(V×s). ② Experiment: sweep the threshold from 0 to 1 on the test data, computing realized profit, and find the maximum. If the two agree, the calculation can be trusted; if they diverge sharply, it's a signal the model's probabilities are not **calibrated** (customers labeled "70% probability" actually churn only 40% of the time, etc.). Uncalibrated probabilities can't serve as inputs to expected-value math, so in practice checking the calibration curve is a precondition for this calculation.

### 3.3 Translating into campaign ROI

The final form for an executive report is not a threshold but ROI: `ROI = net profit / cost invested`. Written like "spend 14.88M KRW of coupon budget on the 1,240 highest-risk customers for an expected net profit of 19M KRW — ROI 128%." Attach conservative/base/optimistic scenarios (s at 20/30/40%) and the campaign plan is complete.

### 3.4 The limits of this calculation — two honest footnotes

First, s (the intervention success rate) should come from past campaigns or an A/B test. If it's an estimate, report a range via scenarios. Second, this model picks "people likely to churn," but ideally you'd pick "people who will change their mind only if given a coupon" (persuadable customers). Coupons spent on people who were staying anyway, or leaving anyway, are waste. Just know that a more advanced topic called uplift modeling addresses this directly — it needs control-group experiment data and is beyond this lecture's scope.

## 4. Hands-on — main.py

```bash
python3 main.py
```

- [1] Trains a churn model on `churn_table` and gets churn probabilities for the test customers.
- [2] Computes the theoretical break-even probability p\* from the business unit prices (V, C, s).
- [3] Sweeps the threshold from 0 to 1, computing realized profit on the test data, and finds the optimum.
- [4] Strategy comparison table: do nothing / coupon everyone / convention (0.5) / theoretical p\* / empirical best — five strategies' profits side by side.
- [5] Saves the profit-curve PNG to `outputs/` (with the theoretical p\* and empirical optimum marked).
- [6] Scales the campaign ROI at the optimal threshold to a 100,000-customer service across 3 scenarios (s=20/30/40%) and prints the reporting sentences.

Key observations: the profit gap in [4] between threshold 0.5 (convention) and the optimum, and how close the theoretical p\* lands to the empirical best. The model is unchanged — only the decision changed — and profit multiplies. That is this level in its entirety.

## 5. Try it yourself

1. **(Easy)** Raise the coupon cost `COST_C` to 30,000 KRW. How do p\* and the empirical optimum move? Check it against the intuition "the pricier the intervention, the surer you must be."
2. **(Medium)** Add a "top 100 probabilities only" (budget-constrained) strategy to the table in [4]. Explain the difference between profit-maximizing and budget-constrained strategies, connecting to level10's alert-budget concept.
3. **(Challenge)** Assume the success rate differs by customer. E.g., new customers under 6 months of tenure: s=45%; long-tenured customers: s=20%. Re-select the intervention targets with per-customer expected profit `p×V×s_i − C` and check whether profit beats the single-s version.

## 6. Common mistakes

- **Treating the 0.5 threshold as sacred**: 0.5 is just the library default, justified only when the cost structure is symmetric. Real-world costs almost never are.
- **Computing expected value from uncalibrated probabilities**: probabilities from models trained with class_weight or resampling tend to be inflated upward. The ranking stays valid, but the absolute values need adjustment — cross-check the empirical optimum against theory like main.py [3], or apply calibration.
- **Choosing the threshold on test data and reporting profit on the same data**: the standard is to pick the threshold on validation data and estimate final profit on a sealed test set.
- **Assuming s optimistically**: a success rate of 30% vs 15% is the difference between a profitable and a money-losing campaign. Always report a scenario range, and measure s with a control group in the live campaign.
- **Spending the whole budget on model improvement and neglecting decision design**: moving the threshold is often far cheaper and faster than raising AUC by 0.01. Order of operations: decision optimization first, model improvement second.

## Next level preview

lecture07 is complete. You can now run a full lap of a data project — from framing the problem to the final calculation that turns probabilities into money. In the next lecture (lecture08 — Deep Learning Foundations) we step beyond tabular data into the world of neural networks for images and text.
