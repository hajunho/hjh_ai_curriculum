# Lecture 07 · Level 09 — Case Study 2 — Customer Churn Prediction

> EDA → features → cross-validation → interpretation → action list: assemble every part learned in this lecture and complete a churn-prediction project. The deliverable is "top-10 at-risk customers + a recommended action for each."
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level01–06 / **Estimated time** 60 min

## 1. Why this matters — the business view

It's marketing's oldest maxim that acquiring a new customer costs far more than keeping an existing one. So in subscriptions, memberships, telecom, and finance alike, churn defense — "find the customers about to leave and hold on to them" — is the data team's perennial project number one.

The heart of this level is not technique but **completeness**. In the real world, a project that ends with "achieved AUC 0.83" causes nothing to happen. It becomes a finished project only when it's shaped into something the marketing team can pick up and act on — **who (the list), why (the reason), how (the action)**. That's why main.py's final output is not a performance metric but a "top-10 at-risk customers with recommended actions" table.

## 2. An analogy to hold onto

Picture the veteran manager of a gym with lots of regulars. A month before a member quits, the manager feels the signals: "their attendance has been thinning out," "they asked about fees twice last week," "that newish member is already losing steam." And when a signal appears, the response is tailored — a check-in message and a PT trial pass for the low-attendance member, a plan consultation for the fee-complainer.

A churn-prediction project turns this manager's instincts into a system. ① find the signals (EDA), ② turn signals into numbers (features), ③ combine signals into a risk score (model), ④ explain which signal is responsible (interpretation), ⑤ a response playbook per signal (the action map). Not one manager's gut, but that same sense running automatically every week across all 2,000 customers — that is the model's added value.

## 3. Core concepts

### 3.1 The project spec comes first

Per level00's gates, write the spec before the code. Prediction target: churn this month, yes/no. Use: a weekly CRM campaign to the highest-risk customers. Target metric: precision 25%+ at recall 55%+ (campaign budget constraint; random sends hit about 15%, so that's 1.7×). Baseline: random sends. This spec becomes the yardstick for every later technical choice (threshold, metrics).

### 3.2 Risk scores are probabilities

Using `predict_proba()` (churn probability) rather than `predict()` (a 0/1 call) is the professional standard. Only with probabilities can you ① sort by risk and intervene from the top until the budget runs out, and ② design actions of different intensity by probability band (80%+ gets a phone call, 50–80% gets a coupon). The 0/1 call is just the special case of one threshold laid over the probability.

### 3.3 Interpretation: reading coefficients as odds ratios

A logistic regression's standardized coefficients give the direction and size of "does risk rise or fall as this feature grows?" `exp(coefficient)` is the odds ratio: "if this feature grows by one standard deviation, churn odds multiply by how much?" For non-specialist reports, translate into direction-plus-ranking language like "more support calls means more risk (the strongest signal)." One more thing — **an individual customer's risk reason** can be approximated by "how far that customer's feature value deviates from the mean in the risky direction" (standardized value × coefficient). The "main driver" column in main.py is this calculation.

### 3.4 The action map — design outside the model

The model only answers "who is at risk." "So what do we do?" is a response rulebook designed by humans. Example: main driver "usage dropped sharply" → re-engagement content/perks; "surge in support calls" → priority CS consultation; "no auto-pay" → an incentive to switch to auto-pay. The driver-to-action map belongs to domain knowledge, and it's what turns a list into a campaign.

### 3.5 How do you verify impact?

If you drew a list and ran the campaign, verify the effect against a **control group you didn't touch** (randomly exclude some at-risk customers). Intervene on everyone and you'll never know "was it the model, or were they never going to leave anyway?" This thread continues into level11's expected-value calculations.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Runs the complete project on the 2,000 customers of `churn_table`.

- [1] Prints the problem spec (prediction target, use, target metric).
- [2] Mini-EDA: checks the target ratio and the top churn-vs-stay group-gap signals.
- [3] Prepares features + adds derived features (`usage_per_tenure`, etc.).
- [4] Reports AUC mean ± std with 5-fold cross-validation, then finalizes the model on a train/test split.
- [5] Interpretation: prints the standardized coefficients with their direction (risk up / risk down).
- [6] Deliverable: prints the table of the 10 highest churn probabilities among test customers — customer ID, churn probability, main driver, recommended action.

The highlight of the output is [6]. Take away the sense that the final deliverable is not a performance number but "a table the marketing team can receive on Monday morning and execute as-is."

## 5. Try it yourself

1. **(Easy)** Grow the top-10 to a top-20 and check whether customers below 50% probability enter the list. If they do, weigh "should we spend intervention budget this far down?" using level01's expected-profit lens.
2. **(Medium)** Modify the output rule in [6] so actions for customers at 80%+ probability get a `[CALL]` tag and the rest get `[MESSAGE]`. It's practice translating 3.2's "intensity by probability band" into code.
3. **(Challenge)** Swap logistic regression for RandomForest and replace the interpretation in [5] with `feature_importances_`. Note the difference — importances don't tell you direction (whether risk goes up or down) — and count how much the two models' top-10 lists overlap.

## 6. Common mistakes

- **Ending the project with a performance report**: "AUC 0.83" is an intermediate artifact. Only list + reason + action gets the business moving.
- **Feeding customer_id in as a feature**: an ID is an identifier with no generalizable signal. It only invites overfitting by memorizing coincidences.
- **Scoring all customers with a model trained on all data, then reporting performance on the same data**: performance must be measured on data unused for training. Separate production scoring from performance validation.
- **Delivering a list with no actions**: throw over "here are 500 at-risk customers" and the business won't know what to do. Cause and recommended response are part of the data team's job.
- **Claiming intervention impact with no control group**: "the campaign group churned less" may not mean the model is good — you may simply have given coupons to people who were never leaving.

## Next level preview

Case study three returns to fraud detection. In level10 we run supervised learning (which uses answer labels) side by side with IsolationForest (which hunts "strangeness" without labels), and design an operating threshold matched to the investigation team's alert budget.
