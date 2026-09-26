# Lecture 06 · Level 11 — Gradient Boosting and Model Interpretation

> We raise performance with boosting — repeated study of a mistake journal — and learn to answer "why?" using permutation importance and per-prediction explanations.
**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level07, level08 / **Estimated time** 60 min

## 1. Why Learn This — The Business View

If you had to name a single reigning champion of tabular-data prediction, it's gradient boosting. The top ranks of data competitions and a large share of corporate churn, delinquency, and demand systems are boosting-family models. With sklearn's `HistGradientBoostingClassifier` alone, you get access to that world's standard technique.

But the stronger the model, the newer the problem: **"The model says this customer is risky — why?"** An agent can't make a call without a reason, an underwriter can't decline without a reason, and regulators demand explanations. So this final level has two halves — the first raises performance with boosting, the second reopens the black box (permutation importance, partial dependence, per-prediction explanations). Only when you can handle performance and explanation together does ML take root in an organization.

## 2. Understanding by Analogy

**The mistake-journal study method** is the whole of boosting. If the random forest (level08) is "300 students each study on their own, then vote," boosting is **a serial process where one student grows by keeping a journal of mistakes**.

Period 1: work through the first problem set. The grade is mediocre. Period 2: build a **journal of only the problems you got wrong** and drill just those. Period 3: whatever's still wrong goes into another journal… repeat hundreds of times. Each period's study (one small tree) is shallow and weak, but because they stack up in the direction of "correcting the mistakes so far," the final ability is very strong.

But it inherits the mistake-journal trap too. Start memorizing the problem set's **typos (the noise)** and you overfit. So boosting requires brakes — a learning rate that limits how much each period is absorbed, and early stopping — level09's philosophy returning verbatim.

Let's pin the interpretation tools to analogies too. **Permutation importance is "shuffling one subject's answer sheet."** Randomly shuffle one feature's values, feed it to the model, and if the score plummets, the model leaned on that feature heavily; if it doesn't, the feature was dispensable.

## 3. Core Concepts

### 3.1 The Structure of Boosting

Prediction = tree1 + η·tree2 + η·tree3 + … (η = learning rate). Each tree is trained to predict "the error left over by the predictions so far" (residual correction, seen through a regression lens — the *gradient* in the name refers to this error direction). The contrast with random forests is the key.

| | Random forest | Gradient boosting |
|---|---|---|
| Tree relationship | Parallel, independent (voting) | Serial, dependent (mistake correction) |
| Individual trees | Deep and strong | Shallow and weak (depth 3–8) |
| Main effect | Reduces variance (wobble) | Reduces bias (systematic error) |
| Overfitting risk | Fairly low | **High if untuned** |
| Main knobs | Tree count | Learning rate × tree count × depth |

`HistGradientBoosting` is a modern implementation that bins continuous values into histogram buckets for speed, with built-in early stopping (`early_stopping`) that halts on its own when the validation score stops improving.

### 3.2 Permutation Importance — a Fairer Importance

The impurity importances of levels 07–08 were biased, measuring "how much a feature got used during training." Permutation importance **interrogates the finished model with test data**: shuffle just one feature's column (leaving the others intact) and measure the performance drop. It works regardless of model type, and since it measures "actual contribution to predictive power," it's more honest for reporting. Caveat: highly correlated features can stand in for one another, so both may come out low.

### 3.3 Partial Dependence — One Feature's Influence Curve

Partial dependence is the curve of "holding everything else as-is, how does the average prediction change as one feature's value moves?" — "as usage days go 0→30, what shape does the churn probability fall in?" Unlike logistic regression, which compresses this into one coefficient, it reveals boosting's nonlinear relationships (e.g., a sharp spike below 10 days). It remains **the shape of the association the model learned**, not causation.

### 3.4 Per-Prediction Explanations — "Why This Customer?"

Apart from global tendencies (importance, partial dependence), you often need the grounds for one specific customer's verdict. The industry standard is contribution-decomposition tooling (the SHAP family), but you can implement the principle yourself today: for each feature, measure **"if I swap this customer's value for an average one, how much does the prediction change?"** The features with the biggest swings are the main grounds for that customer's verdict. With just this concept, you can read the professional tools' output too.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

We build a boosting model on churn_table and run the interpretation to the finish line.

- **[1]** Data prep, train/test split.
- **[2]** Compare test AUC across three models — logistic (level05), random forest (level08), `HistGradientBoostingClassifier` — the graduation exam for everything learned so far. Spoiler: on this synthetic data, logistic wins. The data's true structure is a linear scorecard, and the real observation of this comparison is "it's not the strongest model that wins, but the model that fits the data's structure."
- **[3]** A mini-experiment on the learning rate × tree count relationship: lower the learning rate and more trees are needed, but results tend to stabilize.
- **[4] Permutation importance**: use `permutation_importance` to print each feature's AUC drop and compare rankings against impurity importance (for reference).
- **[5] Partial dependence**: use `partial_dependence` to print the influence curves of usage days and support calls as text graphs — read off where the risk changes sharply.
- **[6] Per-prediction explanation**: pick one high-risk customer and print, per feature, "the churn-probability change when this feature is swapped to the average value." A sentence like "4 support calls → average level would lower the probability by N points" becomes the basis of a call script.

The heart of the code: `HistGradientBoostingClassifier(early_stopping=True)`, `permutation_importance(..., scoring="roc_auc")`, `partial_dependence(...)`, and [6]'s `explain_one()` — a 12-line swap-based explainer.

## 5. Try It Yourself

1. **(Easy)** In [3], try `learning_rate=1.0` (a mistake journal with no brakes). What happens to the score? Diagnose overfitting from the train/test scores.
2. **(Medium)** Apply [6]'s explainer to the lowest-risk customer (minimum churn probability). Can it also explain "what makes this customer safe"?
3. **(Challenge)** Look at [4]'s permutation importance with the standard deviation over 10 repeats (`n_repeats`), and add code that flags feature pairs whose importance difference is within one standard deviation as "ranking not determinable" — an application of level08's "report the wobble too."

## 6. Common Mistakes

- **Running boosting on defaults and walking away**: it's more tuning-sensitive than a random forest. At minimum, adjust learning rate, early stopping, and depth via validation.
- **Skipping explanation because performance is good**: a model that never gets adopted has an effective AUC of zero. Package the performance table with the interpretation (importance, partial dependence, case explanations) in every report.
- **Mixing up permutation and impurity importance in write-ups**: they have different definitions and can rank differently. State which one you used.
- **Promoting partial dependence or contributions to causation**: it's not "reduce the calls and churn falls" but "the model views customers with many calls as risky." Intervention effects are proven only by experiments.
- **Explaining on the training set**: permutation importance and performance reporting must be based on test (or validation) data to be free of the optimism bias (level04).

## Next Level Preview

Lecture06 ends here. Starting from rules vs learning, through regression, classification, evaluation, overfitting, unsupervised learning, boosting, and interpretation — the full skeleton of tabular-data machine learning is now standing. In the next lecture (lecture07, Business Machine Learning in Practice), we harden these tools against real work scenarios: leakage traps, imbalance, and post-deployment performance decay.
