# Lecture 06 · Level 07 — Decision Trees

> We build a model that branches by asking questions like twenty questions, then print the learned tree as human-readable rule sentences and interpret it.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level05, level06 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

Logistic regression is excellent, but as a "scorecard" model it leaves two things wanting. First, it captures only straight-line relationships — it struggles with combined-condition patterns like "risk spikes only when usage days are under 10 AND the customer is new." Second, explaining it requires a translation step through coefficients and odds ratios.

The decision tree's charm is exactly the opposite. Its learned result **is itself a collection of "if ~ then → else →" rules**, ready to show to business staff and executives as-is. Asked "why was this customer flagged as risky?", you can answer "because usage days ≤ 9, auto-pay not enabled, and tenure under 8 months." When regulated industries (finance, healthcare) demand explainability, and when you want to import a model's discoveries back into business rules, the decision tree remains a first-choice candidate. It is also the building block of the next two levels (random forests, boosting).

## 2. Understanding by Analogy

Think of **twenty questions**. A skilled player doesn't blurt "is it an elephant?" They open with **the question that cuts the candidate pool in half the hardest**: "Is it alive?" → "Is it an animal?" → "Four legs?"

Decision-tree learning is exactly this strategy. 1,500 customers, churners and stayers mixed, sit in one room. The tree tries every possible question ("usage days ≤ 9.5?", "tenure ≤ 3.5 months?"…) and picks **the question that leaves the two resulting rooms purest**. Then it repeats the same thing inside each room. After a few splits you get rooms that are "almost all churners" and "almost all stayers," and a new customer walks down the questions to a room, where the majority vote decides the prediction.

The caveat is also the same as twenty questions. Allow unlimited questions and the tree can resort to "wait — are you John Smith?", building 1,500 one-person rooms — 100% right on the training data, useless memorization for new customers (level04's overfitting). So we cap the number of questions (the depth).

## 3. Core Concepts

### 3.1 Impurity — Scoring "How Mixed Is the Room"

A good question = the question that reduces the room's mixedness (impurity) the most. The standard measure is Gini impurity.

```
Gini = 1 − (churn share² + stay share²)
```

A room of one kind scores 0 (pure); fifty-fifty scores 0.5 (worst). For each candidate question the tree computes "the (size-weighted) average impurity of the two rooms after the question" and adopts the question with the maximum reduction. It's level00's threshold search applied recursively — here too, learning is search, not magic.

### 3.2 Depth and Overfitting

- **Depth 1–2**: 2–4 rooms. Catches only the big picture, but can never overfit. Good for "mining candidate business rules."
- **Depth 3–5**: the sweet spot for practical interpretation. Also the largest size a human can still read.
- **Unlimited depth**: training accuracy ~100%, test accuracy actually drops. The phenomenon you already witnessed in level04's experiment A.

Besides depth there are other brakes, like `min_samples_leaf` (a room's minimum occupancy — no one-person rooms).

### 3.3 Strengths and Weaknesses of Trees

**Strengths**: the rules are directly visible / no standardization or scaling needed (threshold comparisons are unit-free) / captures nonlinear, combined-condition patterns / handles mixed numeric and categorical data well.
**Weaknesses**: **instability** — change the data slightly and the very first question can change, reshaping the whole tree / staircase-style predictions are inefficient for smooth relationships / standalone performance is usually similar to or slightly above logistic regression. Solving this instability with "a vote among many trees" is next level's random forest.

### 3.4 The Rule Re-Import Use Case

A top split discovered by the tree ("usage days ≤ 9.5 AND auto-pay off → 45% churn rate") can be used directly as a CRM rule, no model deployment required. That's why tree analysis alone produces value even in organizations that never ship a model.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

We build and interpret a churn-prediction tree on churn_table.

- **[1]** Data prep and split (the code confirms that no standardization is needed).
- **[2]** Compute Gini impurity by hand: the whole room's impurity vs the average impurity after the question "usage days ≤ 9.5" → the reduction is the question's worth.
- **[3]** Train a depth-3 tree, print the full rules with `export_text`, then summarize the top 3 highest-churn leaf rooms by occupancy and churn rate. Since only 15% are churners, we used `class_weight="balanced"` to weight the minority class so churn-side rules surface in the tree — read the printed rules out loud from the top and trace the path to a risky room.
- **[4]** Vary the depth over 1, 3, 5, 10, unlimited and print the train/test accuracy table — revisiting the fork in the road to overfitting.
- **[5]** Print the feature importances: each feature's share of the impurity reduction.

The heart of the code: `DecisionTreeClassifier(max_depth=3)` and `export_text(tree, feature_names=...)`. Read the printed rule tree aloud from top to bottom — the business rules the model discovered become plain sentences. And the 8-line `gini()` function in [2] is the entirety of the math behind it all.

## 5. Try It Yourself

1. **(Easy)** Reduce `max_depth` in [3] to 2. How many rules remain, and how much test score do you lose? For an executive briefing, would you pick depth 2 or 3?
2. **(Medium)** Train with `min_samples_leaf=50` and unlimited depth. Is overfitting suppressed even without a depth cap? Explain in words how the two brakes differ.
3. **(Challenge)** Change the training data's seed from 7 to 8, retrain a depth-3 tree on the new data, and compare how much the rules change. This is the "tree instability" from 3.3. Keep this observation in mind and level08 will make far more sense.

## 6. Common Mistakes

- **Being impressed by a deep tree's training score**: an unlimited-depth tree's 100% training accuracy is memorization, not skill (level04).
- **Treating tree rules as permanent truths**: trees are unstable. When the data refreshes, the rules may change — give re-imported rules an expiration date.
- **Interpreting impurity importance as causation**: feature importance measures "how much the model leaned on it," not a ranking of causes. It also has a bias that overrates features with many distinct values (the alternative is level11's permutation importance).
- **Wasting time on scaling/one-hot worries in front of a tree**: it's a threshold-comparison model — no standardization needed. Remember that preprocessing needs differ per model.

## Next Level Preview

A single tree is easy to read but unstable. In level08 we make a trade — giving up some interpretability to buy performance and stability — with the random forest: "grow hundreds of slightly different trees and let them vote."
