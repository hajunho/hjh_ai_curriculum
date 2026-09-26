# Lecture 06 · Level 08 — Random Forests and Ensembles

> Instead of one unstable tree, grow hundreds of slightly different trees and let them vote — we verify by experiment why that makes the model stronger: the principle of collective intelligence.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level07 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

For real-world prediction problems on tabular data — churn, delinquency, demand, conversion — the random forest is the "powerful default you just run first." It's undemanding about preprocessing, delivers respectable performance with hardly any hyperparameter tuning, and is comparatively resistant to overfitting. That's why it held the throne of "first-submission model" for so long, in data competitions and corporate practice alike.

From a business standpoint, this level's other payoff is gaining **stability** as an axis of evaluation. If the model built on this month's data and the one built on next month's say completely different things, the organization won't trust it no matter how well it performs. Here you build the habit of asking "how much do the results wobble?" as seriously as "how good is the performance?"

## 2. Understanding by Analogy

The **ox-weighing contest** story is famous. At a county fair, hundreds of spectators each guessed the weight of an ox. Individually they were far off, but **the average of all guesses was more accurate than the experts**. Because each person's errors pointed in different directions, averaging canceled them out. That is collective intelligence — and the principle of the ensemble.

There's a condition, though: **the participants must think independently**. If everyone copies one person's answer, averaging preserves that one person's error intact. The random forest uses two devices to keep its trees from resembling each other.

1. **Bagging**: each tree gets its own dataset, redrawn from the original **with replacement** at the same size (the bootstrap). Each tree ends up with a slightly different "life experience."
2. **Random feature selection**: at each split, only a subset of the features is offered as candidates. Without this, every tree would open with the same strongest feature and grow alike. In meeting terms, it's the rule "the loudest person doesn't get to speak first."

Hundreds of trees grown this way produce the final answer by vote (classification) or average (regression).

## 3. Core Concepts

### 3.1 Why Averaging Reduces Variance

A single tree's error splits into "systematic bias + moment-to-moment wobble (variance)." The tree instability you saw in level07 is exactly that variance. Average T near-independent trees and the wobble shrinks roughly by 1/√T — the same statistical principle as coin flips converging to 50% heads as you flip more. Bias doesn't shrink through averaging, so the forest's strategy is "average many deep trees (low bias, high variance)." Individual trees may overfit, but their average largely tames it.

### 3.2 The Random Forest's Main Knobs

- `n_estimators` (number of trees): more is more stable. Performance converges past a point, so 100–500 is typical.
- `max_features` (feature candidates per split): smaller makes the trees more different from each other (diversity up, individual skill down).
- `max_depth`, `min_samples_leaf`: the same brakes as for a single tree. In a forest you can leave them fairly loose.

### 3.3 Feature Importance and OOB

The forest averages its trees' impurity-reduction contributions into a feature importance. It's far more stable than a single tree's importance, so it shows up often in business reports (though the bias warning from level07 still applies — the fairer permutation importance comes in level11). As a bonus, each tree can score itself on the data it never saw during its bootstrap draw (OOB, out-of-bag), giving a rough performance estimate without a separate validation set.

### 3.4 What You Give Up: Interpretability, Weight

A single tree's rules could be read aloud; a vote among 300 trees cannot. Model size and prediction time also grow. **The trade between "one readable tree" and "a strong, stable forest"** — which to pick is decided by the use (reporting vs operations). The interpretation techniques for having both are level11's topic.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

On churn_table we compare a single tree vs a random forest on both performance and stability.

- **[1]** Data prep.
- **[2] Mini-bagging by hand**: before using sklearn's forest, we build "grow 25 deep trees on bootstrap resamples and let them vote" in 15 lines and compare it to a single deep tree. This step confirms the power of averaging is not library magic.
- **[3] Performance comparison**: compare test AUC and recall for single trees (capped/uncapped depth) vs `RandomForestClassifier`.
- **[4] Stability comparison**: re-measure the AUC of a single tree and the forest over 12 different random splits, comparing the "wobble" as mean ± standard deviation. The key observation is the forest's visibly smaller standard deviation.
- **[5]** Print the forest's feature importances beside the single tree's — which looks more plausible and stable?

The heart of the code: `bagged_predict()` in [2] — building bootstrap samples with `rng.choice(n, n, replace=True)` and voting by averaging the trees' probabilities. sklearn's one-liner (`RandomForestClassifier(n_estimators=300)`) is this principle plus random feature selection.

## 5. Try It Yourself

1. **(Easy)** Change `n_estimators` in [3] to 10, 50, 300. How do performance and runtime change? Confirm "tree count converges past a point."
2. **(Medium)** In [2]'s mini-bagging, grow the tree count 1→5→25→100 and record the test AUC. Can you see the 1/√T-flavored diminishing returns?
3. **(Challenge)** Train the forest with `max_features=None` (all features at every split) and redo the stability experiment in [4]. This tests section 2's claim that without random feature selection, the trees grow alike and the ensemble effect shrinks.

## 6. Common Mistakes

- **Overconfidence that "forests don't overfit"**: it's mitigation, not immunity. With little data or a leaky feature, they're fooled just the same.
- **Concluding from one split without measuring stability**: a one-off difference (e.g., AUC 0.821 vs 0.815) can be split luck. Speak in mean ± standard deviation over repeated measurements.
- **Reporting feature importance as causal or absolute rankings**: correlated features share importance between them (dilution). Low importance does not mean "no effect."
- **Offering only the forest where interpretation is required**: for executives, one depth-3 tree diagram often persuades better than a forest's scorecard. Pick the model that fits the use.

## Next Level Preview

We've brushed against overfitting many times, but always out of the corner of the eye. In level09 we face it head-on — raising polynomial degree until you witness the train/validation errors split apart on a chart, then taming it with the brakes called regularization (L1/L2).
