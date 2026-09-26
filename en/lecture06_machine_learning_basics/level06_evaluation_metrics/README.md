# Lecture 06 · Level 06 — Evaluation Metrics — The Accuracy Trap

> We build a "98.5% accurate" model that catches not a single fraud case with our own hands, then learn to read models properly with the confusion matrix, precision, recall, F1, and ROC-AUC.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level05 / **Estimated time** 45 min

## 1. Why Learn This — The Business View

This level is the minimum literacy needed to avoid being fooled by ML reports. "A fraud-detection model with 98.5% accuracy" sounds impressive — but on data where fraud is 1.5% of cases, **a hollow model that insists "everything is normal" also scores 98.5% accuracy**. Projects worth hundreds of thousands of dollars really do get approved on top of this illusion.

Someone who knows evaluation metrics asks differently: "What's the recall? The precision? At that threshold, how many false alarms per day?" Being able to ask those questions is what separates people who merely consume ML results from people who can verify them. And metric selection isn't a technical question — it's a **business profit-and-loss question**. The person who knows the cost of one missed fraud and the cost of one false alarm should choose the metric.

## 2. Understanding by Analogy

Think of **airport security screening**. Only a handful of passengers in 10,000 carry anything dangerous. If the checkpoint shouts "everyone through!", its accuracy exceeds 99.9% — yet that checkpoint has no reason to exist. A checkpoint's skill must be measured with two questions.

1. **Of the passengers carrying something dangerous, what % did we catch?** — That's recall. Missing one means a major incident.
2. **Of the people the alarm flagged, what % were real?** — That's precision. Frequent false alarms make the lines longer, and eventually nobody trusts the alarm at all.

The two metrics sit on a seesaw. Crank the checkpoint's sensitivity to maximum (alarm on every belt and watch) and recall hits 100% while precision collapses; dial it down and the reverse. Where to sit isn't decided by the machine but by **the operator who knows the cost of a miss versus the cost of a false alarm**. And there's a separate metric for "regardless of where you set the sensitivity, how discerning is this checkpoint itself?" — that's ROC-AUC.

## 3. Core Concepts

### 3.1 The Confusion Matrix — the Source of Every Metric

Prediction × reality has only 4 cells.

| | Predicted: fraud | Predicted: normal |
|---|---|---|
| **Actual: fraud** | TP (caught) | FN (missed — fraud losses incurred) |
| **Actual: normal** | FP (false alarm — customer friction, investigation cost) | TN (passes quietly) |

Every metric comes from these 4 cells.

- **Accuracy** = (TP+TN) / total — on imbalanced data TN dominates, creating the illusion
- **Precision** = TP / (TP+FP) — "of the alarms, the fraction that were real"
- **Recall** = TP / (TP+FN) — "of the real ones, the fraction we caught"
- **F1** = the harmonic mean of precision and recall — the compromise when one number is required. Being a harmonic mean, if either side nears 0 the F1 collapses too (designed so averaging can't lull you).

### 3.2 Why Accuracy Breaks Down under Imbalance

Accuracy rises just from getting the majority class right. With 1.5% fraud, predicting everyone normal = 98.5% accuracy. Meanwhile recall is 0% and precision is undefined (zero alarms). **On imbalanced data, accuracy belongs in a footnote, not the headline.**

### 3.3 ROC-AUC — Discernment Independent of the Threshold

As level05 taught, the model outputs probabilities and the threshold makes decisions. Sweep the threshold from 0→1 and the (false-alarm rate, recall) points trace a curve — the ROC curve — and the area under it is the AUC. Its intuitive reading is excellent: **"if you draw one fraud case and one normal case at random, the probability the model assigns the higher score to the fraud."** 0.5 is a coin flip; 1.0 is perfect. Use it to compare "the caliber of the model itself" before fixing a threshold. But a high AUC can still coexist with low precision at your operating threshold, so a final report must also attach the confusion matrix at the operating point.

### 3.4 Derive the Metric Backwards from the Action

- Investigators can process only 100 cases a day → "precision of the top 100" is the real metric
- Miss cost ≫ false-alarm cost (cancer screening, fraud) → prioritize recall; precision only needs a floor
- False-alarm cost ≫ miss cost (wrongly blocking a VIP) → prioritize precision

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

Using fraud_table (8,000 card transactions, ~1.5% fraud) we demonstrate the accuracy trap.

- **[1]** Load the data + split train/test (using `stratify` to preserve the imbalance).
- **[2] The hollow model**: confirm that a model that always outputs "normal" scores about 98.5% accuracy. Then print its confusion matrix — recall 0% — and the accuracy trap is laid bare in a 4-cell table.
- **[3] The real model**: train a logistic regression and produce the same metrics. The accuracy looks similar to the hollow model's, but confirm that the confusion matrix, precision, recall, and F1 are worlds apart.
- **[4] The threshold seesaw**: sweep the threshold 0.9→0.1 and print the table of precision and recall trading places.
- **[5] ROC-AUC**: compare the hollow model's AUC (0.5) with the real model's, and verify the interpretation "probability that a random fraud/normal pair has the fraud scored higher" by simulation.

The heart of the code: `confusion_matrix`, `precision_score/recall_score/f1_score`, `roc_auc_score`. And in [5], `auc_by_sampling()` — a function that verifies AUC's probabilistic definition directly by drawing 10,000 random pairs. Once you see it nearly match the library's number, AUC will never feel like a cipher again.

## 5. Try It Yourself

1. **(Easy)** From the table in [4], if "we can only investigate 40 cases a day," which threshold would you pick? (Hint: look at the alarm-count column.)
2. **(Medium)** Assume a missed fraud costs 2,000,000 KRW on average and a false alarm costs 20,000 KRW to investigate. Add code that computes total cost per threshold in [4] (FN×2,000,000 + FP×20,000) and find the lowest-cost threshold.
3. **(Challenge)** Give the logistic regression `class_weight="balanced"` and retrain. How do recall and precision change? Look up the docs and explain what this option does on imbalanced data.

## 6. Common Mistakes

- **Reporting accuracy on imbalanced data**: the reason this level exists. Always report the baseline ("all majority class" model) alongside.
- **Looking only at F1 and never at precision/recall**: an F1 of 0.6 means completely different business realities depending on whether it's "precision 0.9 + recall 0.45" or "both 0.6."
- **Thinking a high AUC means you're done**: AUC measures caliber, not the operational report card. Only with the operating threshold's confusion matrix is the report sign-off-ready.
- **A split whose test set doesn't reflect the imbalance**: cut without `stratify` and only a few fraud cases land in the test set, making the metrics jump around.
- **Starting to model before fixing the metric**: this was element 4 of level02's spec. The metric is a contract signed up front, not a choice made after the fact.

## Next Level Preview

The models so far (logistic regression) are "scorecards" — easy to explain, but they only capture straight-line relationships. In level07 we learn decision trees, which branch by asking questions like twenty questions — a model that generates human-readable rules on its own.
