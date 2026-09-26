# Lecture 07 · Level 07 — Handling Imbalanced Data

> How to catch fraud when only 1.5% of 5,000 transactions are fraudulent — class_weight, resampling, threshold tuning, and reading the trade-off from the PR curve.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level01, level06 / **Estimated time** 45 min

## 1. Why this matters — the business view

Almost every classification problem with real money on the line is imbalanced. Fraudulent transactions 1.5%, equipment failures 0.5%, critical defects 2%, false claims 1% — the more it matters to catch something, the rarer it is. But apply standard settings to a rare-event problem and the model settles into the lazy optimum of answering "everything is normal." Accuracy 98.5%, frauds caught: zero — a model with dazzling numbers that does nothing.

This level has two goals. First, get your hands on the three techniques that wake up a lazy model (weights, resampling, thresholds). Second, internalize — via tables and curves — that recall and precision are a **trade-off**. Only when you can answer the demand "get me 90% recall" with "then precision drops this much and the investigation team's workload multiplies" does a realistic target negotiation begin.

## 2. An analogy to hold onto

It's finding needles in a haystack: 5,000 handfuls of hay, 75 needles. Tell the inspector (the model) only "minimize your mistakes," and the most comfortable strategy is to **pick up nothing** (only 75 errors — 98.5% correct). There are three ways to get the inspector moving.

- **Weights (class_weight)**: rewrite the penalty table — "missing one needle costs a 65× penalty." Now the inspector fears missing a needle and grabs anything suspicious.
- **Resampling**: doctor the training haystack. Duplicate the needles (oversampling) or remove hay (undersampling), so the inspector practices in a world where needles are common. The exam (the test set) is still taken at the real-world ratio.
- **Threshold tuning**: instead of grabbing only when the "suspicion score" is above 50, lower the bar to 20. Grabbing more finds more needles (recall up) — but also more hay (precision down).

None of them is free. Catch more and false alarms grow too — and the tool for seeing that trade-off with your own eyes is the PR curve.

## 3. Core concepts

### 3.1 Why accuracy becomes meaningless

With 1.5% positives, predicting "all negative" scores 98.5% accuracy. That's why the default metrics for imbalanced problems are recall, precision, and PR-AUC. ROC-AUC is used too, but when positives are extremely rare the ROC curve tends to flatter — the **PR curve (precision-recall curve)** is more honest.

### 3.2 The three techniques compared

| Technique | What it does | Advantage | Watch out |
|---|---|---|---|
| class_weight="balanced" | big penalty in the loss for minority-class mistakes | one line, data unchanged | only for models that support it |
| Oversampling | duplicate minority class in the training data | works with any model | **after the split, training data only!** repeated samples risk overfitting |
| Threshold tuning | move the predict_proba cutoff | no retraining, adjustable in production | must be set on validation data (setting it on the test set is leakage) |

An advanced variant called SMOTE (synthesizing minority samples by interpolation) is also widely used. It needs a dedicated package, so here we just note the concept and verify the principle with simple duplication oversampling.

### 3.3 The golden rule of resampling

Resampling is applied **only after the train/test split, and only to the training data**. Oversample before splitting and copies of the same sample land on both sides, badly inflating performance (a variant of leakage). And never doctor the test set's ratio — the exam must be taken at the real world's ratio.

### 3.4 How to read a PR curve

Sweep the threshold from 1.0 down to 0.0, plotting (recall, precision) points, and you get the PR curve. The shape — going right (recall up) pulls the curve down (precision down) — is the trade-off made visible. The area under it (PR-AUC, average precision) is the model's overall score, and **which point to operate at** is decided not by the model but by the business (investigation staff, false-alarm cost). Designing that "operating point" is the subject of levels 10–11.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Trains and evaluates logistic regression four ways on the 5,000 transactions of `fraud_table` (about 1.5% fraud).

- [1] Default settings: sees that behind the dazzling 99%-range accuracy, about 30% of frauds are being missed.
- [2] class_weight="balanced": watches recall jump while precision falls.
- [3] Manual oversampling (duplicating the minority class in the training data only): confirms an effect similar to the weights.
- [4] Threshold tuning: sweeps thresholds 0.9→0.1 on the same model, printing a recall/precision/alert-count table.
- [5] Summarizes the PR curve in text (PR-AUC), then computes which threshold is reasonable under the constraint "the investigation team can handle only 15 alerts."

Things to observe: [2] and [3] give similar results (both are variants of "rewriting the penalty table"), and in the [4] table recall and precision move in opposite directions. Take away the sense that choosing a row in that table is a management question, not a technical one.

## 5. Try it yourself

1. **(Easy)** Add 0.05 to the threshold list in [4]. How high does recall go, and how many alerts does it balloon to? Evaluate it from the standpoint of investigation-team labor costs.
2. **(Medium)** Vary the oversampling multiple (duplicate the minority class 2×/10×/70×). Observe where the benefit plateaus, and explain it from the view that "copies add no information."
3. **(Challenge)** Deliberately write the wrong code that oversamples **before** the split and see how much performance inflates. Hint: duplicate the fraud rows in the full df, then train_test_split — copies leak into the test set and recall/precision rise unrealistically.

## 6. Common mistakes

- **Reporting results as accuracy**: on imbalanced problems, accuracy is essentially a forbidden word. Speak in recall/precision/PR-AUC.
- **Resampling before the split**: the most common and least visible leakage. In pipeline terms, resampling must apply "at training time only."
- **Tuning the threshold on the test data**: the threshold is a hyperparameter too. Set it on validation data and keep the test set sealed.
- **Accepting a demand for 100% recall at face value**: 100% recall is also achieved by "block every transaction." Always set the target together with a precision (or alert-count) condition.
- **The reflex "imbalanced = must resample"**: threshold tuning alone is often enough. Try the simple methods first.

## Next level preview

All the parts are ready. From level08 it's the three-part case-study series. First up: a full pipeline that **forecasts next week's revenue** from a year of `sales_table` — day-of-week, seasonality, and moving-average features plus time-based splitting are the heart of it.
