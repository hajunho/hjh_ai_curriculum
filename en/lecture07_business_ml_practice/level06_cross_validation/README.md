# Lecture 07 · Level 06 — Cross-Validation — Performance You Can Trust

> See by experiment how much of a single-split score is pure luck, learn to produce mean ± standard deviation performance reports with 5-fold cross-validation, and learn the special splitting rules for time-series data.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level05, lecture06 level04 / **Estimated time** 35 min

## 1. Why this matters — the business view

A vendor reports "AUC 0.87 achieved." How much should you trust that number? If the same model can score 0.83 or 0.90 depending on how the data happened to be cut, then competing over the second decimal place is meaningless. And it really does swing that much — the smaller the sample and the more severe the imbalance, the bigger the wobble.

Cross-validation (CV) measures that wobble head-on. The moment the report changes from "0.87" to **"0.86 ± 0.02,"** you can judge whether the difference between two models is real or luck. For a decision-maker, this is the information that decides "should we spend more budget on this improvement?" And for a reviewer, one single question — "how many times did you measure that number?" — gauges the credibility of the whole report.

## 2. An analogy to hold onto

Think of a taste test for a new product. What if you ask **one group** of customers and report "90% satisfaction!"? If that group happened to have a sweet tooth, the number is completely distorted. The common-sense fix is to **ask several groups in turn and report the mean together with the spread**.

5-fold cross-validation is exactly this. Cut the data into 5 pieces; each round, study (train) on 4 pieces and sit the exam (validate) on the remaining one. Rotate the pieces, take 5 exams, get 5 scores: the **mean** is the estimate of skill, the **standard deviation** is the size of the luck. Every data point serves as an exam question exactly once, so no single piece's quirks can dominate.

One extra rule applies to time series, though. You study from past mock exams and sit the future final — you must never **study from the final and then sit the mock**. Shuffle randomly and you manufacture exams where the future predicts the past.

## 3. Core concepts

### 3.1 Single-split variance — the score is a random variable

Change only the n in `train_test_split(random_state=n)` and the score changes. A score is not a fixed truth but **a random variable contaminated by the accident of the split**. In main.py [1] we evaluate the same model on 30 different splits and look directly at the score's distribution (its min-to-max spread). That spread is the hidden error bar behind every "single score" you've ever been shown.

### 3.2 k-fold cross-validation

- **k=5 or 10** is the convention. With very little data, raise k (the extreme is LOO); with lots, 5 is plenty.
- For classification use **StratifiedKFold**: it keeps each piece's class ratio (churn rate 18%) equal to the original. Especially important for imbalanced data.
- Reporting format: `mean ± standard deviation`. When comparing two models, if the difference in means is smaller than the standard deviation, "no difference" is the honest conclusion.
- Cost: training runs k times, so k× the time. That's the price of trust.

### 3.3 The final model and the final test

Cross-validation measures "how good is this method (model + preprocessing + hyperparameters) on average." Once the method is decided, **retrain the final model on all the training data**. And if you also used CV to choose hyperparameters, the score for the final presentation should be measured on a **sealed test set** never touched by CV (measure with the same ruler you used to choose, and the score comes out generous).

### 3.4 Time-series splitting — TimeSeriesSplit

Use random k-fold on time-ordered data (sales, stock prices, traffic) and you create "predicting the past from the future" leakage. Especially with lag features like moving averages or yesterday's revenue, the validation piece's information sits directly inside the training piece's features, inflating the score badly. The fix is `TimeSeriesSplit`: cut so the training window always comes before the validation window (train on Jan–Mar → validate April; train Jan–Apr → validate May; ...). In main.py [4] we compare how inflated the same sales-forecasting problem looks under random splitting.

## 4. Hands-on — main.py

```bash
python3 main.py
```

- [1] The luck experiment: evaluates `churn_table` + logistic regression on 30 random splits, printing the AUC's min/max/standard deviation with a mini histogram.
- [2] Computes `mean ± standard deviation` with 5-fold StratifiedKFold cross-validation. Also prints the per-fold scores.
- [3] Fairly compares two models (logistic regression vs random forest) under CV, and judges "is the difference meaningful?" by weighing the mean gap against the standard deviation.
- [4] The time-series trap: adds a previous-day revenue feature to `sales_table` daily revenue, then compares R² under random KFold vs TimeSeriesSplit.

The key line of code is `cross_val_score(pipe, X, y, cv=StratifiedKFold(...), scoring="roc_auc")`. Note that we feed in the entire pipeline from the previous level (preprocessing is re-fit per fold).

## 5. Try it yourself

1. **(Easy)** Shrink the sample in [1] from `n=2000` to `n=400`. Watch the score's wobble (standard deviation) grow, and write down why "the smaller the data, the more CV is mandatory."
2. **(Medium)** Change k in [2] to 3, 5, and 10 and observe how the mean and standard deviation move. As k grows, training data increases but validation pieces shrink, so fold-score spread can widen.
3. **(Challenge)** Add `KNeighborsClassifier` (with StandardScaler in its pipeline) as a third contender in [3] and make it a three-way fight. Judge whether any pair satisfies "mean difference < standard deviation."

## 6. Common mistakes

- **Choosing a model from a single measurement**: a 0.87-vs-0.86 contest flips on split luck. Compare with CV, always.
- **Reporting the CV score as "the test score"**: if hyperparameters were chosen by CV, the CV score is already slightly generous. Final reports use the sealed test set.
- **shuffle=True on time series**: randomly shuffling data with moving-average/lag features quietly inflates the score. If there's a time axis, evaluate TimeSeriesSplit first, no exceptions.
- **Plain KFold on imbalanced data**: some folds may contain almost no churners. For classification, StratifiedKFold is the default.
- **Letting preprocessing leak through CV**: scale/encode outside CV and the folds peek at each other. Feed the pipeline in whole (level05).

## Next level preview

You can now measure performance you can trust. In level07 you meet the toughest opponent — imbalanced data with only 1.5% positives (fraud detection) — and learn to cook the recall-precision trade-off with class_weight, resampling, and threshold tuning.
