# Lecture 06 · Level 04 — Train / Validation / Test Splits

> "Anyone can ace a test made of the questions they studied" — obvious, yet in ML this becomes a major incident. We prove why, by experiment.
**Difficulty** ⭐⭐ / **Prerequisites** level03 / **Estimated time** 35 min

## 1. Why Learn This — The Business View

When someone reports "our model is 97% accurate," there is exactly one first question to ask: **"That 97% — measured on which data?"** If it was measured on the training data, the number is almost certainly inflated, and performance will crater the moment the model hits production. This trap competes for the number-one cause of real-world ML incidents, and it happens without any bad intent — anyone who doesn't know the separation principle will commit it naturally.

Splitting data is a one-line-of-code technique, but people who know it and people who don't read ML results in fundamentally different ways. Vetting a vendor's work, verifying an internal report, checking your own experiments — this level is the seatbelt for all of them.

## 2. Understanding by Analogy

Picture **a student who memorized last year's exam wholesale**. A student who memorized 100 past questions, answers and all, scores 100 on a mock exam built from those 100 questions. But that 100 proves memorization, not ability. To measure ability, you must test with **new questions the student has never seen**.

Models are the same. The training data is "the past questions they studied," and the training score is always more generous than true ability. So you set aside part of the data in a vault from the very start (the test set) and take it out exactly once, at the end, to measure real skill.

But there's one more trap. Building a model involves a stage of "trying different settings and picking the best one." If you keep grading against the test set while picking, **you're sneaking peeks at the exam paper to steer your studying**. You never memorized it outright, but you're steadily contaminated in the exam's favor. So you keep a third pile just for choosing: the validation set. In short — **the training set is the textbook, the validation set is the practice exam, the test set is the final board exam.** You sit the board exam once in your life.

## 3. Core Concepts

### 3.1 The Roles in a Three-Way Split

| Set | Analogy | Purpose | Times used |
|---|---|---|---|
| Train | Textbook | Learn the model's parameters | Unlimited |
| Validation | Practice exam | Choose settings (hyperparameters), compare models | Many times |
| Test | Board exam | Report final ability | **Once, at the end** |

Customary ratios are 60:20:20 or 70:15:15. With little data, reinforce validation using cross-validation — rotating the validation set and measuring several times.

### 3.2 The Optimism Bias Comes from "Memorizing Coincidences"

Training data mixes real patterns with coincidental noise. The more flexible the model, the more it memorizes the noise too — and noise is a coincidence unique to that data, useless on new data. So **the more flexible the model (e.g., a deep tree), the wider the gap between training score and real-world score.** In today's experiment [2] you'll see that gap in numbers. This phenomenon is called overfitting, treated in depth in level09.

### 3.3 Test-Peeking Can Be Proven by Experiment, Too

Experiment [3] is subtler. Compare choosing among several candidate settings **by their test-set scores** versus **choosing by validation and touching the test set only once at the end**: the former's final reported score is systematically inflated. Because among many candidates, you end up picking "the one that happened to do well on that particular exam paper." The more candidates, the bigger the inflation — the same principle as coin flipping: have 100 people flip coins and someone will get 10 heads in a row.

### 3.4 Rules for Splitting

- **Split before everything else**: doing any data-"looking" work — normalization, feature selection — on the full dataset before splitting is itself leakage. Even the scaler must be `fit` on the training set only.
- **Randomly, with a fixed seed**: cutting the first 70% of sorted data is biased (e.g., data sorted by sign-up date). Exception: time-series prediction must be cut **in time order** — training on the future to predict the past is cheating.
- **Don't let one customer's rows land on both sides**: if a customer's data appears in both train and test, it's no longer "a question never seen."

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

Using sklearn's synthetic classification data (`make_classification`), we prove both traps by experiment.

- **[1]** Generate 1,000 rows and split 60:20:20 (two calls to `train_test_split`).
- **[2] Experiment A — the optimism bias of self-grading**: train decision trees from shallow to deep and print training vs test scores side by side. Watch the training accuracy race toward 100% while the test accuracy actually falls — the gap widening.
- **[3] Experiment B — peeking at the exam paper**: with 15 candidate values of k for a k-nearest-neighbors model, compare (a) choosing by test score vs (b) choosing by validation and using the test just once, repeated over 30 different random splits. Output statistics on how much (a)'s reported score is inflated over true ability.
- **[4]** Summarize the lessons of both experiments.

Output worth noticing: in experiment B, the difference between "the score chosen on the test set" and "the score on new data" is consistently positive. The inflation is not chance — it is **structural**.

## 5. Try It Yourself

1. **(Easy)** Change the `test_size` ratios in [1] to make an 80:10:10 split. When the test set shrinks, what happens to the trustworthiness of the test score? (Hint: watch the standard deviation in [3]'s repeated experiment.)
2. **(Medium)** Grow the k candidates in [3] from 15 to 50. Verify 3.3's claim that more candidates means bigger peeking inflation.
3. **(Challenge)** Replace [3]'s validation step with 5-fold cross-validation using `sklearn.model_selection.cross_val_score`. Does the chosen k differ from the simple validation-set approach? (Hint: apply it to the combined train+validation data.)

## 6. Common Mistakes

- **Putting the training score in a report**: "99% training accuracy" may be a warning sign, not an achievement.
- **Grading against the test set repeatedly while choosing a model**: the moment you choose with it, it has become a validation set — and your clean data for the final measurement is gone.
- **Preprocessing on the full dataset before splitting**: scaling, missing-value imputation, feature selection must all be fit on the training set, with only transform applied to the rest.
- **Randomly shuffling a time series before cutting**: that's building a model to predict tomorrow while training on the day after tomorrow.
- **Not fixing the seed, so results can't be reproduced**: change the split and the score changes. A reported number must be reproducible.

## Next Level Preview

Seatbelt buckled — now we build a classification model properly. In level05 we train logistic regression on the churn data to answer in probabilities, like "this customer has a 73% chance of churning," and interpret the coefficients in business language.
