# Lecture 07 · Level 10 — Case Study 3 — Fraud Detection

> Pit supervised learning, which learns from answer labels, against IsolationForest, which hunts "strangeness" without labels, on the same data — then design an operating threshold matched to the investigation team's alert budget.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level07 / **Estimated time** 55 min

## 1. Why this matters — the business view

Card issuers, banks, commerce, insurance — wherever money flows there is fraud, and a detection system to filter it. Designing that system raises two worries no other problem has.

First, **the limits of labels**. Fraud labels contain only "fraud that got caught." A brand-new scheme nobody has seen yet isn't in the labels, so a supervised model trained on them can be weak against novelty. That's why production detection systems often **run in parallel**: a model that learns from labels, plus an unsupervised model that hunts "different from usual" without them. Second, **the operating budget**. Alerts are investigated by people. If the investigation team can handle 20 cases a day, then no matter how brilliant the model, the alerts must be designed to land near 20 a day. The threshold comes from workforce planning, not statistics.

This level is where you live through both worries in code.

## 2. An analogy to hold onto

Say a department store's security team has two staff members.

- **The veteran (supervised learning)**: has memorized the entire photo album of past shoplifters. Catches every trick in the album like a hawk — but can miss a new trick that isn't in it.
- **The rookie (unsupervised, anomaly detection)**: has no album, but has watched "how ordinary shoppers behave" for a long time. Anyone who deviates sharply from the usual pattern — at odd hours, on a strange route, with a large item — gets attention first. Can catch new tricks too, but also flags innocent shoppers who are merely unusual (more false alarms).

Posting both staff members together is the real-world answer. And "how sensitively to react" is tuned to the number of CCTV alerts the security office can check per day (the budget). Setting that sensitivity (the threshold) is the security chief's job, not the two staffers'.

## 3. Core concepts

### 3.1 IsolationForest — the outlier is the point that isolates easily

IsolationForest repeatedly splits the data on random criteria and measures **how few cuts it takes to isolate each point**. Normal data has many similar neighbors and takes many cuts to isolate; an outlier sits in a remote spot and isolates in just a few. The faster it isolates, the higher the anomaly score. It's an **unsupervised** method that uses no answer labels at all, which makes it precious when labels don't exist (a new service) or can't be trusted (novel schemes). The `contamination` parameter is the assumption "about this fraction of the data is anomalous" — a knob for the alert rate.

### 3.2 Supervised vs unsupervised — how to compare fairly

Both approaches ultimately produce a **suspicion score** per transaction. The fair comparison is "on the same test data, allowed the same number of alerts, how many frauds does each catch?" (precision@K, recall@K). The typical result: with plentiful labels and stable schemes, supervised wins big; with weak labels or shifting schemes, the gap narrows and unsupervised earns its keep as insurance.

### 3.3 Alert-budget-driven threshold design

The real-world procedure for setting the operating threshold starts from the budget, not the performance curve.

1. Confirm investigation capacity: K cases per day
2. Estimate daily transaction volume N → target alert rate = K/N
3. Set the threshold at the top K/N quantile of the validation data's suspicion-score distribution
4. Report the expected recall and precision at that threshold, and get agreement

The operating language is not "threshold 0.83" but "20 alerts a day, about 60% of them genuine, covering about 70% of all fraud." Report alongside it how much recall rises if the budget grows (the marginal benefit), and you have the evidence for an investigation-headcount request.

### 3.4 Parallel operation in practice

A typical production setup: ① high supervised score → immediate alert (the main line), ② low supervised score but high unsupervised score → a separate "suspected novel scheme" queue, sampled and investigated in small volume, ③ investigation outcomes become new labels that retrain the supervised model — a virtuous cycle. main.py [5] demonstrates the idea behind ②.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Compares the two approaches on the 5,000 transactions of `fraud_table` (about 1.5% fraud) and does the operating design.

- [1] Splits the data. Features reuse level03's derived features (log_amount, is_night, etc.) under the same assumption as level07 (no history-aggregation system yet).
- [2] Supervised: builds suspicion scores with logistic regression (class_weight applied).
- [3] Unsupervised: trains IsolationForest **without labels** to build anomaly scores.
- [4] Under the same alert budget (20 out of the 1,500 test transactions), compares the two approaches' frauds caught, precision, and recall.
- [5] Cross-tabulates the two scores: inspects transactions in the "suspected novel scheme zone" — rated low by the supervised model but high by the unsupervised one.
- [6] Sweeps the alert budget from 10 to 60, printing how recall changes (the marginal benefit), and drafts a budget-increase reporting sentence.

Things to observe: in [4] the label-using supervised model leads (labels are powerful); nevertheless in [5] the unsupervised model contributes a different viewpoint; and in [6] each budget increase buys less recall than the last (diminishing marginal returns).

## 5. Try it yourself

1. **(Easy)** Add 100 to the budget list in [6]. Can you see where the recall gains nearly stop? Evaluate the cost-effectiveness of "doubling the investigation team."
2. **(Medium)** Change IsolationForest's `contamination` to 0.005, 0.03, and 0.1. Think about why the comparison in [4] barely changes (it's rank-based, so the threshold assumption matters little).
3. **(Challenge)** Simulate a "novel scheme": remove all `is_foreign=1` fraud from the training data (a world that has never seen that trick), retrain both models, and compare who catches the test set's foreign fraud better. See whether unsupervised's insurance value shows.

## 6. Common mistakes

- **Evaluating the unsupervised model only with accuracy-style metrics**: even though it learned without labels, you can evaluate it with them (historical data has labels). But remember the labels themselves are incomplete — an alert that looked like a "false positive" may in fact be undetected fraud.
- **Mistaking contamination for a performance-tuning knob**: it's only an alert-rate assumption. The ranking (scores) barely moves, so design the operating alert count separately via the budget quantile.
- **Setting the threshold once and forgetting it**: transaction patterns and fraud schemes keep shifting. A weekly loop monitoring alert rate and precision and re-tuning the threshold is mandatory.
- **Concluding supervised-vs-unsupervised as winner-take-all**: they have different purposes. Supervised: "known schemes at maximum efficiency"; unsupervised: "early warning for unknown schemes." Write them up as a division of labor.
- **Confusing blocking with alerting**: instantly blocking a transaction on a high score turns one false positive into a lost customer. Design tiered responses (block / step-up authentication / post-hoc investigation).

## Next level preview

The final level. In level11 we feed the model's probabilities into expected-value calculations to find the profit-maximizing threshold for "who gets the coupon," and close out the whole lecture with a campaign-ROI simulation.
