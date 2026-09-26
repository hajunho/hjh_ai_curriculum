# Lecture 07 · Level 05 — Scaling and Pipelines

> Learn scaling, which puts features with wildly different units on the same ruler, and the sklearn Pipeline, which bundles preprocessing and model into one unit and blocks leakage structurally.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level04 / **Estimated time** 35 min

## 1. Why this matters — the business view

A large share of data accidents come not from fancy algorithms but from **preprocessing-order mistakes**. Two classic cases: ① computing the mean and standard deviation over data that includes the test set, then scaling (leakage); ② forgetting in the deployment code a preprocessing step done at training time (train–serving skew). Both produce the hardest-to-diagnose type of failure: "it looked great in validation but acts strange in production."

sklearn's Pipeline is a tool that **shuts down both accidents at the level of code structure**. Bundle preprocessing and model into one object, and fit happens only on training data while transform always applies in the same order. When practitioners exchange "the model file," what actually travels is this pipeline object. As a reviewer, the single question "is the preprocessing inside a pipeline?" tells you a lot about a project's quality.

## 2. An analogy to hold onto

**Scaling = unifying the units.** In one meeting, Team A says "revenue: 3 (hundred million KRW)" and Team B says "revenue: 250,000,000 (KRW)" — anyone who looks only at the digits thinks Team B's item is 80 million times bigger. Distance-based models (KNN and friends) make exactly this mistake. Measure monthly fee (9,900–29,900) and usage days (0–30) with the same ruler, and a 1-KRW difference in fee counts the same as a 1-day difference in usage — the fee monopolizes the distance. Standardization converts every variable into the shared unit "mean 0, standard deviation 1."

**A pipeline = a factory conveyor belt.** Once a part (the data) is on the belt, washing (scaling) → assembly (the model) always happens in the same order. Without the belt, a worker forgets the wash, or dips the outgoing-inspection parts (test data) into the wash water (mean/std computation). With the belt, an ordering mistake becomes impossible.

## 3. Core concepts

### 3.1 StandardScaler — standardization

For each column, compute `(value − mean) / standard deviation`. The result is a unit-free score: "how many standard deviations from the mean." The core rule: **compute (fit) the mean and standard deviation on training data only, and merely apply (transform) those values to the test set.** Average over the test set too, and information about the test distribution leaks into training.

### 3.2 Models that need scaling, models that don't care

- **Sensitive**: KNN, SVM, logistic/linear regression (with regularization), neural networks, K-means — models that use distances or magnitudes directly
- **Insensitive**: decision trees, random forests, boosting — they only split on "bigger than X?", so units are irrelevant

If you'll only ever use tree models you could skip scaling — but keep it in the pipeline and switching models stays safe.

### 3.3 How a Pipeline blocks leakage

Bundle with `make_pipeline(StandardScaler(), KNeighborsClassifier())` and:

- `pipe.fit(X_train, y_train)` → the scaler learns mean/std **from the training data only**, and the model trains on the transformed data
- `pipe.predict(X_test)` → the test set is transformed with the training-time mean/std, then predicted

Put the whole pipeline into cross-validation (`cross_val_score(pipe, X, y)`) and the scaler is re-fit **per fold** on that fold's training portion only. Scale the full dataset up front and then cross-validate, and every fold's validation information is already inside the scaler — the textbook case of preprocessing leakage. For plain scaling the leakage effect is small, but for preprocessing that uses the answer or the distribution heavily — target encoding, missing-value imputation, feature selection — scores can inflate badly. So the rule is unified into one sentence: **all preprocessing goes inside the pipeline.**

### 3.4 The deployment bonus

Save the single pipeline object and preprocessing + model travel together. Deployment code becomes the one line `pipe.predict(new_data)`, and the "forgotten preprocessing" flavor of train–serving skew disappears.

### 3.5 How far a pipeline grows

Today's belt has two stations — scaler + model — but real-world pipelines grow to five or more: imputation → encoding (level04) → scaling → feature selection → model. With sklearn's `ColumnTransformer` you can put different treatments for different columns on one belt — "scale the numeric columns, one-hot the text columns." The structure is exactly what you learned today; look up the names when you need them.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Predicts churn with KNN on `churn_table` while experimenting with scaling and pipelines.

- [1] Feature units check: prints the scale gap between monthly_fee (tens of thousands of KRW) and usage_days_30d (0–30).
- [2] KNN without scaling → the fee monopolizes the distance; even picking the top-100 risk list is barely better than random sampling.
- [3] StandardScaler + KNN (pipeline) → same model, same data; watch AUC and "actual churners among the top 100" jump together.
- [4] Wrong-order demo: scaling the full dataset first, then cross-validating vs putting the pipeline into cross-validation — compare the scores and see that the point is "structural safety" more than the numeric gap.
- [5] Predicts churn probability for 3 new customers with the single pipeline object, demonstrating the deployment convenience.

The key thing to see in the code: the object built by `make_pipeline(...)` is used in `fit / predict / cross_val_score` exactly like an ordinary model. One API to learn, three safety guarantees gained (order guaranteed, fit scope guaranteed, deployment as one unit).

## 5. Try it yourself

1. **(Easy)** Rerun [3] with KNN's `n_neighbors` set to 3, 15, and 51. Check that scaling's benefit holds regardless of the neighbor count.
2. **(Medium)** Swap the model in [2]/[3] for `DecisionTreeClassifier(max_depth=5, random_state=0)`. Confirm the with/without-scaling gap almost disappears, and explain why using 3.2.
3. **(Challenge)** Switch to `MinMaxScaler` (squeezes into the 0–1 range) and compare against StandardScaler. Think about what kind of data would widen the gap between the two (hint: columns with big outliers).

## 6. Common mistakes

- **Calling fit_transform on the test data**: the test set gets `transform` only. With a pipeline, this mistake becomes impossible by construction.
- **Scaling ahead of time, outside cross-validation**: out-of-fold information leaks in. Feed `cross_val_score` the whole pipeline.
- **Ignoring scaling entirely because "we use trees"**: true today — until someone swaps in logistic regression and performance quietly collapses. Keep it in the pipeline and model swaps stay safe.
- **Forgetting what the values mean after scaling**: don't quote standardized coefficients or values (like −1.2) in a report as if they were KRW. When interpreting, convert back to original units or state explicitly that they're in "standard-deviation units."

## Next level preview

Preprocessing is now safe — but can you trust the performance number itself? In level06 we show by experiment how much a single-split score is down to luck, and learn to produce mean ± standard deviation performance reports with cross-validation.
