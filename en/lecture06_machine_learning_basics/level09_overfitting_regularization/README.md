# Lecture 06 · Level 09 — Overfitting, Regularization, and Hyperparameters

> Raise the polynomial degree and witness on a chart the moment the train/validation errors split apart, then learn to tame complex models with the brakes called regularization (L1/L2).
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level04 / **Estimated time** 50 min

## 1. Why Learn This — The Business View

Overfitting has already appeared three times in this lecture. Level04's optimism bias, level07's unlimited-depth tree, level08's instability — all different symptoms of the same disease. In this level we treat the disease head-on.

Why this matters at work: it's **the language of model selection**. "Should we raise or lower the degree?", "What should alpha be?", "Why has the validation score stopped improving?" — most of a data team's daily conversation happens somewhere on the axis between overfitting and underfitting. Understand this axis and you can answer "does making the model more complex help?" with "the training score always improves; the real-world score improves only up to a point." Finding that point is the procedure called hyperparameter tuning, and the brake that pulls you back after you've passed it is regularization.

## 2. Understanding by Analogy

Think of **tailoring a suit**. The customer's body (the data) mixes an essential build (the pattern) with that day's accidents — thick undergarments, the lunch they just ate (noise).

- **Underfitting**: a one-size-fits-all plastic rain poncho. Roughly fits everyone, fits no one well. The model is too simple to capture even the build — like trying to draw a curved pattern with a first-degree straight line.
- **A good fit**: a well-tailored suit that follows the build while ignoring the day's accidents. It still fits tomorrow.
- **Overfitting**: a garment shrink-wrapped to the body of that exact moment, lunch bulge and all. **Perfect today, unwearable tomorrow.** A 15th-degree curve threading through every training point is exactly this state.

**Regularization is the instruction you give the tailor**: "follow the body, but don't overreact to every bump." Mathematically, it fines the model for letting its coefficients grow, keeping the curve gentle even at the cost of fitting the data slightly less. Adjusting the fine's severity (alpha) is hyperparameter tuning.

## 3. Core Concepts

### 3.1 Parameters vs Hyperparameters

- **Parameters**: values that learning extracts from the data (regression coefficients, a tree's split points). The machine's job.
- **Hyperparameters**: knobs a human sets before learning starts (polynomial degree, tree depth, regularization strength alpha, forest tree count). The human's job — chosen not by gut but by **validation-set score** (level04's practice exam was exactly for this).

### 3.2 The Validation Curve — Two Lines That Split Apart

Put complexity (e.g., polynomial degree) on the x-axis and error on the y-axis, and plot training error and validation error together: that's the validation curve. Its typical shape:

- Training error: **monotonically decreasing** as complexity grows (more capacity to memorize)
- Validation error: falls (capturing the pattern better), then from some point **rises again** (starting to memorize the noise)

Where the two lines start to diverge is where overfitting begins, and the validation error's minimum is the point to choose. Today's PNG is exactly this picture.

### 3.3 L2 and L1 — Two Kinds of Fine

Regularization adds a "coefficient-size fine" to the loss function.

- **L2 (Ridge)**: fine = α × (sum of squared coefficients). Shrinks all coefficients **uniformly**. Never drives them to zero. The default choice.
- **L1 (Lasso)**: fine = α × (sum of absolute coefficients). Drives less-important coefficients **exactly to zero**, automatically selecting features. Use it when you need to "sift the real signals out of hundreds of features."

At α = 0 there is no regularization (ordinary regression); too large and every coefficient is crushed into underfitting. α is itself a hyperparameter, chosen by validation. One practical rule: **when using regularization, feature standardization is effectively mandatory** — features with larger units get smaller coefficients and unfairly dodge the fine.

### 3.4 Other Prescriptions for Overfitting

Regularization is just one prescription. More data (the surest one), fewer features, a simpler model (lower degree or depth), ensembles (level08), early stopping — the shared philosophy is one sentence: **grant the model only as much freedom as the data can support.**

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

On synthetic curve data (true pattern + noise, 30 training points) we walk the entire arc of overfitting.

- **[1]** Generate data whose true pattern is a curve and split into train/validation. We deliberately use only 30 training points so overfitting shows clearly.
- **[2] Validation-curve experiment**: sweep polynomial degrees 1–15 and print train/validation RMSE in a table. Watch the training error keep falling while the validation error bends into a U (bottoming out around degree 3) — and the coefficients at high degree explode into the thousands and tens of thousands.
- **[3]** Save three fitted curves (underfit degree 1, good degree 3, overfit degree 15) plus the validation curve to `outputs/overfitting.png`. This one picture is the whole level.
- **[4] Ridge in action**: on the runaway degree-15 model, raise α through 0→0.001→0.1→10 and watch in a table how the validation RMSE and coefficient sizes calm down. Confirm "complex model + the right fine ≈ a well-sized model."
- **[5] A taste of L1 (Lasso)**: count how many of the 15 polynomial features get coefficients of exactly 0, confirming automatic feature selection.

The heart of the code: `PolynomialFeatures(degree)` builds the x, x², … features, and a `Pipeline` standardizes before feeding `LinearRegression`/`Ridge`/`Lasso`. Note where in the code the two hyperparameters — degree and α — enter as "human decisions."

## 5. Try It Yourself

1. **(Easy)** Increase the training sample in [1] from 30 to 300 points. How much does the degree-15 model's overfitting ease? This experiment confirms "more data is the surest prescription."
2. **(Medium)** In [4], set α very large (e.g., 10000). The validation RMSE will worsen again. Look at the train/validation error pattern and diagnose: is this overfitting or underfitting?
3. **(Challenge)** Use `sklearn.model_selection.validation_curve` to draw a 5-fold cross-validation validation curve over 10 candidate α values and add code to pick the best α. (Hint: `param_name="ridge__alpha"`.)

## 6. Common Mistakes

- **Mistaking a falling training error for progress**: raise the complexity and training error falls **unconditionally**. It carries zero information. Only the validation error answers.
- **Pushing "more complex" after the validation error has started rising**: every investment past the divergence point (degree, parameters, training time) goes into memorizing noise.
- **Applying regularization without standardization**: features with large units dodge the fine. Bundle standardization into a Pipeline.
- **Choosing α with the test set**: level04's exam-paper peeking applies to regularization just the same. Choose α with validation (or cross-validation).
- **Misdiagnosing underfitting as overfitting**: if even the training error is large, tightening regularization makes things worse. Diagnose by looking at **both** train and validation errors.

## Next Level Preview

Everything so far was supervised learning — "data with answer labels." In level10 we cross over to unsupervised learning, finding structure without answers: splitting customers into segments (k-means) and compressing dozens of variables into a 2-D map you can see with your eyes (PCA).
