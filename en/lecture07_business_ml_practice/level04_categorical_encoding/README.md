# Lecture 07 · Level 04 — Encoding Categorical Variables

> Three ways to turn text data like "Downtown", "Coffee", "Sat" into numbers a model can eat — compared by experiment, trap by trap.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03 / **Estimated time** 40 min

## 1. Why this matters — the business view

Half of real-world data is not numeric. Store names, product categories, signup channels, job grades, regions — all **categorical** variables. Yet most models only accept numbers. Encoding — converting text to numbers — is therefore a rite of passage for every tabular-data project.

The problem: there are several ways to do it, and **choosing badly injects false information into the model**. Turn store names into 1, 2, 3 and the model learns a ranking that doesn't exist — "store 3 is three times store 1." Choose well, and the single piece of information "which store" can lift sales-forecast accuracy substantially. Knowing which encoding fits which situation is also a review skill: it lets you ask a data team's report, "why was this handled this way?"

## 2. An analogy to hold onto

Imagine three ways of keying in the "occupation" field of a survey.

- **One-hot encoding**: one checkbox column per occupation. "Office worker ☑ / Self-employed ☐ / Student ☐". Zero rank distortion, but 500 occupations means 500 columns.
- **Ordinal encoding**: number the occupations. "Office worker=1, self-employed=2, student=3." One column and done — but the sizes of the numbers mean nothing, and the model will try to find meaning anyway. That said, for categories with a **real order** — like junior < associate < manager < director — it's actually the honest method.
- **Target encoding**: instead of the occupation, write **the group's average of the answer** — e.g., "that occupation group's historical delinquency rate." A powerful way to compress the information into one column, but because it uses the answer as an ingredient, a miscalculation leaks the answer key into the exam paper.

## 3. Core concepts

### 3.1 The three encodings at a glance

| Method | Columns | Rank distortion | Leakage risk | Best fit |
|---|---|---|---|---|
| One-hot | as many as categories | none | none | the default when categories are few (a few dozen or less) |
| Ordinal | 1 | **yes** (when used on unordered categories) | none | truly ordered categories (grades, ranks), tree-based models |
| Target | 1 | none | **yes** | high cardinality (hundreds to tens of thousands of categories) |

Tree-based models (decision trees, random forests) use numbers only as "split points," so they are relatively insensitive to ordinal encoding's distortion. Rank distortion is fatal mainly for linear models, which take the magnitude of a number at face value.

### 3.2 Target encoding's leakage and its defenses

Target-encoded values must be computed **from the training data only**. Compute the group means over data that includes the test set and the test answers seep into the feature. Even within the training data, there's the problem of "each row's own answer feeding its own feature," so in practice people use a cross-validation scheme (compute leaving out your own fold) or smoothing (pull small groups toward the global mean). main.py implements the most basic defense: "compute on training data only + unseen categories get the global mean."

### 3.3 The high-cardinality problem

When a category has thousands to tens of thousands of values — customer IDs, product codes, postal codes — one-hot causes a column explosion (curse of dimensionality + memory). The options then: ① target encoding, ② frequency encoding (replace with the count of appearances), ③ keep the top N and bucket the rest as "other," ④ drop the column entirely (for something like customer ID with no generalizable signal, dropping it is the right answer).

### 3.4 Handling unseen categories

After deployment, categories the training never saw will arrive — like a "newly opened store." Decide the **unseen-category policy** up front — one-hot: all columns 0; target encoding: substitute the global mean — and you avoid production incidents. Applying `pd.get_dummies` separately to train and test misaligns the columns, so the professional pattern is sklearn's `OneHotEncoder(handle_unknown="ignore")` inside a pipeline (level05).

## 4. Hands-on — main.py

```bash
python3 main.py
```

On one year of `sales_table`, trains a linear (Ridge) regression predicting revenue from store, category, and weekday — three times, changing only the encoding.

- [1] Data cleaning: drops rows with missing or negative revenue and checks how many categories each of the 3 categorical columns has.
- [2] One-hot: sees how many columns `OneHotEncoder` produces and measures performance (MAE, R²).
- [3] Ordinal: replaces categories with numbers. Only 3 columns, but watch performance collapse (rank distortion).
- [4] Target: replaces categories with the training data's group-mean revenue. Sees 3 columns get close to one-hot performance.
- [5] Summary table: "columns vs performance" side by side per method, plus the selection criteria for high-cardinality situations.

Key observations: ordinal's R² is clearly below one-hot's (the linear-model + unordered-category mismatch), and target encoding closes most of that gap with just 3 columns.

## 5. Try it yourself

1. **(Easy)** In [3], change the order in which category numbers are assigned (e.g., reverse the `categories` order). Performance changes again — proof that ordinal encoding's result depends on the accident of "what order you numbered things in."
2. **(Medium)** Swap Ridge for `DecisionTreeRegressor(max_depth=8)` and compare the three encodings again. Confirm that ordinal's performance drop shrinks dramatically for trees, and explain why using 3.1.
3. **(Challenge)** Implement "frequency encoding" (replace each category with its training-data appearance count) and enter it as a fourth contender. Hint: `map`ping a `value_counts()` is two lines.

## 6. Common mistakes

- **Casually numbering unordered categories**: the most common accident is feeding a model data that was coded "Seoul=1, Busan=2" in a spreadsheet, as-is. For linear models, always expand with one-hot.
- **Computing target encoding over the full dataset**: if validation performance suddenly improves, suspect leakage before celebrating. Compute from training data, always.
- **Applying get_dummies separately to train and test**: the two sides end up with different columns and it blows up at deployment. `OneHotEncoder(handle_unknown="ignore")` + a pipeline is the standard.
- **Target-encoding customer IDs and seeing performance "improve"**: with only a few rows per customer, the group mean becomes a near-copy of the answer. ID-like columns should be excluded on principle.

## Next level preview

Preprocessing like encoding and scaling must obey the rule "fit on training data only" — and managed by hand, mistakes are guaranteed. In level05 we bundle preprocessing and model into one unit with an sklearn Pipeline and solve this problem structurally.
