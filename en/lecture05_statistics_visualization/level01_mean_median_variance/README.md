# Lecture 05 · Level 01 — Mean, Median, and Variance

> A summary statistic is your data's spokesperson. Pick a different spokesperson and you get a different story.
**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 40 min

## 1. Why learn this — the business view

"Our team's average salary is KRW 68 million." You join on the strength of
that sentence — and then discover the team has nine people earning around
KRW 40 million and one manager earning KRW 300 million. Nobody lied. The
average was computed correctly. But because the **wrong summary statistic**
was chosen, the listener pictured something completely different.

At work you choose summary statistics every day: when reporting spend per
customer, when publishing delivery lead times, when comparing store
performance. Whether you use the **mean** or the **median**, and whether you
also report the spread — the **variance** and the **standard deviation** —
changes the decisions people make. By the end of this level you will be able
to answer for yourself: "which summary do I use, and when?"

## 2. Understand it with an analogy

A summary statistic is the **spokesperson you send into a meeting**.
Thousands of data points cannot all fit in the conference room, so you elect
one representative.

- The **mean** is the spokesperson who "blends everyone's opinion a little".
  Every value gets a say — but one loud extreme value can yank the
  spokesperson around badly.
- The **median** is the spokesperson who "stands exactly in the middle of the
  line". However loudly the extremes at either end shout, their position
  barely moves.
- The **standard deviation** is not a spokesperson but a **footnote**: "note,
  however, that opinions differ by about this much." Without that footnote,
  people mistake the spokesperson's statement for unanimous opinion.

For data with a long tail toward large values — salaries, house prices,
revenue — the mean gets pulled toward a few very large values. That is why
national statistics offices publish median household income alongside the
mean.

## 3. Core concepts

### 3.1 The mean — a center of gravity

The mean adds every value and divides by the count:
x̄ = (x₁ + x₂ + … + xₙ) / n. Physically, it is the **center of gravity** —
the point where the data would balance on a seesaw. Precisely because it is a
center of gravity, one value placed very far away (an outlier) acts like a
lever and pulls the mean toward itself.

### 3.2 The median — the middle of the line

Sort the values from smallest to largest; the median is the value exactly in
the middle (with an even count, the average of the two middle values). The
**size** of an outlier does not affect the median: whether the largest value
is 300 million or 3 billion, it is still just "the largest one". This
property is called being **robust**.

### 3.3 Variance and standard deviation — the size of the spread

Take each value's distance from the mean (its deviation), square it, and
average: that is the variance. Squaring makes the unit strange (KRW², say),
so we take the square root to return to the original unit: that is the
standard deviation.

| Concept | Computation | Unit | Intuition |
|---|---|---|---|
| Variance | mean of squared deviations | KRW² | mathematically convenient |
| Standard deviation | square root of variance | KRW | "values typically stray this far from the mean" |

A standard deviation of 0 means every value is identical. A standard
deviation of the same order as the mean is a warning that "the mean is barely
doing its job as a representative".

### 3.4 Which one, when

- Values clustered symmetrically → the mean (with the standard deviation alongside)
- Long tail / outliers present (salaries, house prices, revenue) → median first, mean as reference
- In a report, write **both** whenever possible — and if they differ a lot,
  add one line explaining why (outliers, skew). That is the professional habit.

### 3.5 Bonus — the coefficient of variation

Store A averages KRW 10 million in revenue with a standard deviation of
KRW 1 million; store B averages KRW 1 million with a standard deviation of
KRW 0.5 million. Which store is "choppier"? In absolute terms A (1 million)
is bigger, but relative to its mean, B (50%) is far more unstable. The
standard deviation divided by the mean is called the **coefficient of
variation (CV)**, and it is used to compare volatility across things of
different scale. Exercise [3] computes it per store.

## 4. Practice — main.py

```bash
python3 main.py
```

This exercise uses the cafe-chain revenue from the shared dataset
`hjh_data.sales_table()`. The flow:

- [1] Load the sales data and filter out missing values (None) and negative
  contamination. (A quick review of the cleaning you learned in lecture03.)
- [2] Compute the mean, median, variance, and standard deviation of all
  revenue with pandas, and cross-check against values computed by hand with
  numpy.
- [3] Compare mean and median side by side for each store.
- [4] **The outlier experiment**: artificially add "one giant group order
  (KRW 500 million)" to a 30-row sample of Downtown's revenue, and print step
  by step how much that single value moves the mean, the median, and the
  standard deviation.

The heart is [4]. Watch how, in a 30-row sample, one extreme value drags the
mean up by millions of KRW while the median barely moves. Also notice the
standard deviation exploding — because it squares deviations, it too is
extremely sensitive to outliers.

## 5. Try it yourself

1. **(Easy)** In experiment [4], change the outlier from KRW 500 million to
   KRW 5 billion and rerun. Predict which of the mean and the median will
   change, and by how much, before you check. (Hint: the median does not care
   *how large* the single largest value is.)
2. **(Medium)** Compute the mean and the median per category (Coffee,
   Dessert, and so on). Which category has the biggest gap between mean and
   median, and why?
   (Hint: `df.groupby("category")["revenue"].agg(["mean", "median"])`)
3. **(Challenge)** Implement the **trimmed mean** — the mean after cutting
   off the top and bottom 5% — with numpy and add it to experiment [4].
   Observe where its sensitivity to outliers sits between the mean and the
   median. (Hint: `np.sort`, then slice off both ends.)

## 6. Common mistakes

- **Not asking why the mean and median differ**: the gap is not an error — it
  is information that the data is skewed. The gap itself is worth reporting.
- **Reporting the mean without the standard deviation**: "5-day average
  delivery" with a 3-day standard deviation means customers waiting more than
  8 days are common. Complaints come from the tail, not the mean.
- **Forgetting the variance's unit**: a variance of 25 million is in KRW².
  Write the square root — the standard deviation (KRW 5,000) — in reports to
  avoid confusion.
- **Computing statistics before cleaning**: filling missing values with 0
  distorts the mean. None and 0 are not the same. Decide your
  missing-value/outlier policy before you compute.

## Next level preview

Sometimes two or three summary statistics are still not enough. In the next
level we draw the data's entire shape — its distribution — as a histogram,
and experiment with how the same data can look different when you change the
bin size.
