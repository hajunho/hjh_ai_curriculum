# Lecture 06 · Level 03 — Linear Regression

> We answer "if we raise ad spend, how much does revenue grow?" with the single straight line that best fits the data — and compute that line two ways, by NumPy formula and with sklearn.
**Difficulty** ⭐⭐ / **Prerequisites** level02, lecture03 (Pandas) / **Estimated time** 40 min

## 1. Why Learn This — The Business View

When the marketing budget meeting asks "if we spend 10 million KRW more on ads, how much more revenue do we get?", there is a world of difference between "it'll probably go up" and "based on historical data, each additional 10,000 KRW of ad spend was associated with about 15,000 KRW more revenue." Linear regression is the oldest, most widely used, and most explainable tool for answering "how much?" with a number.

There's a second reason: linear regression is the grammar underlying this whole lecture. Logistic regression (level05) is linear regression with a probability transform on top; regularization (level09) is the art of taming linear regression's coefficients; even neural networks are stacks of linear regressions. Nail down "coefficient," "intercept," and "minimizing error" here, and everything that follows gets easier.

## 2. Understanding by Analogy

Imagine **laying a ruler through a cloud of dots**. On a scatter plot of ~1,800 daily (ad spend, revenue) points, where would you place a transparent ruler to draw "the one line that represents all the points"? Some points will sit above the ruler, some below. A good line is **the one whose vertical gaps (errors) to the points are smallest overall**.

Least squares makes "smallest overall" precise: choose the line that minimizes the sum of the squared vertical distances from each point to the line. Squaring serves two purposes — it stops above/below errors from canceling out, and it penalizes big errors more heavily. The remarkable part: this optimal line is computed **directly by a one-line formula**, with no trial and error. Unlike level00's "try every threshold" search, linear regression is a case where the math has already finished the search for you.

## 3. Core Concepts

### 3.1 The Shape of the Model: y = a·x + b

- **x**: the input feature (ad spend), **y**: the prediction target (revenue)
- **a (slope, the coefficient)**: how much y rises when x rises by 1. **The heart of the business interpretation.** "Per 10,000 KRW of ad spend, revenue +a×10,000 KRW."
- **b (the intercept)**: y when x = 0. Read it as "baseline revenue with zero advertising," but interpret with care — it's extrapolation beyond the data's range.

### 3.2 The Least-Squares Formula

The slope and intercept have a closed-form calculation from means, variance, and covariance.

```
a = Cov(x, y) / Var(x)      (how much x and y move together ÷ how spread out x is)
b = mean(y) − a · mean(x)   (the line always passes through (x̄, ȳ))
```

In today's exercise you compute these two lines directly with NumPy, and confirm they match sklearn's `LinearRegression` answer down to the decimals. The point is to see with your own eyes that the library is not magic — it's packaging around this formula.

### 3.3 How Well Does It Fit: R²

The coefficient of determination, R² (R-squared), is "the fraction of y's fluctuation that the model explains." 0 means no better than guessing the mean; 1 means perfect. On real-world data, R² between 0.3 and 0.7 is common. A low R² doesn't make a model useless, and a high one doesn't prove causation.

### 3.4 The Interpretation Trap: Correlation ≠ Causation

A regression coefficient only summarizes a **co-movement**: "on days with higher ad spend, revenue was higher too." Maybe the ads drove the revenue — or maybe it was peak season, and both ads and revenue rose together. To assert "raising ad spend raises revenue" you need an experiment (an A/B test) or further analysis. In a report, the safe phrasing is "an **association** of +15,000 KRW revenue per 10,000 KRW ad spend was observed."

### 3.5 Using Several Features: Multiple Regression

You can include several features in the form y = a₁x₁ + a₂x₂ + … + b (multiple linear regression). Each coefficient's interpretation then changes to "the effect, holding the other features fixed," and because features come in different units, comparing coefficient sizes requires standardization. Today we focus on simple regression; multiple regression comes up naturally with the classification models from level05 onward.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

We analyze the ad-spend→revenue relationship in cafe-chain sales data (synthetic; amounts in KRW).

- **[1]** Load `sales_table` with pandas and aggregate to (day, store) granularity. The cleaning step removes missing values and negative contamination (a lecture03 refresher).
- **[2]** Compute the least-squares formula (Cov/Var) directly with NumPy to get slope a and intercept b.
- **[3]** Fit sklearn's `LinearRegression` on the same data and confirm the two results agree.
- **[4]** Translate the coefficients into business language: "per 10,000 KRW of ad spend, revenue +about N," R², and the predicted revenue at 300,000 KRW of ad spend.
- **[5]** Save the scatter plot + regression line to `outputs/regression.png`. Do open the file and look at the scatter of the points and where the line sits.

The heart of the code: the single line `slope = np.cov(x, y)[0, 1] / np.var(x, ddof=1)` is exactly the formula from 3.2. On the sklearn side, `model.coef_[0]` and `model.intercept_` hold the same values. Different tools, same math, same answer.

## 5. Try It Yourself

1. **(Easy)** Change the prediction in [4] to an ad spend of 1,000,000 KRW. That's extrapolation beyond the data's ad-spend range (50,000–400,000 KRW). Should you trust this prediction? (Hint: a line extends forever beyond the range, but reality doesn't.)
2. **(Medium)** Fit a separate regression line per store and compare the slopes. Which store appears to get the best return on advertising? (Hint: `df.groupby("store")`, then apply the same formula per store.)
3. **(Challenge)** Run a multiple regression in sklearn adding a 0/1 weekend feature. Does the ad-spend coefficient change from the simple regression? If so, why? (Hint: if both ad spend and revenue are higher on weekends, the simple regression's ad coefficient was quietly absorbing the weekend effect.)

## 6. Common Mistakes

- **Running regression before cleaning**: with missing values and negative contamination left in, the coefficients get distorted. Step [1] in today's code is long for a reason.
- **Interpreting the intercept at face value**: "revenue of b at zero ad spend" is extrapolation if no day in the data actually had zero ad spend.
- **Discarding on low R² / trusting on high R²**: even with a low R², the coefficient's direction and size can inform decisions; even with a high one, it may be leakage or coincidence.
- **Reporting coefficients as causation**: distinguish "an association was observed" from "raise it and it rises." Reread 3.4.
- **Quoting performance measured on all the training data**: once again we deliberately computed R² on the full data. The problem with that score is exactly the next level's topic.

## Next Level Preview

For three levels running, the same warning has repeated: "grading on the data you trained on is optimistic." In level04 we finally prove why by experiment, and buckle machine learning's seatbelt — the train/validation/test three-way split.
