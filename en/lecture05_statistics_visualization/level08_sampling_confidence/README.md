# Lecture 05 · Level 08 — Samples and Confidence Intervals

> Surveying everyone is impossible. Instead, we certify the reliability of "the way we throw the net".
**Difficulty** ⭐⭐⭐ / **Prerequisites** level07 / **Estimated time** 55 min

## 1. Why learn this — the business view

You cannot ask all 500,000 customers about their satisfaction. You ask 500
and infer the whole. And the question that inevitably follows is: "how much
can we trust the result from 500?" The standard-format answer is the one you
see in poll coverage: "approval 42%, margin of error ±3.1 percentage points
(95% confidence level)."

Being able to read and build a **confidence interval** changes your work.
Instead of reporting "new-product satisfaction 78%", write "78% (95% CI:
74–82%)" and the reader knows exactly how far this number can be trusted for
a decision. Conversely, you will be able to explain why "85% satisfaction"
from a 20-person survey says almost nothing. This level is inferential
statistics' first practical tool.

## 2. Understand it with an analogy

To taste the soup you do not need to drink the pot. **One spoonful is enough
— provided you stirred well.** The heart of sample surveys is not the size
of the spoon but "did you stir?" (random sampling). Scoop only from the top
of an unstirred pot (a biased sample) and no spoon, however large, will catch
the saltiness.

The meaning of "95% confidence" is subtle, so the analogy matters
especially. Think of a **ring toss**. The stake (the true value) is fixed in
place; we throw rings (intervals). "95% confidence level" means **this
throwing method hooks the stake 95 times out of 100**. Looking at one ring
already thrown and saying "there is a 95% probability the stake is inside
this ring" is subtly off — a thrown ring either hooked the stake or it
didn't, and the 95% is the report card of the **throwing method**, not of
any one ring. In the exercise, we throw the ring 100 times and count the
hits.

## 3. Core concepts

### 3.1 Population and sample

- **Population**: the whole you want to know about (all customers, all transactions)
- **Sample**: the part you actually observed
- **Parameter**: the population's true value (the true mean μ) — forever unknown
- **Statistic**: a value computed from the sample (the sample mean x̄) — what you hold

Inference is the operation of encircling μ with x̄.

### 3.2 Sampling error and the confidence interval formula

In level07 we saw that the sample mean's wobble is the standard error σ/√n.
Since 95% of a normal distribution lies within ±1.96σ, the 95% confidence
interval is:

x̄ ± 1.96 × (s / √n)

Here s is the sample standard deviation (we don't know σ, so the sample
stands in; for small n we use the t-distribution's value instead of 1.96 —
scipy handles that).

How to read it: a narrow interval means plenty of information; a wide one
means "we don't know yet". Width scales as 1/√n, so halving it requires four
times the sample.

### 3.3 What "95%" precisely means

Build intervals by the procedure in 3.2 and, **over endless repetitions of
sampling, 95% of those intervals contain the true μ**. Any single interval
either captured it or it didn't — and we don't know which. Why this subtlety
matters at work: reporting "with 95% probability the mean revenue is in this
range" doesn't sound wrong, but precisely speaking it is "this method is
right 95 times out of 100, and we hope this was one of the 95." (If you want
to assign probability to the interval itself — that is level11's Bayesian
territory.)

### 3.4 Scarier than the margin of error — bias

The margin of error is computed **assuming random sampling**. Survey by
landline during the day and you capture only the stay-at-home population;
when only satisfied customers leave reviews, ratings inflate; study only the
companies that survived and the recipe for success is distorted (survivorship
bias). **Bias does not go away as the sample grows** — an unstirred pot
tastes salty even from a big spoon. Before asking about sample size, ask
"how was it drawn?"

## 4. Practice — main.py

```bash
python3 main.py
```

We actually play the ring toss.

- [1] Take the cafe chain's entire sales (~20,000 rows) as the "population"
  and compute the true mean μ up front (unknowable in real life, but this is
  an experiment).
- [2] Draw an n=50 sample from the population and build a 95% confidence
  interval — 100 times — counting how many intervals contain μ. Check the
  capture rate is near 95%.
- [3] `outputs/ci_rings.png`: the 100 intervals drawn against a vertical
  line (the true μ). Only intervals that missed μ are red — the "throws that
  missed the stake" stand out at a glance.
- [4] For sample sizes n = 20/50/200/800, confirm with a table and
  `outputs/ci_width.png` that the interval width narrows as 1/√n.
- [5] **Bias experiment**: build intervals from a biased "weekends-only"
  sample and watch them miss μ no matter how large the sample grows.

[5] is the most important part for practice. A biased sample's confidence
interval is narrow and self-assured — and **consistently wrong**. Remember:
what statistics certifies extends only as far as the randomness of the
procedure.

## 5. Try it yourself

1. **(Easy)** Rerun [2] with confidence levels of 90% (1.645) and 99%
   (2.576). How do the capture rate and interval width change? What is the
   price of "a more confident report"?
2. **(Medium)** Increase the repetitions in [2] from 100 to 2,000 and check
   whether the capture rate moves closer to 95%. (The law of large numbers
   is at work here too.)
3. **(Challenge)** Build a confidence interval for a proportion: in a
   survey, 210 of 500 answered "satisfied". Compute the 95% interval with
   p̂ ± 1.96×√(p̂(1−p̂)/n), then write a report sentence in the "margin of
   error ±X percentage points" format. (Hint: start from p̂ = 0.42.)

## 6. Common mistakes

- **Judging reliability by sample size alone**: an n=10,000 biased sample is
  worse than an n=100 random one. How it was drawn comes before how big it
  is.
- **Reporting point estimates without intervals**: "churn 12.3%" alone could
  mean 12.3±0.2 or 12.3±8 — nobody can tell. Attach an interval to any
  number a decision rides on.
- **"The intervals overlap, so no difference"**: two intervals can overlap
  slightly while the difference between the values is still significant.
  Testing differences properly is the next level (the t-test).
- **Mistaking the margin of error for the maximum error**: ±3.1 percentage
  points is just the 95%-level width; one time in 20 the truth falls
  outside. And bias is a separate problem altogether.

## Next level preview

"Store A looks like it outsells store B — but couldn't that be chance?" The
tool that answers this question procedurally is hypothesis testing. Using the
presumption of innocence in a courtroom as our metaphor, we learn what a
p-value really means.
