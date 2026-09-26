# Lecture 05 · Level 09 — Hypothesis Testing and p-values

> A hypothesis test is a trial. The data is the evidence, and the p-value is "the chance of seeing evidence like this if the defendant is innocent".
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level08 / **Estimated time** 60 min

## 1. Why learn this — the business view

You report "the renovated store's revenue is 8% higher", and an executive
asks: "couldn't that be chance?" The tool that answers this question with a
procedure rather than a hunch is **hypothesis testing**. Drug approvals,
quality pass/fail calls, journal peer review, A/B tests — a great many of the
world's official judgments run on this procedure.

At the same time, the p-value is probably the most misunderstood concept in
statistics. The interpretation "p < 0.05, so our hypothesis is 95% likely to
be right" is widespread — and wrong. This level has two goals: to **build a
p-value with your own hands (by simulation)** so its definition sinks in, and
to confirm that scipy's one-line test agrees, so you can use the tool with
confidence.

## 2. Understand it with an analogy

Criminal trials operate under the presumption of innocence: the defendant is
**considered innocent until proven guilty**. The prosecutor's job is to
present evidence "too strange to be consistent with innocence".

Hypothesis testing has exactly this structure.

| The trial | The hypothesis test |
|---|---|
| The defendant is innocent (default premise) | Null hypothesis H₀: "no difference, pure chance" |
| The prosecutor's claim | Alternative hypothesis H₁: "there is a real difference" |
| The evidence | The observed data (the revenue gap between two stores) |
| "If innocent, evidence like this shouldn't appear" | p-value: probability of a gap this large arising by chance if H₀ is true |
| Beyond reasonable doubt | p < 0.05 → reject H₀ (5% significance level) |
| Acquittal ≠ proof of innocence | Failing to reject ≠ "no difference proven" |

The last row matters most. Just as an acquittal for lack of evidence is not
proof of innocence, p = 0.3 does not prove "there is no difference" — it only
says "chance alone could explain this".

## 3. Core concepts

### 3.1 The four-step testing procedure

1. **State the null hypothesis H₀**: "the two stores' true mean revenue is equal."
2. **Compute the test statistic**: the observed difference divided by its
   standard error (the t-statistic).
3. **Compute the p-value**: in a world where H₀ is true, the probability of
   a difference this large or larger arising by chance.
4. **Judge**: if p is below the pre-chosen significance level (usually 0.05),
   reject H₀.

The significance level must be fixed **before the trial**. Moving the
threshold after seeing the result is painting the target around the arrow.

### 3.2 The list of things a p-value is NOT

- ❌ the probability that H₀ is true (that is a probability about the
  hypothesis, not the data — Bayesian territory)
- ❌ the probability that the alternative hypothesis is right
- ❌ the size of the effect (with a big enough sample, a trivial difference gets a tiny p)
- ❌ the probability of replication
- ⭕ **"in a world with no difference, the probability of observing a
  difference this large (or larger) by chance"**

The correct report for p = 0.03: "If the two stores were truly identical,
a gap this size would arise by chance 3% of the time. That is hard to call
chance, so we judge there is a difference."

### 3.3 The two errors — false alarms and misses

| | Truly no difference | Truly a difference |
|---|---|---|
| Test says: difference | **Type I error (α)** false alarm | correct detection (power) |
| Test says: withhold judgment | correct acquittal | **Type II error (β)** miss |

A significance level of 0.05 is a contract: "even with no difference, we
accept a false alarm 5 times in 100." Fields where false alarms are fatal
(new drugs) lower α; fields where misses are fatal (cancer screening) raise
power. There is no free lunch.

### 3.4 The permutation test — a p-value made by simulation

You can build a p-value without the t-test formula. If H₀ says "the two
stores are the same", then the store labels attached to the revenue data
should mean nothing. So **shuffle the labels wildly** (a permutation),
compute the difference, and repeat ten thousand times: out comes the
distribution of "differences produced by chance alone". Count where the
actually observed difference sits in that distribution's tail — that is the
p-value. A p-value built straight from the definition, no formula — the
heart of this exercise.

## 4. Practice — main.py

```bash
python3 main.py
```

We put the two stores' revenue gap on trial.

- [1] Draw per-transaction revenue samples for the Airport and University
  stores from the `hjh_data` sales, and print the observed difference in
  means.
- [2] **Permutation test**: shuffle the store labels 10,000 times to build
  the "chance difference" distribution, and count the share more extreme
  than the observed difference = the p-value.
- [3] **The scipy test**: test the same question with one line —
  `stats.ttest_ind()` (Welch's t-test) — and confirm the two p-values are
  close.
- [4] `outputs/permutation.png`: the chance distribution's histogram with
  the observed difference marked — engraving "p-value = the area of the
  tail" as a picture.
- [5] **Control experiment**: run the same test on two samples with no real
  difference (one store split in half) and see p come out large — plus what
  a Type I error means.

In the output, check that the permutation p and the scipy p agree to two or
three decimal places. The formula is just a shortcut for the simulation —
they compute the same thing.

## 5. Try it yourself

1. **(Easy)** Halve the sample size in [1]. For the same size of gap, how
   does p change? Confirm that "with a small sample, even a real difference
   is hard to prove."
2. **(Medium)** Switch to two stores with almost no gap (adjacent ranks) and
   test. If p exceeds 0.05, how should the report sentence read — given it
   must *not* say "there is no difference"? (Hint: the last row of the table
   in section 2.)
3. **(Challenge)** Extend [5]: repeat the "no-difference test" 1,000 times
   and count how often p < 0.05. Is it near 5%? That is the Type I error
   rate in the flesh. (Hint: re-split the sample fresh inside each loop
   iteration.)

## 6. Common mistakes

- **Reporting p as "the probability the hypothesis is right"**: reread the
  list in 3.2 before the meeting. Get it wrong in front of someone who knows
  statistics and you lose credibility.
- **Treating p = 0.049 and 0.051 as heaven and earth**: 0.05 is a
  conventional line, not a cliff. Report effect size, cost, and
  reproducibility alongside.
- **Significant = important**: with a million rows, even a 0.1% revenue
  difference gets p < 0.001. "Statistically significant" and "meaningful for
  the business" are separate questions.
- **Testing many things and reporting only the good ones**: test 20 metrics
  and, even with no differences at all, on average one comes out p < 0.05
  (the multiple-comparisons problem). A report that hides how many tests
  were run is a distortion. This is the sibling of the next level's peeking
  problem.
- **Reporting a failed rejection as "proven equal"**: insufficient evidence
  and proof of innocence are different things.

## Next level preview

Transplant hypothesis testing into business product development and you get
the A/B test. Comparing conversion rates, designing the sample size, and
"why you must not peek at results early" — verified with a simulator.
