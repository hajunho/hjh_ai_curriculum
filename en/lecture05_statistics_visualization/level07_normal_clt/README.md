# Lecture 05 · Level 07 — The Normal Distribution and the Central Limit Theorem

> Whatever shape individual data points take, their "means" gather into a bell. It is the law of gravity of statistics.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level06 / **Estimated time** 55 min

## 1. Why learn this — the business view

The margin of error in opinion polls, the control limits in quality
management, significance calls in A/B tests — beneath all of these working
tools lies one single fact: **the sample mean follows a normal
distribution**. Whether the underlying data has a long right tail like
revenue, or takes only two values like a coin flip, the distribution of "the
average of several draws" turns into a bell. This is the **Central Limit
Theorem (CLT)**.

Thanks to this theorem, we can make probabilistic statements about a mean
without knowing the underlying distribution. Sentences like "the probability
that this month's sample mean falls outside this range is under 5%" are
possible entirely because of the CLT. The next level's confidence intervals,
and the p-values after that, all stand on top of it — this is the cornerstone
of inferential statistics.

## 2. Understand it with an analogy

Think of the **taste of one bowl of soup** at a restaurant. Inside the pot
there are saltier spots and blander spots (the raw distribution is all over
the place). But the taste of a bowl filled by several ladle scoops, stirred
together? Every bowl tastes about the same. The mixing of many spots (the
averaging) cancels the extremes. Record the taste of hundreds of bowls, and
the distribution of "bowl taste" becomes a narrow bell — regardless of how
lumpy the pot is. The bigger the ladle (the larger the sample), the smaller
the bowl-to-bowl differences.

The same logic explains why normal distributions are so common in the world.
Height is the sum of countless genetic and environmental factors;
measurement error is the sum of countless small wobbles. **"The sum of many
independent small factors becomes bell-shaped"** — that is the everyday
version of the CLT.

## 3. Core concepts

### 3.1 The normal distribution — two knobs

The normal distribution is a symmetric bell shape completely determined by
just two knobs: the mean μ and the standard deviation σ. The famous
68-95-99.7 rule:

| Range | Share included |
|---|---|
| μ ± 1σ | about 68% |
| μ ± 2σ | about 95% |
| μ ± 3σ | about 99.7% |

The quality-control convention "outside ±3σ means the process is off", and
the way "±2σ ≈ 95%" turns into the 1.96 of confidence intervals — both come
straight from this table.

### 3.2 The Central Limit Theorem — the precise statement

For any distribution with mean μ and standard deviation σ (as long as its
tails are not pathologically heavy), if you repeatedly draw samples of size
n and take their means:

1. the mean of the sample means approaches μ,
2. the standard deviation of the sample means becomes **σ/√n** (called the
   **standard error**), and
3. the shape of the sample means' distribution approaches the **normal
   distribution** as n grows.

The third point is the astonishing one. Whether the source is a flat uniform
distribution or a left-piled exponential, by around n=30 the distribution of
means is already quite bell-shaped.

### 3.3 The economics of √n

That the standard error is σ/√n means, in practice: "to double the
precision, collect four times the sample." Growing a survey from 1,000 to
2,000 respondents shrinks the error only by a factor of 1.4. Data collection
costs grow linearly while precision improves only with the square root, so
"how much sample do we collect?" is always a cost-versus-precision trade-off
(this leads into level10's sample-size design).

### 3.4 Where the CLT breaks down

The CLT is not almighty. In distributions dominated by extremes — a business
where one jackpot deal is half the annual revenue, an insurer's catastrophe
losses — the mean itself is unstable, so convergence is very slow or fails.
That is why you must first ask, "is the mean even meaningful for this data?"
(connecting back to level01's lesson).

## 4. Practice — main.py

```bash
python3 main.py
```

You witness the CLT directly by simulation.

- [1] Verifies the 68-95-99.7 rule with a million random numbers and prints
  the counts.
- [2] From a **uniform distribution** (a boxy shape), draws sample means of
  size n = 1, 5, 30 — 5,000 times each — and histograms them.
- [3] Repeats the experiment with an **exponential distribution** (piled on
  the left, long right tail).
- [4] Saves the results to `outputs/clt_grid.png` as a grid of 2 rows
  (distribution type) × 3 columns (sample size). Each panel overlays the
  theoretical normal curve, showing the histogram gluing itself to the curve
  as n grows.
- [5] Prints a table checking that the standard deviation of the sample
  means matches the theoretical σ/√n.

What to look for in the figure: the left column (n=1) is the raw
distribution — boxy or lopsided. The middle (n=5) is already rounding, and
the right (n=30) hugs the bell. **In the right column, what the original
looked like has vanished almost without a trace.**

## 5. Try it yourself

1. **(Easy)** Redraw with sample sizes n = 2, 10, 100. Around which n does
   the exponential's skew feel like it has "sufficiently" disappeared?
2. **(Medium)** Using the table in [5], check that quadrupling n (5→20)
   exactly halves the standard error. (Hint: in σ/√n, √4 = 2.)
3. **(Challenge)** Extend the experiment to an extreme-dominated
   distribution: build one where, with 1% probability, a value 100 times the
   mean sneaks in, then histogram the n=30 sample means. What shape do you
   get, and why does CLT convergence slow down?
   (Hint: `np.where(rng.random(n) < 0.01, 100 * mu, base_sample)`)

## 6. Common mistakes

- **The misconception "with enough data, the data becomes normal"**: the CLT
  is a theorem about **the distribution of sample means**, not about the raw
  data. Revenue data still has a right tail after a million rows.
- **Confusing σ and σ/√n**: the standard deviation is the scatter of
  individual values; the standard error is the scatter of sample means. In
  reports, always state which one follows the "±".
- **Memorizing n=30 as a magic number**: 30 is just a convention. The more
  skewed the source, the larger the n you need. When in doubt, simulate —
  exactly as we did today.
- **Assuming everything is normal**: assume normality for heavy-tailed data
  like financial returns or viral view counts, and the "once in a millennium"
  event happens three times a decade.

## Next level preview

Now that the CLT tells us how much a sample mean wobbles, we can speak in
reverse: "given this sample mean, the true mean should be within this range."
In the next level we build confidence intervals and verify what "95%" really
means through repeated experiments.
