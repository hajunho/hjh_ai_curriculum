# Lecture 05 · Level 05 — Correlation and Causation

> Between "they move together" and "one moves *because of* the other" lies the distance of exactly one business decision.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03 / **Estimated time** 50 min

## 1. Why learn this — the business view

"Revenue was higher in the months we spent more on ads. Let's double the ad
budget." A line of reasoning you hear in meetings all the time. But what if
it was peak season — so ads went up *and* customers came in? Then doubling
the ad budget will not budge off-season revenue one bit, and that budget
evaporates into thin air.

**Correlation** is "a tendency for two values to move together";
**causation** is "a relationship where changing one changes the other". What
data analysis uncovers is, in most cases, only correlation. Yet **actions** —
allocating budget, hiring, setting prices — presuppose causation. The moment
you mistake correlation for causation, data-driven decision making becomes
gambling wrapped in data. This distinction is probably the single most
valuable page of this entire lecture for working professionals.

## 2. Understand it with an analogy

Look at the statistics of a summer beach town and **ice cream sales and
drowning accidents** move together with uncanny precision. Does ice cream
push people into the water? Of course not. A third variable — **hot weather**
— increases both ice cream sales and the number of people swimming. A
variable that influences both sides and manufactures a fake correlation like
this is called a **confounder**.

A confounder is **an octopus hiding its body**. All we see are the two arms
above the water (ice cream, drownings). Because the two arms move together
they look connected to each other, but in reality the body under the water
(the weather) is moving both. Grab and shake one arm (ban ice cream sales)
and the other arm will not move. The body hasn't changed.

## 3. Core concepts

### 3.1 How to read a correlation coefficient

The Pearson correlation coefficient r is a number between −1 and +1 that
measures how much two variables travel together **in a straight line**.

| r | Reading |
|---|---|
| +1.0 | perfect proportional co-movement |
| +0.7 ~ +0.9 | strong positive correlation |
| +0.3 ~ +0.5 | weak-to-moderate positive correlation |
| around 0 | no linear relationship |
| negative | move in opposite directions (e.g., price↑ sales↓) |

Two cautions. First, r catches only straight-line relationships. A U-shaped
relationship (revenue peaks at an ideal temperature) can come out as r ≈ 0.
That is why you always draw the scatter plot before reading the coefficient.
Second, the size of r is not the "slope" — it measures how tightly the dots
hug a straight line.

### 3.2 Four routes to a correlation

When a correlation between X and Y is observed, there are at least four
possible explanations.

1. **X → Y**: X really causes Y (ads → revenue)
2. **Y → X**: the reverse (we raised the ad budget in months when revenue was good)
3. **Z → X, Z → Y**: a confounder (peak season raised both ads and revenue)
4. **Chance**: with little data, entirely unrelated values happily move together

The analyst's job is not to *find* correlations, but to **eliminate**
possibilities until you know which of the four you are looking at.

### 3.3 Controlling a confounder — slicing into strata

The easiest way to probe a suspected confounder is **stratification**:
recompute the correlation only within groups where the confounder's value is
similar. "Looking only at the hot days", the ice-cream/drowning correlation
vanishes. The exercise reproduces this in code. If the correlation disappears
within strata, the original correlation was most likely the confounder's
handiwork.

### 3.4 The gold standard for causation — experiments

Establishing causation from observational data alone is very hard. The gold
standard is a **randomized experiment**. Flip a coin to decide which regions
get the ad and which don't, and every confounder — season, income, all of
them — gets evenly mixed into both sides and cancels out. This is exactly the
A/B test you will learn in level10, the practical endpoint of the
"correlation → causation" problem.

## 4. Practice — main.py

```bash
python3 main.py
```

The exercise has two parts.

**Part 1 — observing a real correlation**: compute the Pearson correlation
between daily ad spend and revenue in the `hjh_data` sales with numpy
(`np.corrcoef`), and save the scatter plot `outputs/ad_revenue.png`.

**Part 2 — manufacturing and dismantling a fake correlation**: we build the
confounder experiment ourselves with synthetic data.

- [3] Create a "heat index" (confounder Z), then generate ice cream sales
  X = Z + noise and swimming accidents Y = Z + noise. X and Y never reference
  each other, yet a strong correlation of r ≈ 0.7 appears.
- [4] **The stratification experiment**: split the heat index into 5 bands
  and recompute the X–Y correlation inside each band. Within bands, the
  correlation collapses toward 0.
- [5] `outputs/confounder.png`: the left panel is the overall scatter plot
  (looks strongly correlated); the right panel colors the dots by heat band
  (inside each colored cluster — disorder).

The output's key comparison is "overall r = 0.7 versus mean within-band
r ≈ 0". That moment — the correlation appearing and disappearing in the same
data — is everything this level wants to teach.

## 5. Try it yourself

1. **(Easy)** In part 2, increase the noise size (`noise_scale`). How does
   the overall correlation r change? Observe how the confounder's strength
   and the noise level together determine the strength of the correlation.
2. **(Medium)** Stratify part 1's ad-spend/revenue correlation by store.
   Do the overall correlation and the within-store correlations differ? If
   so, what are the confounder candidates?
   (Hint: `df.groupby("store")`, then `np.corrcoef` in each group)
3. **(Challenge)** Implement the correlation coefficient from its formula and
   check it matches `np.corrcoef`. (Hint: r = covariance / (std of X × std
   of Y), and covariance is `((x - x.mean()) * (y - y.mean())).mean()`)

## 6. Common mistakes

- **"r is big, so it's causal"**: even at r = 0.9, r does not tell you which
  of the four routes in 3.2 you are on. Strength of correlation and evidence
  of causation are separate things.
- **Reading the coefficient without the scatter plot**: U-shaped
  relationships and fake correlations created by a single outlier are visible
  only in the scatter plot. "Numbers first, skip the picture" is forbidden.
- **"No correlation = no relationship"**: r ≈ 0 only means "no *linear*
  relationship". Curved relationships are perfectly possible.
- **Ignoring reverse causation**: "products with many reviews sell well" may
  mean they sold well and therefore collected reviews. Check the time order —
  which came first?
- **Controlling one confounder and relaxing**: removing one via
  stratification does not mean another isn't still lurking. Conclusions from
  observational data are always provisional.

## Next level preview

To treat "it could be chance" seriously, you need the language of
probability. In the next level: probability basics, conditional probability,
and a famous problem that betrays intuition (the disease-testing paradox),
verified by simulation.
