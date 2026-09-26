# Lecture 05 · Level 10 — Designing and Interpreting A/B Tests

> An A/B test is a machine that converts correlation into causation. It works only when you follow the rules.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level09 / **Estimated time** 65 min

## 1. Why learn this — the business view

"If we change the buy button from green to orange, will conversion go up?"
Historical data cannot answer this — the promotion probably changed at the
same time the button did (a confounder, level05). Only an experiment can
answer it. Split visitors **randomly** in half, show half the current design
(A) and half the new one (B), and compare conversion rates — that is the A/B
test. Random assignment sprinkles every confounder — season, weekday,
customer disposition — evenly across both sides, so what remains is only the
button color's causal effect plus chance. Chance gets filtered out with
level09's test.

Tech companies run thousands to tens of thousands of A/B tests a year, and
the practice long ago became standard in e-commerce, finance, and gaming.
But more important than knowing the procedure is **knowing how it gets
ruined**. Above all, "ending early because the result looks good" (peeking)
is the most common act of fraud committed in good faith.

## 2. Understand it with an analogy

The rules of an A/B test are the rules of a **coin-flip wager**. If you
agreed to "settle it over 100 flips", you must flip all 100. Now imagine
your opponent watches the running score and yells "Stop! I win!" the moment
they pull ahead. Even with a fair coin, a moment when one side is briefly
ahead almost always comes. **Choose the ending time after seeing the score,
and luck masquerades as skill.** That is what peeking is.

Sample-size design is **choosing a telescope**. Reading a small sign far
away (a small effect) requires a large telescope (a large sample). Just as
an unreadable sign isn't a nonexistent sign, a non-significant result from an
undersized sample isn't a nonexistent effect. Computing "how many people do
I need to see an effect this size?" before the experiment is power analysis.

## 3. Core concepts

### 3.1 The design procedure — four things fixed before the experiment

1. **The metric**: what decides the contest (conversion, basket size, return
   rate — exactly one primary metric)
2. **The minimum detectable effect (MDE)**: how many percentage points of
   difference matter to the business (e.g., +0.5pp)
3. **The sample size**: how many per group to catch that effect with 80% power
4. **The stopping rule**: when the sample is complete, test **once**

It is barely an exaggeration to say these four, written down before the
experiment, *are* the A/B test. Everything else is level09's test, verbatim.

### 3.2 Testing a difference in conversion rates

A conversion is binary data — did or didn't. With n per group and conversion
rates p̂_A and p̂_B, the difference of the two proportions is tested with a
z-test. The principle is the t-test's (difference ÷ standard error), and in
the exercise we compute it by hand without scipy. The standard error
contains p̂(1−p̂)/n, so the lower the conversion rate and the smaller the
difference, the faster the required sample balloons.

### 3.3 A feel for power and sample size

Power is "the probability of catching a real effect when there is one".
The conventional target is 80%. A rough feel:

| Baseline conversion | Difference to detect | Sample needed per group (approx.) |
|---|---|---|
| 3% | +1.0pp | ~5,000 |
| 3% | +0.5pp | ~19,000 |
| 3% | +0.2pp | ~110,000 |

Halve the difference and you need about 4x the sample (the economics of √n,
level07). When someone asks "we'll have results in a day, right?", this
table is the answer.

### 3.4 Peeking — why you must not look early

What happens if you check the p-value daily and stop the moment p < 0.05?
Even at a designed 5% significance level, the actual Type I error rate
(concluding "there is an effect" when there is none) soars to 20–30%. Run
the test many times and one of those days will cross the threshold by chance
(the multiple-comparisons problem, played out in time). The p-value is a
tool built on the premise of "one look, at a predetermined time".

Three countermeasures: (1) don't look until the planned sample is full,
(2) if interim checks are essential, tighten the threshold accordingly
(sequential testing designs), (3) Bayesian monitoring (level11). The easiest
in practice is (1) — merely hiding the p-value from the dashboard reduces an
organization's false alarms.

## 4. Practice — main.py

```bash
python3 main.py
```

We build a conversion A/B test simulator and run three experiments.

- [1] **With a real effect**: A=3.0%, B=3.6%, 20,000 per group. Prints the
  observed conversion rates, the z-test p-value for the difference, and the
  verdict.
- [2] **With no effect (an A/A test)**: A=B=3.0%. Watch how the test behaves
  in a "no difference" world.
- [3] **Verifying the Type I error rate**: repeat the no-effect test 2,000
  times and confirm the share with p < 0.05 lands near the designed 5%.
- [4] **The peeking experiment**: in the same 2,000 no-effect tests, mimic a
  policy that checks p after every 1,000 users and "declares victory and
  stops" if p < 0.05 even once. Count how many times over the false-positive
  rate multiplies.
- [5] `outputs/peeking.png`: p-value trajectories of several no-effect
  tests. The trajectories that dip briefly below the 0.05 line and come back
  up are "the moments peeking takes the bait".

Remember the output of [4] — "honest test 4.9% vs peeking 24%". That is the
number to pull out when someone in a meeting says "the interim results look
good, let's stop early."

## 5. Try it yourself

1. **(Easy)** Cut [1]'s per-group sample from 20,000 to 3,000. Does the same
   0.6pp effect still come out significant? Does that match the feel of the
   table in 3.3?
2. **(Medium)** In [4], shrink the check interval from 1,000 to 200 users
   (peek more often). What happens to the false-positive rate? Run it and
   see.
3. **(Challenge)** Build a simple sample-size calculator: take the baseline
   conversion p and the target difference d, and reproduce the 3.3 table
   with the approximation "n per group ≈ 16 × p(1−p) / d²".
   (Hint: this approximation is the conventional rule of thumb for a
   two-sided test at 5% significance and 80% power.)

## 6. Common mistakes

- **Breaking random assignment**: morning visitors get A, afternoon gets B —
  that is not an A/B test, it is a morning/afternoon comparison. Assign by
  per-user random numbers, always.
- **Choosing the metric after seeing results**: conversion didn't move but
  session time did, so — success! That is painting the target around the
  arrow. One primary metric, chosen in advance.
- **Early stopping (peeking)**: the subject of this level. "It looked good
  so we stopped early" is a false-positive factory.
- **Segment-slicing abuse**: no overall effect, but "it was significant for
  women in their 30s" — rummage through 20 segments and one will trip by
  chance. Use segment findings only as hypotheses for the next experiment.
- **Statistical significance = launch decision**: even a significant effect
  isn't worth launching if it is smaller than the operating cost. Report the
  effect size and confidence interval along with the p-value.

## Next level preview

The p-value only answers "is it chance?". To answer the executive's real
question head-on — "what is the probability that B is better?" — you need
Bayesian thinking. In the final level we answer it with Beta-distribution
updates on conversion rates.
