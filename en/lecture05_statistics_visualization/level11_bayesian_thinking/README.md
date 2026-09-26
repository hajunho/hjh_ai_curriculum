# Lecture 05 · Level 11 — Bayesian Thinking and Communicating Uncertainty

> Bayesian analysis produces not "the answer" but "a ledger of belief". Each piece of evidence updates the balance.
**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level10 / **Estimated time** 70 min

## 1. Why learn this — the business view

The p-value of level09–10 was a language unnatural enough to require
training. Say to an executive, "assuming no difference, the probability of a
gap this size arising by chance is 3%", and the question that comes back is
always the same: **"So what's the probability that B is actually better?"**
The p-value cannot answer that question — but the **Bayesian** approach
answers it head-on: "On current data, the probability that option B is
superior is 92%."

You already met the skeleton of Bayesian thinking in level06 (the
disease-testing paradox: prior × evidence). This level extends it to
continuous problems — estimating a conversion rate and comparing A vs B —
and finishes with a skill arguably more important than the statistics
itself: **how to talk about uncertain results with decision makers**.

## 2. Understand it with an analogy

A Bayesian update is a **reputation update after a first date**. Even before
meeting, you are not a blank slate — the matchmaker's description and the
profile photo form your **prior**. If the first meeting (the evidence) goes
well, your regard rises — but you are still not certain. As meetings
accumulate (data), the belief distribution narrows, and at some point the
matchmaker's opinion (the prior) loses nearly all its influence.
**Posterior = prior × evidence** — and today's posterior becomes tomorrow's
prior.

Communicating uncertainty is **the language of the weather forecast**. The
forecaster does not say "it will/won't rain tomorrow" but "70% chance of
rain". This is a good report not because it can never be wrong, but because
**it lets the listener weigh the cost of an umbrella against the cost of
getting wet for themselves**. "92% probability that B is superior; if it is
inferior, the expected loss is KRW 3 million a month" — that is a report
formatted for a decision maker.

## 3. Core concepts

### 3.1 Two views of probability

- **Frequentist**: probability = long-run repetition ratio. Parameters are
  fixed unknowns, and probability attaches only to the data (the sampling).
  → confidence intervals, p-values
- **Bayesian**: probability = degree of belief. The parameter itself gets a
  probability distribution, updated each time you see data.
  → posterior distributions, probability of superiority

They are not enemies but tools with different uses. Regulatory submissions
and papers lean on frequentist conventions; continuous business decisions
and early monitoring suit the Bayesian idiom.

### 3.2 Beta-distribution updates for a conversion rate

A belief about a value between 0 and 1 — like a conversion rate — is
expressed with the **Beta distribution**. Read Beta(α, β) roughly as "the
belief of someone who has seen α−1 successes and β−1 failures". The update
rule is astonishingly simple:

> prior Beta(α, β) + observations (s conversions, f non-conversions) → posterior Beta(α+s, β+f)

- The know-nothing start: Beta(1, 1) — flat, every value from 0 to 1 equally plausible
- Observe 3 conversions in 100 visits → Beta(4, 98): a bump near 3%, still wide
- After 10,000 visits → the bump narrows to a needle: "now fairly certain"

The **width of the distribution is the size of the uncertainty**. Thinking
in distributions ("somewhere between 2.1% and 4.2%, most plausibly near 3%")
instead of point estimates ("conversion is 3%") is the core Bayesian habit.

### 3.3 Computing "the probability B is superior"

Given the posteriors of A and B, draw tens of thousands of random pairs from
the two distributions and count **the fraction where the B draw exceeds the
A draw**. That is the Monte Carlo probability of superiority, P(B > A). No
formula — level06's "build it and count" works here too. As a bonus, the
distribution of "by how many percentage points is B better" (expected lift,
risk of loss) falls out of the same draws for free.

### 3.4 How to talk uncertainty to executives

1. **Answer in the language of the question**: not "p=0.03" but "92%
   probability that B is better".
2. **Speak in intervals**: not "conversion 3.4%" but "3.1–3.7% (most
   plausible value 3.4%)".
3. **Connect to the decision**: "92% confidence + KRW 3M/month loss if
   wrong + 5,000 more samples per week if we wait" — whether to decide now
   or wait is the executive's call; the analyst's job is to deliver those
   three numbers accurately.
4. **The courage to say "we don't know"**: when the distribution is wide,
   "it is too early to judge; we need n more observations" is the only
   honest report.

## 4. Practice — main.py

```bash
python3 main.py
```

- [1] **The growth of a belief**: for a virtual service whose true
  conversion rate is 3.2%, compute how the Beta posterior narrows as
  visitors accumulate 0 → 100 → 1,000 → 10,000, printing the 95% credible
  interval at each stage. The four distribution curves are overlaid and
  saved to `outputs/beta_update.png`.
- [2] **The Bayesian A/B verdict**: from the posteriors of A (610/20,000
  conversions) vs B (668/20,000), draw 100,000 Monte Carlo pairs and compute
  P(B > A), the expected lift, and "the average loss when B is actually
  worse (the expected loss)". The two posteriors are drawn together in
  `outputs/ab_posterior.png`.
- [3] **Auto-generated report**: prints the numbers from [2] as a three-line
  executive report in the format of 3.4.

Note the new term, *credible interval*. A Bayesian interval is one you
**may** describe as "the true value is in this interval with 95%
probability" (contrast level08, where the confidence interval forbade that
phrasing). The price is stating your assumption — the prior — explicitly.

## 5. Try it yourself

1. **(Easy)** Replace the prior in [1] with Beta(3, 97) — encoding "the
   industry average is around 3%" — instead of Beta(1,1). How does the
   early-stage (100-visitor) distribution change, and why has the difference
   nearly vanished after 10,000?
2. **(Medium)** In [2], reduce B's conversions from 668 to 640 and watch how
   P(B>A) changes. When you get an awkward probability between 60% and 70%,
   write the report sentence yourself in the 3.4 format.
3. **(Challenge)** Implement the decision rule "launch when the expected
   loss drops below 0.02 pp", and run a simulation that evaluates the rule
   after every 5,000 visitors. How does Bayesian monitoring soften the
   peeking problem? (Hint: the *interpretation* of the probability is
   different, so looking repeatedly keeps its meaning — but the decision
   rule must still be fixed in advance.)

## 6. Common mistakes

- **Hiding the prior**: the prior is not an embarrassing assumption but an
  input to declare. Write one line in the report, e.g. "prior:
  uninformative Beta(1,1)".
- **Misreading P(B>A)=92% as "8% chance of catastrophe"**: most of the
  remaining 8% is "B is very slightly worse". Only with the expected loss
  next to the superiority probability can you see the size of the risk.
- **Looking at the point instead of the distribution**: extract only the
  posterior mean and you have gained nothing from going Bayesian. The width
  (the credible interval) is the key information.
- **Blaming the prior when the sample is plentiful**: as data accumulates,
  all reasonable priors converge to nearly the same posterior. When the
  prior debate drags on, it usually signals a shortage of data.
- **Mistaking uncertainty reporting for lack of confidence**: the analyst
  who says "92%" earns more trust over the long run than the analyst who
  says "certain". The latter loses all credibility the moment they are
  wrong.

## Next level preview

lecture05 ends here. Summary (means) → pictures (distributions) →
relationships (correlation) → inference (tests) → decisions (A/B, Bayesian):
you have made one full loop of handling reality with numbers. In the next
lecture (lecture06, Introduction to Machine Learning), we place a
"prediction machine" on top of this statistics. The language of probability
you learned today becomes the language for reading model outputs.
