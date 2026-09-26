# Lecture 05 · Level 06 — Probability Fundamentals

> Probability is not a fortune-teller's language — it is an accountant's language for "what happens if you repeat this for a long time".
**Difficulty** ⭐⭐⭐ / **Prerequisites** level05 / **Estimated time** 55 min

## 1. Why learn this — the business view

"87% probability this transaction is fraud", "32% probability this customer
will churn" — the outputs of the machine learning models you are about to
study are all probabilities. If you cannot read probability, you cannot read
the reports of the AI era. And probability betrays intuition constantly. The
fact that most alerts from a "99% accurate" fraud-detection system can be
false alarms is simply unbelievable until you do the math.

This level covers what probability means, conditional probability, and a
first taste of **Bayes' theorem**. In particular, we verify the famous
"disease-testing paradox" not with a formula but with a **simulation**. Have
the computer create a hundred thousand people and count — and you will see
with your own eyes that intuition was wrong and the calculation was right.

## 2. Understand it with an analogy

A 3% probability is not a prophecy that "it will/won't happen this time" —
it is a ledger entry: **live through the same situation 1,000 times and it
happens roughly 30 of them**. An insurance company has no idea whether *I*
will crash this year, but it knows quite precisely how many of 100,000
drivers like me will. That is how it can price the premium. Probability is a
statement about **the group's ledger**, not an individual's future.

Conditional probability is **filtering a business-card box**. "Of my client
contacts, the probability of being director-level or above" and "of
director-level contacts, the probability of being a client" filter the same
box in different orders — and give completely different values. Confusing
P(A|B) with P(B|A) is the most expensive mistake in probability, and it is
also the whole secret of the disease-testing paradox.

## 3. Core concepts

### 3.1 What probability means — long-run frequency

Saying a coin's heads probability is 1/2 does not mean "the next flip is
heads". It means that if you repeat the flip endlessly, the ratio of heads
approaches 1/2. This is the **law of large numbers**. Getting 7 heads in 10
flips is common; getting 7,000 in 10,000 essentially never happens. The
ratio converges — but slowly. That is the basis of the maxim "trust nothing
from a small sample".

### 3.2 Conditional probability — information changes probability

P(A|B) is "the probability of A given the information B". The calculation is
simple: keep only the cases where B holds, then count the fraction that are
also A.

P(A|B) = P(A and B) / P(B)

Direction is everything. P(positive|sick) — the probability that a sick
person tests positive — is the performance of the testing machine.
P(sick|positive) — the probability that someone who tested positive is
actually sick — is what matters to **you**, the person holding the result.
They are entirely different numbers.

### 3.3 The disease-testing paradox

- Prevalence 1% (10 sick people per 1,000)
- Sensitivity 99% (sick people test positive with 99% probability)
- Specificity 95% (healthy people test negative with 95% probability, i.e., 5% get an unfair positive)

Count it out for 100,000 people. Of 1,000 sick people, 990 test positive.
Of 99,000 healthy people, 4,950 get an **unfair positive**. Of the 5,940
positives, the actually sick number 990 — about 17%. "Positive on a 99%
accurate test", yet the chance of actually having the disease is one in six.
The reason: healthy people are overwhelmingly more numerous to begin with
(prior probability 1%), so their small false-alarm rate (5%) produces more
false alarms than there are patients at all.

That is the core insight of Bayes' theorem: **do not judge on the test
result (the evidence) alone — multiply in how common the thing was in the
first place (the prior).** Fraud alerts, spam filters, and hiring tests all
share this structure.

### 3.4 The Monty Hall problem — information is not free

A prize sits behind one of three doors. You pick one; the host — who knows
the answer — opens one of the other two doors to reveal a dud, and asks,
"Would you like to switch?" Is switching better? Intuition says "50:50", but
the answer is **2/3 if you switch, 1/3 if you don't**. The host's action
carries information (he was able to *choose* a losing door to open). When the
argument breaks out, skip the algebra: ten thousand simulated games settle it
— which is exactly what the exercise does.

## 4. Practice — main.py

```bash
python3 main.py
```

We run three simulations — each a duel between intuition and repeated
experiment.

- [1] **Law of large numbers**: prints how close the cumulative heads ratio
  is to 0.5 after 10 / 100 / 10,000 flips.
- [2] **The disease-testing paradox**: creates a virtual population of
  100,000, assigns disease status and test results, and directly counts "the
  fraction of positives who are actually sick". Compared against the
  theoretical value from Bayes' theorem.
- [3] **Monty Hall**: simulates 10,000 games each for the "switch" strategy
  and the "stubborn stay" strategy and compares win rates.

[2] is the heart of this level. The code uses almost no probability formulas
— it just creates a hundred thousand people and **counts** those matching
each condition. The very structure of the code shows that conditional
probability is, in the end, "filter, then count".

## 5. Try it yourself

1. **(Easy)** Change the prevalence in [2] from 1% → 10% → 30%. How does the
   fraction of positives who are actually sick change? Feel the power of the
   prior.
2. **(Medium)** Rework [2] into a "fraud-detection" version: fraud rate
   1.5%, detection sensitivity 90%, false-alarm rate 3%. Out of 100 alerts,
   how many are real fraud? (Hint: rename the variables — the structure is
   identical.)
3. **(Challenge)** Turn Monty Hall into a 100-door version. The host opens
   98 duds; what is the win rate if you switch? Predict first, then confirm
   by simulation. (Hint: the probability your first pick was right is still
   1/100.)

## 6. Common mistakes

- **Confusing P(A|B) and P(B|A)**: whenever you read "the test is 99%
  accurate", check which direction the probability points. It is usually
  P(positive|sick).
- **Trusting a small sample's ratio as a probability**: 4 satisfied
  customers out of 5 does not justify reporting "80% satisfaction". The
  uncertainty of that number is level08's confidence intervals.
- **Multiplying non-independent things as if independent**: "each customer
  churns with 10% probability, so both churn with 1%" holds only if the two
  are independent. Customers who suffered the same outage churn together.
- **Forgetting the rare-event + imperfect-detector combination**: the rarer
  the target, the more of your alerts are false alarms. When adopting any
  alert system, always ask about the prior.

## Next level preview

Flip a coin very many times and the distribution of the number of heads
becomes a beautiful bell shape. Why does this bell (the normal distribution)
appear all over the world? We confirm the theorem behind it — the Central
Limit Theorem — with simulation pictures.
