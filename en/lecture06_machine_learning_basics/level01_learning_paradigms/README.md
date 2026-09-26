# Lecture 06 · Level 01 — Supervised, Unsupervised, and Reinforcement Learning

> We organize machine learning's three learning paradigms with analogies — private tutoring, a self-directed study group, and puppy training — and run a toy example of each.
**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 30 min

## 1. Why Learn This — The Business View

When a vendor pitches "our anomaly-detection solution is based on unsupervised learning," or your data team says "this needs supervised learning, so labeling has to happen first," you can't even hold the conversation without knowing these three terms. There's a more practical reason too: **which paradigm you use completely changes the data you must prepare and what it will cost.** Supervised learning carries the cost of creating answer labels; unsupervised learning carries the human cost of interpreting the results; reinforcement learning needs an environment that can tolerate trial and error. The moment you decide which paradigm a problem gets solved with, the broad shape of the project budget is decided too.

## 2. Understanding by Analogy

Picture three ways of studying.

**Supervised learning = private tutoring.** The tutor hands you problems together with the answer key. "The answer to this one is C." The student sees problem→answer pairs over and over, absorbs the pattern, and later solves new problems that come without an answer key. Someone has to exist who can produce that answer key (the labels).

**Unsupervised learning = a self-directed study group.** There is no answer key. Instead you receive a pile of material and the assignment "group similar things together and find the structure." It's the job of a librarian handed 10,000 books with no classification scheme, who has to shelve similar books together. The machine creates the groupings, but naming each group — "ah, this shelf is the mystery novels" — is a human's job.

**Reinforcement learning = puppy training.** No answer key, no pile of material. When the puppy sits on "sit," it gets a treat (reward); when it goofs off, nothing (no reward). The puppy was never taught the correct answer, yet through repeated **try → reward** cycles it gradually discovers the behaviors that earn the most treats. The essence: "you don't know what's good until you try," plus the constant balancing act between repeating what already looks good (exploitation) and testing something new (exploration).

## 3. Core Concepts

### 3.1 The Three Paradigms at a Glance

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| What's given | Inputs + answer labels | Inputs only | Environment + reward signal |
| What's learned | Input→answer mapping | Hidden structure in the data | Actions that maximize reward |
| Analogy | Private tutoring | Self-directed study | Puppy training |
| Preparation cost | Labeling work | Interpretation work | An environment safe for trial and error |

### 3.2 A Map of Workplace Use Cases

- **Supervised**: churn prediction (customer→churned or not), revenue forecasting (ad spend→revenue), fraud classification, automatic document sorting, demand forecasting. **More than 80% of real-world ML projects live here.** In this lecture, levels 03–09 and 11 are all supervised learning.
- **Unsupervised**: customer segmentation (there's no ground truth for who belongs to which group), detecting novel anomaly patterns (fraud that never happened before has no labels), grouping documents by topic, compressing variables (dimensionality reduction). Covered in level10.
- **Reinforcement**: real-time ad-budget allocation, optimizing recommendation order, game and robot control — and, recently, the stage where conversational AI is tuned to human preferences. You will rarely implement it directly in a typical company, but you need the concept.

### 3.3 The Deciding Question Is "What Are You Given?"

The same business problem lands in different paradigms depending on the data situation. Take fraud detection: if past fraud cases are well labeled, it's supervised learning; if there are no labels but you want to find "transactions that sit far away from the normal crowd," it's unsupervised; if the loss/profit consequences of each block/allow decision come back as real-time feedback, a reinforcement-learning design is even possible. **It's not the type of problem but the shape of your data and feedback that decides the paradigm.**

### 3.4 The Two Branches of Supervised Learning

Supervised learning splits in two based on the shape of the answer. If the answer is a number (revenue, demand volume), it's **regression**; if the answer is a category (churn/stay, fraud/normal), it's **classification**. Level03 is regression; classification starts at level05.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

You experience each of the three paradigms through an ultra-mini example.

- **[1] Supervised — tutoring**: train a logistic regression on churn data while showing it the answer labels, then have it predict churn for 3 customers it has never seen. Confirm that "teach with the answer key, and it solves new problems."
- **[2] Unsupervised — self-directed study**: take the same kind of customer data, **deliberately remove the answer column**, and tell k-means only "group these into 3 clusters." When the machine prints each group's average profile, do the human part yourself — label them, e.g. "this group is the high-value customers."
- **[3] Reinforcement — puppy training**: not knowing which of 3 coupons (with different redemption odds) works best, an agent sends coupons out (tries) and learns from the responses (rewards) alone, gradually homing in on the best coupon (a multi-armed bandit with epsilon-greedy, implemented by hand). Watch how the average reward differs between the first 100 and the last 100 sends.
- **[4]** Summarize the three experiments in a table.

What to look for in the code: [1] calls `fit(X, y)` — **passing the answers y** — while [2] calls `fit(X)` — **no y at all**. The difference between two paradigms is visible in a single function signature. [3] is a pure-Python loop of about 20 lines with no library — the exploration rate `epsilon` governs how fast it learns.

## 5. Try It Yourself

1. **(Easy)** Set `EPSILON` in [3] to 0.0 and rerun. What happens with no exploration at all? Watch how a few unlucky early draws can trap the agent on a bad coupon. (Hint: try several different seeds.)
2. **(Medium)** Change the number of clusters in [2] from 3 to 2, then 5. How do the group profiles change? "How many groups is right?" having no correct answer is the defining trait of unsupervised learning.
3. **(Challenge)** In [1], shrink the training data from 2000 customers to 50. How unstable do the predictions become? Confirm that "the power of supervised learning = the amount of labeled data."

## 6. Common Mistakes

- **Believing "unsupervised learning needs no labels, so it's free"**: you trade labeling cost for interpretation cost. Judging whether the machine's groups mean anything for the business is still a human's job.
- **Proposing reinforcement learning everywhere**: in work where a single trial-and-error mistake is expensive (loan approval, medicine), "just try it and learn" is not allowed.
- **Mixing up regression and classification in conversation**: "predict the churn *rate* with regression" and "predict *whether* each customer churns with classification" are different problem designs. Check the shape of the answer first.
- **Choosing the paradigm first and bending the data to fit**: that's backwards. The data and the feedback you have decide the paradigm.

## Next Level Preview

Now that you know the paradigms, the genuinely hard part remains: translating a fuzzy business wish like "we want to reduce churn" into a precise problem a machine can solve. Level02 teaches that translation skill.
