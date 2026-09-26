# Lecture 05 · Level 00 — Summarizing Reality with Numbers

> Statistics is the craft of "summarizing" and "inferring". A summary is convenient — but it always loses something.
**Difficulty** ⭐ / **Prerequisites** lecture03 completed / **Estimated time** 30 min

## 1. Why learn this — the business view

Almost every number that circulates in a company is a summary. "Average basket
size this month: KRW 18,400", "customer satisfaction 4.2 out of 5", "daily
revenue per store: KRW 3.2 million". The raw data has tens of thousands of
rows, but the report compresses it into a single line. The problem is that
information disappears during this compression. The same "average of 4.2" can
describe a shop where nearly everyone gives 4s — or a shop where half give 5s
and half give 1s. The latter may be going seriously wrong right now.

There are two reasons to learn statistics. First, to **summarize** data and
communicate it to others. Second, to look at only part of the data (a sample)
and make **inference** about the whole. The first half of this lecture
(level00–04) covers summarizing; the second half (level06–11) covers
inference. This level is the starting point: you will see with your own eyes
*what a summary loses*.

## 2. Understand it with an analogy

A summary is like a **résumé**. A résumé compresses ten years of a person's
life onto one page. Thanks to résumés, a recruiter can compare a hundred
candidates quickly — but hiring on the résumé alone sometimes ends in regret.
The things not written on the page — how the person collaborates, how they
handle a crisis — turned out to matter more.

The mean is a résumé for your data. It is convenient, but if you never ask
"what data sits behind this average?", you end up making a bad hire — that is,
a bad decision. That is why the first habit of people who are good at
statistics is not calculation but **suspicion**: "How well does this summary
actually represent the raw data?"

One more analogy: a map is a summary of reality. A subway map distorts real
distances and directions, yet it is the best possible map for changing trains.
In other words, **a good summary chooses what to lose, on purpose, to fit its
purpose**. As long as you know what was lost, a summary is a powerful tool.

## 3. Core concepts

### 3.1 Descriptive vs inferential statistics

- **Descriptive statistics**: summarizes and describes the data you have.
  Mean, median, variance, and histograms belong here.
- **Inferential statistics**: looks at a sample and guesses about the whole
  (the population). Opinion polls, A/B tests, and confidence intervals belong here.

Excel's AVERAGE is descriptive statistics; "can I speak for all our customers
based on these 500 survey responses?" is a question of inferential statistics.

### 3.2 Same mean, different realities

This is the key experiment of this level. We build four datasets whose means
are all exactly 100.

| Dataset | Shape | Example business situation |
|---|---|---|
| A | Clustered evenly around 100 | Steady revenue from regulars |
| B | Two clusters, around 60 and around 140 | Weekday and weekend customers are completely different |
| C | Mostly around 80, a few extremes | Quiet most days, then one big contract |
| D | Spread widely between 40 and 160 | Revenue is hit-or-miss |

By the mean alone, the four shops are the same shop. But the right strategy
for inventory, staffing, and cash-flow management is different for each. A
single summary (the mean) shows none of that.

### 3.3 Three questions for interrogating any summary

1. **Spread**: are the values huddled around the mean, or scattered widely? (→ level01, variance)
2. **Shape**: is it one lump, several clusters, or skewed to one side? (→ level02, distributions)
3. **Outliers**: is a handful of values dragging the summary around? (→ level01, median)

These three questions are also the table of contents for the first half of
this lecture.

### 3.4 When summaries are gamed on purpose

Summaries are not distorted only by accident. They are distorted deliberately,
too. Behind a job posting that says "average salary of KRW 50 million" may
hide a few executives' enormous pay packages; behind an ad claiming "90% of
customers are satisfied" may hide a survey that hand-picked its respondents.
Learning statistics is both a skill for making numbers and a self-defense
skill against numbers other people made. Throughout this lecture you will
build the habit of asking, "How was this summary produced?"

## 4. Practice — main.py

```bash
python3 main.py
```

The code builds four synthetic revenue datasets (A–D) whose means all equal 100, then

- [1] computes each mean to confirm they are "identical on paper",
- [2] lines up the min, max, and median to expose the differences hiding behind the summary,
- [3] draws each dataset's shape with a text histogram (asterisk chart), and
- [4] finally shows that judgments like "which shop needs more evening staff?"
  are impossible to make from summary values alone.

The heart of the code is the `ascii_hist()` function. It splits values into
bins, counts them, and prints one `*` per unit of frequency. You have not met
matplotlib yet, but the whole principle of a histogram — split into bins,
count frequencies — is already here. In the output, compare how differently
shaped A–D are, and ask yourself why their means are nevertheless identical.

## 5. Try it yourself

1. **(Easy)** Inside `make_datasets()`, change dataset C's extreme value 1200
   to 3000 and rerun. What would the remaining values have to do to bring the
   mean back to 100? (Hint: the code does not re-correct, so the mean drifts.
   The point is to watch how much influence one extreme value has.)
2. **(Medium)** Put ten real numbers from your own work or life into a list
   (e.g., your last 10 days of step counts or spending) and run `summarize()`
   and `ascii_hist()` on them. If the mean and the median differ, explain why.
3. **(Challenge)** Design and add a fifth dataset E where "the mean is the
   same but the median is very different". (Hint: many small values plus a few
   large ones keeps the mean up while pushing the median down.)

## 6. Common mistakes

- **Concluding from the mean alone**: "the average went up, so things are
  better" is half the truth. One extreme value may have lifted it.
- **Dumping raw data with no summary**: the opposite extreme is also a
  problem. Nobody reads a report with tens of thousands of rows. Summarize —
  but say what the summary lost.
- **"Statistics is hard", so relying on gut feel**: the only tools in this
  level were addition, division, and printing asterisks. Half of statistics is
  this kind of diligent looking.

## Next level preview

Next we formally meet the star summary statistics: mean, median, variance, and
standard deviation. With the "distorted average salary" story, you will
experiment with just how far a single extreme value can drag a mean.
