# Lecture 05 · Level 02 — Distributions and Histograms

> If summary statistics are your data's résumé, the distribution is its face. Meet the face and there is no misunderstanding.
**Difficulty** ⭐⭐ / **Prerequisites** level01 / **Estimated time** 45 min

## 1. Why learn this — the business view

You have learned the mean and the median, but a handful of numbers still
misses things. Take customer wait-time data: mean 4 minutes, median 3
minutes. Sounds fine. But draw the distribution and you find that while most
customers wait 2–3 minutes, a dozen customers every day wait more than 20.
Complaints and churn come from exactly that **tail**. Improving the mean and
shrinking the tail are two completely different projects.

The most basic tool for drawing a distribution is the **histogram**: split
the range of values into intervals (bins) and draw each bin's frequency as a
bar. If you have ever done a COUNTIF per bucket in an Excel pivot, you have
already built a histogram by hand. In this level you draw one in a single
line of Python — and learn the trap that "changing the bin size changes the
story".

## 2. Understand it with an analogy

A histogram is a **night photo of an apartment complex**. It shows which
building (bin) each household (value) lives in. At a glance you can see which
buildings have the most lights on (high frequency), whether the complex is
one cluster or split in two (number of peaks), and whether there is a lone
lit window far from everything else (an outlier).

Bin size is the **scale of a map**. On a national map (large bins) you can
only see "most people live near the capital"; on a neighborhood map (small
bins) you see every alley but lose the overall pattern. There is no such
thing as a wrong scale — only a **scale that doesn't fit the purpose**.
Histograms are the same: depending on how many bins you choose, the same data
can look bland or bumpy.

## 3. Core concepts

### 3.1 How to read a histogram

Four things to look for:

1. **Center**: where is the peak? (connects to summary statistics)
2. **Spread**: how wide is it? (connects to the standard deviation)
3. **Shape**: symmetric, long-tailed to one side, how many peaks?
4. **Oddities**: a bar far away from the main body (outlier), a strangely empty gap

Two peaks (a bimodal distribution) is a strong signal that "different groups
are mixed together". Split the groups — weekday/weekend, new/returning — and
draw again.

### 3.2 Skewness — which way the tail points

Skewness measures how lopsided a distribution is.

- **Right tail (positive skew)**: revenue, salaries, wait times. Mostly small
  values plus occasional large ones. Here mean > median.
- **Left tail (negative skew)**: like exam scores bunched near a perfect
  score — many large values and few small ones. Here mean < median.

Just comparing the mean and the median hints at the tail's direction. The
"mean > median" phenomenon you saw in level01 was exactly a right tail.

### 3.3 The effect of bin size

| Number of bins | What you see | Risk |
|---|---|---|
| Too few (5) | Only the broad flow | Two peaks smeared into one |
| About right (20–40) | Center, spread, shape | — |
| Too many (200) | Every little bump | Mistaking random noise for a pattern |

There is no formula for the right answer, but the customary starting point is
"about √n bins", or drawing a few versions between 20 and 50 and finding
where the story stabilizes. **Only trust patterns that survive a change of
bins** — that is the working rule.

### 3.4 Histogram vs bar chart

They look alike but differ. A histogram shows **a continuous quantity cut
into intervals**, so the bars touch and their order matters. A bar chart
compares **categories** — stores, product lines — so the bars are separated
and can be reordered freely. Confusing the two in a report makes readers
misread the axis.

### 3.5 A distribution in one number — a taste of percentiles

Percentiles are helper numbers that summarize a histogram. "The 90th
percentile is 8 minutes" means "90% of customers are served within 8
minutes." This is why IT services set response-time targets at p95 or p99
rather than the mean — it is all about managing the tail. Exercise [1]
prints the revenue's p50 (median) and p95 together; feel the length of the
tail in the gap between those two numbers.

## 4. Practice — main.py

```bash
python3 main.py
```

Using the revenue from `hjh_data.sales_table()`, the script saves three
histogram PNGs into the `outputs/` folder. (You will formally learn the
chart-saving code in the next level — for now, focus on the resulting
pictures.)

- [1] After cleaning, prints the revenue distribution's mean, median, and skewness.
- [2] `outputs/hist_bins.png`: the same data drawn with 5 / 30 / 200 bins,
  side by side. Compare how the bins change the story (number of peaks,
  smoothness).
- [3] `outputs/hist_skew.png`: the right-tailed revenue distribution with
  vertical lines for the mean and median, showing the tail dragging the mean.
- [4] `outputs/hist_bimodal.png`: weekday and weekend revenue overlaid. The
  blunt-looking overall distribution turns out to be the sum of two groups.

After running, open the three PNGs and match them against the printed
commentary. In the code, the one line that matters is
`plt.hist(..., bins=...)` — the rest is packaging: titles, axis labels,
saving.

## 5. Try it yourself

1. **(Easy)** Change the bin counts in [2] from 5/30/200 to another
   combination (say 10/50/500) and save again. At what count does "noise"
   start to appear?
2. **(Medium)** Pick a single category (say Coffee) and draw its histogram.
   How does its shape differ from the overall distribution?
   (Hint: `df[df["category"] == "Coffee"]["revenue"]`)
3. **(Challenge)** Draw a histogram of log-transformed revenue
   (`np.log10`). Confirm that the right-tailed distribution becomes nearly
   symmetric on a log axis, and summarize in one line why log transforms are
   so common when analyzing monetary data.
   (Hint: values that grow multiplicatively become additive in log world.)

## 6. Common mistakes

- **Concluding from a single bin setting**: important judgments like the
  number of peaks must be checked across 2–3 bin settings.
- **Interpreting a mixed distribution as one**: don't dismiss two peaks as "a
  weird distribution" — split the groups and redraw. There is almost always a
  hidden axis: weekday/weekend, store, product line.
- **Not saying whether the y-axis is frequency or proportion**: when
  comparing two distributions with different sample sizes, use
  `density=True` (proportions) to be fair.
- **Hiding that you clipped outliers before drawing**: limiting the range for
  readability is fine, but always leave a note like "top 1% excluded" on the
  figure.

## Next level preview

So far you have been *given* the pictures. In the next level you formally
learn Matplotlib's figure/axes structure and draw line charts, bar charts,
and scatter plots yourself.
