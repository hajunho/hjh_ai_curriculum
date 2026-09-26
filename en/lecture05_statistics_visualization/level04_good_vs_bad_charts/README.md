# Lecture 05 · Level 04 — Good Charts vs Bad Charts

> Charts don't lie — but the people who draw them can.
**Difficulty** ⭐⭐ / **Prerequisites** level03 / **Estimated time** 45 min

## 1. Why learn this — the business view

Falsifying numbers gets you caught in an audit, but *staging* a chart breaks
no rule anywhere. That is why conference rooms are full of bar charts with
truncated axes, tilted 3D pie charts, and pictograms with exaggerated sizes.
The problem cuts both ways: you can be fooled by someone else's distorted
chart into a bad decision, and you can carelessly produce a distorted chart
yourself and lose people's trust.

This level has two goals. First, to build the distortion techniques with your
own hands and gain **an eye that sees through them**. Second, to draw honest
versions of the same data side by side and gain **a hand that draws
honestly**. We learn how to deceive not in order to deceive, but in order not
to be deceived.

## 2. Understand it with an analogy

A distorted chart is like a **retouched headshot**. The person in the photo
is real and nothing is composited, yet lighting, angle, and retouching can
create an impression quite different from real life. Truncating an axis is
shooting a tight close-up; a 3D effect is shooting from the angle that makes
the face look slimmer. Is the photo fake? No. But judge the person by that
photo alone and you will misjudge.

The standard for a good chart is simple: **does the first impression the
chart gives match the conclusion you reach by reading the data carefully?** A
chart engineered so that the first impression and the careful reading
disagree is a bad chart.

## 3. Core concepts

### 3.1 Crime #1 — the truncated axis

Start a bar chart's y-axis at some value other than 0 (say 90) and the
difference between 92 and 95 (really 3%) looks like a 2–3x difference. It
breaks the promise that a bar's **length** is proportional to its value.

- Rule: **a bar chart's y-axis starts at 0.** Bars speak through their length.
- Exception: line charts show the *shape* of change, so truncating can be
  acceptable — but you must clearly mark that you did (axis labels, a break
  symbol).

### 3.2 Crime #2 — the 3D-and-pie double act

Pie charts are already hard to read (comparing angles is hard), and tilting
one into 3D makes the **front slice look bigger than it is**. Perspective
inflates the same 25% when it sits in front and shrinks it at the back.
Putting your own market share in the front of a 3D pie is a classic staging
trick. With more than four slices, or with similar proportions, the right
answer is a sorted bar chart instead of a pie.

### 3.3 Crime #3 — pictogram area distortion

To show "revenue doubled", someone scales a coin drawing to twice the width
*and* twice the height — making its area four times bigger. The eye reads
area, so 2x gets exaggerated into 4x. When icon size encodes a value, use
counts instead (two coins) or scale only one dimension.

### 3.4 The ink ratio — data-ink

A practical criterion for good charts: "maximize the share of the chart's ink
that represents data." Excess gridlines, background gradients, drop shadows,
borders, and 3D effects are ink spent on decoration, not data. The more
decoration you erase, the better the data shows. A working trick: once the
chart is done, ask of every element "if I delete this, does information
disappear?" — if the answer is no, delete it.

### 3.5 A checklist for when someone hands you a chart

1. Does the bar chart's y-axis start at 0?
2. Are the axis tick intervals uniform? (No skipped stretches in the middle?)
3. Are there 3D effects? If so — why?
4. Do the things being compared share the same period, unit, and aggregation?
5. Is "n = how many people/cases" stated? It might be the average of a sample of 3.

### 3.6 Most distortion is not malice

Most distorted charts in the wild are produced not by intent to deceive but
by tool defaults. The spreadsheet auto-truncates the y-axis "for a better
look", and the chart wizard suggests 3D. So even the author does not know
their own chart exaggerates. When you draw with code, the axis range, colors,
and form are all explicit — leaving less room for unconscious distortion to
sneak in. That is one more reason analysts draw charts with code.

## 4. Practice — main.py

```bash
python3 main.py
```

From the `hjh_data` sales we take two stores' total revenue (about a 10%
gap) and draw distorted and honest charts side by side from the **same
data**.

- [1] First prints the two stores' actual values and the gap (%) as numbers.
- [2] `outputs/truncated_axis.png`: on the left, a chart whose truncated
  y-axis makes a gap of barely 10% look like an overwhelming lead; on the
  right, an honest chart starting at 0.
- [3] `outputs/pie_vs_bar.png`: on the left, a pie chart of five stores with
  similar shares (angles impossible to compare); on the right, a sorted
  horizontal bar chart (instantly comparable).
- [4] `outputs/inflated_line.png`: on the left, a line chart whose narrowed
  y-axis range turns tiny wiggles into a roller coaster; on the right, a
  chart with a range that provides context.

The key point: in all three figures, the left (distorted) and right (honest)
panels come from **exactly the same numbers**. Open the images, register your
first impression, then compare it against the real values from [1].

## 5. Try it yourself

1. **(Easy)** In [2], vary the y-axis starting value (0, 50%, 90%, 99%) and
   see how much the gap gets exaggerated.
2. **(Medium)** Pick one chart from a recent company report or news article
   and run it through the five checklist items in 3.5. How many does it pass?
3. **(Challenge)** Upgrade the "honest but boring chart" into an "honest and
   persuasive chart": annotate the honest chart from [2] with the gap amount
   as an arrow, and change the title into a conclusion sentence ("Store A
   leads by KRW 3.2M a month").
   (Hint: `ax.annotate` and `ax.set_title` are all you need. Honest and bland
   are not synonyms.)

## 6. Common mistakes

- **Forcing even line charts to start at 0**: for data where absolute zero is
  meaningless — exchange rates, stock prices — starting at 0 actually hides
  the change. Remember: "bars from 0; lines to fit the context, with the cut
  marked."
- **Attacking the person when pointing out distortion**: instead of "this
  chart is manipulated", try "how would it look if the y-axis started at 0?"
  — it lands far better in a meeting.
- **Treating all emphasis as sin**: color highlights, conclusion-style
  titles, and annotations are not distortion — they are good editing.
  Distortion makes the data say what it does not say; editing makes what the
  data says easier to hear.
- **Banning pie charts outright**: with 2–3 slices and clearly different
  proportions, a pie is fine. The problems are many-slice pies and 3D pies.

## Next level preview

You have learned chart distortion; next comes distortion of interpretation.
What more is needed before the observation "ad spend and revenue move
together" can become the conclusion "advertising raised revenue"? We take on
correlation versus causation.
