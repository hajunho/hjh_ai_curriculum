# Lecture 05 · Level 03 — Basic Charts with Matplotlib

> A chart is not a "picture" — it is a "sentence". The figure is the paper; the axes are the paragraph.
**Difficulty** ⭐⭐ / **Prerequisites** level02 / **Estimated time** 50 min

## 1. Why learn this — the business view

You can draw pictures with Excel charts, too. So why draw with code?
First, **repetition**: producing 30 weekly store-revenue charts in Excel
takes half a day; with code it is one run. Second, **reproducibility**: when
someone asks "that chart from last month's report — same thing, just change
the condition", code answers instantly. Third, **extension**: the machine
learning and A/B test results you will study later all end up being checked
with Matplotlib anyway.

Matplotlib is the standard visualization library in Python. Most other
visualization tools — seaborn, pandas' `.plot()` — use Matplotlib under the
hood. Understand its structure (figure/axes) properly once, and every other
tool's documentation becomes easy reading.

## 2. Understand it with an analogy

Matplotlib's structure works like **writing a report**.

- The **figure** is a sheet of paper. You choose its size (figsize) and,
  when finished, save it to a file (savefig).
- The **axes** is a paragraph on the paper (one chart panel). You can put a
  single paragraph on the page (`plt.subplots()`) or four in a 2×2 grid
  (`plt.subplots(2, 2)`).
- **plot/bar/scatter** are the sentences inside the paragraph: draw a line,
  raise bars, place dots.
- The **title, axis labels, and legend** are the paragraph's heading and
  footnotes. A chart without them is a sentence without a subject — nobody
  but its author can read it.

"Take out the paper (figure) → lay out paragraphs (axes) → write sentences
(plot) → add the heading (title/label) → submit (savefig)" — these five steps
are the skeleton of every Matplotlib script.

## 3. Core concepts

### 3.1 The skeleton code

```python
import matplotlib
matplotlib.use("Agg")            # save-to-file mode, no display window
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))   # paper + one paragraph
ax.plot(x, y)                              # draw the line
ax.set_title("Monthly Revenue Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (KRW millions)")
fig.savefig("outputs/monthly.png", dpi=120)
plt.close(fig)                             # put the paper away (free memory)
```

There is also a style that draws without an ax, like `plt.plot(...)`, but in
this course we always start with `fig, ax = plt.subplots()` — the
**object-oriented style**. The moment you have more than one chart, this
style is far less confusing.

### 3.2 The three chart siblings — which one, when

| Type | Function | Situation | Example |
|---|---|---|---|
| Line chart | `ax.plot` | **change over time** | monthly revenue trend |
| Bar chart | `ax.bar` | **comparing categories** | revenue by store |
| Scatter plot | `ax.scatter` | **relationship of two quantities** | ad spend vs revenue |

Use them the wrong way round and it immediately feels off. Draw a store
comparison as a line chart and you create the illusion of something
"transitioning from Downtown into Riverside"; draw a time trend as bars and
the trend gets hard to see. Just remember: **time = line, category = bar,
relationship = dots**.

### 3.3 Fonts and non-Latin text

Matplotlib's default font covers English perfectly, so the labels in this
course just work. But the default font has no glyphs for many non-Latin
scripts — put Korean, Japanese, or Chinese text in a title and it renders as
empty boxes (□□□, "tofu"). If your data ever contains such text, register an
installed font first:

```python
plt.rcParams["font.family"] = "AppleGothic"   # e.g. a Korean font on macOS
plt.rcParams["axes.unicode_minus"] = False    # keep the minus sign rendering
```

A robust pattern is to probe `matplotlib.font_manager` for a list of
candidate fonts and use the first one found — worth keeping in your toolbox
for international data.

### 3.4 Saving rules — Agg and outputs/

Servers and automation scripts have no monitor. `matplotlib.use("Agg")`
declares "don't open a window, just produce image files", and it must be
called **before importing pyplot**. Save with `fig.savefig(path, dpi=120)`;
in this course we always save under the level folder's `outputs/` and print
the path. `os.makedirs(OUT_DIR, exist_ok=True)` — create the folder if
missing, skip if present — is also a standard pattern.

### 3.5 Color is information

Use color for "what you are saying", not for "making it pretty". Paint every
bar a different rainbow color and readers exhaust themselves hunting for a
meaning in each hue. The default: one calm color for everything, and a
different color only for the one bar you want to emphasize — this alone
dramatically improves how well a chart delivers its message. The bar chart in
exercise [3] implements exactly this principle.

## 4. Practice — main.py

```bash
python3 main.py
```

We draw three charts from the `hjh_data.sales_table()` revenue.

- [1] Clean the data and aggregate by month and by store (a lecture03 review).
- [2] `outputs/line_monthly.png`: monthly total revenue as a **line chart**,
  with markers and value labels to make the trend easy to read.
- [3] `outputs/bar_stores.png`: total revenue per store as a **bar chart**,
  sorted descending, with only the top store highlighted in a different color.
- [4] `outputs/scatter_ad.png`: daily ad spend vs revenue as a **scatter
  plot**. The relationship between the two shows up as the slope of the dot
  cloud (a preview of level05).

Each function follows the five steps from 3.1 (paper → paragraph → sentence →
heading → save) exactly — pick one function and map it line by line onto the
skeleton.

## 5. Try it yourself

1. **(Easy)** Change the line chart's color (`color="#d1495b"`) and its line
   style to dashed (`ls="--"`), then save.
2. **(Medium)** Convert the bar chart in [3] to horizontal bars (`ax.barh`).
   See why horizontal bars read better when category names are long.
   (Hint: you must swap what goes into x and y.)
3. **(Challenge)** Use `plt.subplots(1, 2, figsize=(12, 4))` to place the
   line chart and the bar chart side by side on one sheet and save it as
   `outputs/dashboard.png`.
   (Hint: `fig, (ax1, ax2) = plt.subplots(1, 2, ...)` gives you two
   paragraphs to draw into.)

## 6. Common mistakes

- **Calling `matplotlib.use("Agg")` after importing pyplot**: you may get a
  warning, or it may simply not apply. Always put it on the first line right
  after `import matplotlib`.
- **Skipping `plt.close(fig)`**: in a loop that draws dozens of charts,
  memory keeps piling up. Saved it? Close it.
- **Omitting axis labels and units**: "Revenue" vs "Revenue (KRW 10,000s)"
  changes the reading of a chart a hundredfold. Always put the unit in the
  axis label.
- **Using a line chart for category comparison**: there is no "in between"
  between two stores. Connecting them with a line implies a continuity that
  does not exist.
- **Shipping charts that depend on fonts not installed everywhere**: it looks
  fine on your machine and breaks on a colleague's. If your labels need a
  special font, include font-detection code in the script.

## Next level preview

Now that you know how to draw, it is time to learn how to draw *badly*. We
will build charts that deceive — truncated axes, 3D effects — with our own
hands, and put them side by side with honest versions of the same data.
