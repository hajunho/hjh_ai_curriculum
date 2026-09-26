# Lecture 03 · Level 02 — NumPy Array Basics

> Measure for yourself the speed of vectorization — computing a million numbers at once with no loops — and learn NumPy's grammar of shape, dtype, and broadcasting on sales data.

**Difficulty** ⭐⭐ / **Prerequisites** level01 / **Estimated time** 50 min

## 1. Why learn this — the business view

The loop-based code from level01 is instant on 1,500 rows, but it slows down noticeably once data reaches millions of rows. Transaction logs, sensor readings, and ad-click data start at millions of rows. The difference between "code you kick off and go get a coffee" and "code that answers immediately" determines how many analyses you actually run at work. When code runs fast, you can ask more questions.

NumPy is the standard library for handling large quantities of numbers quickly in Python. And everything after it — the Pandas you learn from the next level, and later scikit-learn and PyTorch — runs on top of NumPy arrays. In other words, NumPy's way of thinking — **make a bundle of numbers (an array), not a single number, the unit of computation** — is the shared grammar of the entire data field. Get the feel for it here and every later tool will look like the same principle.

## 2. Understanding through an analogy

Imagine ringing up 1,000 cans of soda at a convenience store.

- **A Python loop** = the part-timer scanning each can's barcode one at a time. Each scan is quick, but the "pick up → scan → put down" prep motion repeats 1,000 times.
- **A NumPy vector operation** = the warehouse that weighs the whole box on a scale in one go. The "prep motion" happens exactly once, and the actual computation is handled by machinery (optimized C code).

The real reason Python loops are slow is not the arithmetic itself but the **prep motion repeated every iteration** (type checks, unboxing objects, and so on). NumPy has promised in advance that "everything in this box is a number of the same type", so it skips the prep and races across contiguous memory at machine speed.

Broadcasting is **using a rubber stamp**. Instead of repeating the instruction "add 10% VAT to this sale" 1,000 times, you press the single stamp `revenue * 1.1` onto the whole array at once.

## 3. Core concepts

### 3-1. ndarray — a contiguous bundle of same-type numbers

NumPy's core data structure is the multidimensional array (ndarray). Two decisive differences from a Python list:

1. **Every element has the same type.** This promise is what enables high-speed computation without type checks.
2. **Elements sit contiguously in memory.** The CPU can prefetch predictively, which makes it even faster.

```python
import numpy as np
a = np.array([1, 2, 3])
a * 10        # array([10, 20, 30]) — applied to everything, no loop
```

### 3-2. shape and dtype — the array's ID card

- **shape**: the array's form. `(1500,)` is one-dimensional with 1,500 elements; `(30, 5)` is 30 rows by 5 columns. Most errors come from shape mismatches, so make it a habit to check `shape` first whenever you receive an array.
- **dtype**: the element type — `int64` (integer), `float64` (floating point), and so on. The missing-value marker `NaN` (Not a Number) can only exist in floats, so you will often see an integer column with missing values silently turn into float.

### 3-3. Vectorization — replacing loops with operators

Any task of the form "do the same thing to each element" is written as an array operation instead of a loop.

| Loop thinking | Vectorized thinking |
|---|---|
| `for r in revenues: total += r` | `revenues.sum()` |
| `for r in ...: new.append(r * 1.1)` | `revenues * 1.1` |
| `for r in ...: if r > 1,000,000: ...` | `revenues[revenues > 1_000_000]` |

The last line is a boolean mask. `revenues > 1_000_000` produces a True/False array, and putting it inside square brackets selects only the values at True positions. This is the skeleton of Pandas filtering in level05, so get it into your fingers here.

### 3-4. Broadcasting — computing across mismatched sizes

When you combine an array with a single number (a scalar), as in `array * 1.1`, NumPy "stretches" the number to the array's size for you. No memory is actually copied — the stretch is conceptual, so it costs nothing. Adding a `(5,)` array to a `(30, 5)` array adds it to every row; whenever dimensions differ but the rules line up, expansion is automatic.

### 3-5. Aggregation functions

`sum()`, `mean()`, `std()` (standard deviation), `min()/max()`, `round()` and friends are available as array methods. On a two-dimensional array, `axis=0` (down the columns) / `axis=1` (across the rows) choose the direction of aggregation — a foundation stone for understanding groupby in level07.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level02_numpy_basics
python3 main.py
```

- **[1] Speed duel**: computes the "sum of squares" of one million random numbers (a) with a Python for-loop and (b) with a NumPy vector operation, timing both with `time.perf_counter`. Along with the multi-fold speed gap, it also verifies the two answers agree.
- **[2] Sales data as an array**: builds an array from the `revenue` values of 90 days of `sales_table` (excluding None) and checks `shape`/`dtype`.
- **[3] One-line aggregation**: computes the sum, mean, standard deviation, and maximum without loops.
- **[4] Broadcasting**: applies 10% VAT to everything with a single `* 1.1`, and handles rounding to the nearest thousand as a vector too.
- **[5] Boolean masks**: builds a weekend mask to compare weekend vs weekday averages, and counts negative contamination via the mask's `sum()`.

In the output, note the speed multiple (tens to hundreds of times, depending on your machine) and the idiom in [5]: "counting Trues = summing the mask".

## 5. Try it yourself

1. **(Easy)** Adapt [4] to the "cut ad spend by 20%" scenario: build the array `ad_cost * 0.8` and find the new average ad cost.
2. **(Medium)** Using a mask, find what percentage of days exceed "mean + 1 standard deviation" of revenue. Hint: `(rev > rev.mean() + rev.std()).mean()` gives the True ratio directly.
3. **(Challenge)** Re-run the speed experiment in [1] with 100 thousand, 1 million, and 10 million numbers and watch how the multiple changes. Hint: the bigger the data, the larger the share of "prep motion", so the gap tends to widen.

## 6. Common mistakes

- **Confusing `*` on lists and arrays**: `[1, 2] * 3` is list repetition (`[1,2,1,2,1,2]`), while `np.array([1, 2]) * 3` is element-wise multiplication (`[3, 6]`). Completely different operations.
- **Converting a list containing None straight into an array**: the dtype becomes `object` and the speed advantage of vectorization vanishes. Remove missing values before building the array, or standardize them to `NaN`.
- **Iterating over an array with a for-loop**: it works syntactically, but it defeats the point of NumPy. Make it a rule: "when you feel like writing a loop, first look for a vector operation".
- **Mixing up the `axis` direction**: on 2-D data, `sum(axis=0)` "squashes the rows" and produces per-column totals. Always check the shape of the result.

## Next level preview

NumPy arrays are fast, but they don't know "what this column is called". In the next level we learn the **Pandas DataFrame — an array dressed in row labels (index) and column names (columns)** — and start handling the sales table by name.
