# Lecture 02 · Level 10 — Comprehensions, Generators, Lambda, and a Taste of Decorators

> Comprehensions write "pick the matching items from a list, transform them, and make a new list" in one line; generators stream data one item at a time instead of building it all at once — a tasting menu of the four signature expressions of Pythonic code.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level09 / **Estimated time** 60 min

## 1. Why learn this — the business angle

With the syntax learned so far you can already write any program. But open
someone else's code at work — a colleague's script, an open-source example,
code an AI generated for you — and one-liners like
`[row["amount"] for row in rows if row["ok"]]` come pouring out. If you don't
know these expressions you can't read them, and what you can't read you can't
fix.

There is a second problem: what happens when data gets big. Load a
100-million-row transaction log into a list, all in memory, and the machine
grinds to a halt. A **generator** solves this by streaming "one item, on
demand" — the doorway to the real-world technique of large-stream
processing. This level's goal is not memorizing syntax but gaining **a
reading eye and a feel for memory**.

## 2. Understanding through analogies

**A comprehension = Excel filter + formula in one stroke.** "From the sales
sheet, keep only the Downtown rows and make a new sheet with the amounts
multiplied by 1.1." In Excel that is two actions — apply a filter, drag a
formula — but in speech it is one sentence. A comprehension moves that one
sentence straight into code:
`[x * 1.1 for x in sales if x.store == "Downtown"]` — "the list of each
Downtown x multiplied by 1.1."

**A generator = a tap.** A list is like bottling water into 100 million
bottles in advance and stacking them in a warehouse (memory). A generator is
a tap — each turn yields one cupful, and no warehouse is needed. Think of
month-end reconciliation done not by "gather everything, then compute" but by
"accumulate each item as it flows in."

**A lambda = a sticky-note instruction.** A one-line instruction that doesn't
deserve a formal procedure document (a def function) — "sort these by
amount" — handed over on a sticky note: that is lambda.

**A decorator = an approval stamp.** Without touching the document's (the
function's) contents, you stamp "Reviewed" on the outside, layering on a
shared procedure (timing, logging). That device is a decorator.

## 3. Core concepts

### 3.1 List comprehensions

```python
# Loop version (4 lines)
result = []
for x in sales:
    if x >= 100000:
        result.append(int(x * 1.1))

# Comprehension (1 line) — meaning: "the list of each x >= 100k multiplied by 1.1"
result = [int(x * 1.1) for x in sales if x >= 100000]
```

Read it **starting from the for**: "for each x in sales → if the condition
holds → evaluate the leading expression and collect it." The dictionary
version `{k: v for ...}` and set version `{x for ...}` work on the same
principle. But once you have more than two conditions or a complicated
expression, **spelling it out as a loop is better.** A cryptogram crammed
into one line is not polished code.

### 3.2 Generators — one item, on demand

```python
def transaction_stream(n):
    for i in range(n):
        yield make_row(i)        # yield instead of return: emit one item, pause, wait

for row in transaction_stream(100_000_000):   # even 100M rows use one row's worth of memory
    process(row)
```

A function containing `yield` does not run when called — it returns a "tap"
(a generator object). Each lap of the for runs only as far as the next yield,
emits one value, and pauses. Because nothing is loaded whole into memory,
this is the standard technique for large-scale processing. Swap a
comprehension's square brackets for parentheses —
`(x * 1.1 for x in sales)` — and you get a **generator expression**, commonly
fed straight into `sum(...)`. One caution: like water from a tap, **once
consumed it is gone.**

### 3.3 Lambda — a nameless one-line function

```python
sorted(rows, key=lambda r: r["amount"], reverse=True)   # descending by amount
max(rows, key=lambda r: r["amount"])                     # the highest-revenue row
```

The form is `lambda args: return-expression`, and its main stage is passing
"by what criterion?" to the `key` argument of `sorted`/`max`/`min`. If you
need two or more lines, just make a named function with `def`.

### 3.4 Decorators — functions that wrap functions (a taste)

```python
@stopwatch                 # shorthand for: report = stopwatch(report)
def report():
    ...
```

A decorator is "a function that takes a function and returns a new function
with a capability layered on." Use it for **the stamp you press onto many
functions alike** — timing, logging, permission checks. You won't write your
own for a while, but being able to *read* the `@` notation is required for
real-world code and framework docs.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level10_comprehensions_generators
python3 main.py
```

Using store-sales data transformations, the four expressions are demonstrated
in turn.

- **[1] Loop vs comprehension**: writes the same transformation (apply VAT to transactions of 100k KRW or more) both ways and confirms identical results. Also builds a "store -> revenue" table with a dictionary comprehension.
- **[2] Generators**: **measures and compares actual memory usage** when 1,000,000 transactions are stacked in a list versus streamed through a generator (using the standard library's `sys.getsizeof`). Roughly 8MB vs 200 bytes — the numbers speak louder than a hundred sentences.
- **[3] Lambda**: sorts the top-3 stores and finds the best/worst store with `key=lambda`.
- **[4] A taste of decorators**: stamps the timing decorator `@stopwatch` onto two functions and confirms a shared capability is layered on without editing the function bodies.

## 5. Try it yourself

1. **Translation drill** — read
   `[r["store"] for r in rows if r["amount"] >= 500000]` out loud in plain
   English, then rewrite it as the equivalent 4-line loop. (Hint: "read from
   the for" — for each r, if the condition holds, collect the store.)
2. **Generator-expression sum** — from the stream in [2], compute "the total
   of transactions of 500k KRW or more" in a single
   `sum( ... for ... if ... )` line, without building a list. (Hint: written
   with parentheses (omittable) instead of brackets, no intermediate list is
   created.)
3. **One more stamp** — model a `@call_log` decorator on `stopwatch` that
   prints the fact that a function was called. (Hint: in the inner function,
   `print` then `return func()`.)

## 6. Common mistakes

- **Comprehension overreach** — a one-liner with 3 conditions and 2 levels of
  nesting is unreadable to everyone. The bar is not "short in one line" but
  "reads as one sentence."
- **Reusing a generator** — run a consumed generator again and you get an
  empty result. If the data is needed twice, collect it into a list or make
  the generator again.
- **Trying len()** — a generator does not know its length in advance;
  `len(gen)` is a TypeError. Need a count? Count while streaming.
- **Stuffing logic into a lambda** — a long lambda with conditionals should
  be promoted to def. A lambda is a "sticky note"; when you need a procedure
  document, write a formal one (def).
- **Fearing comprehension variable leaks** — the loop variable `x` does not
  leak out of a comprehension (in Python 3). It is separate from any
  same-named variable outside.

## Next level preview

The final stop of the syntax journey is turning "code that runs" into "code
you can trust." In level11 we make functions' contracts explicit with type
hints and verify code with assert-based mini tests — the quality-control
habits.
