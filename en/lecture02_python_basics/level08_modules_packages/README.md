# Lecture 02 · Level 08 — Modules and Packages

> A module is a single Python file holding functions and variables, and import is the call that brings another department's specialist into your work. Python ships with the standard library — a full set of in-house specialist departments.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level07 / **Estimated time** 50 min

## 1. Why learn this — the business angle

Once code passes 300 lines, you start getting lost in a single file. Just as
a growing company cannot have one person doing admin, accounting, and sales —
so it forms departments — a program must be split into files by function.
That unit is a **module**, and modules grouped into a folder form a
**package**.

The other practical payoff is simply using "specialist departments that
already exist." Date arithmetic, math functions, random numbers, tallying
tools — you never need to build these yourself. The **standard library**,
included with every Python installation, provides hundreds of modules, and a
single `import` line is all it takes. The developer's proverb "don't reinvent
the wheel" translates at work to "don't manufacture bugs in your own code
when a proven library exists."

## 2. Understanding through analogies

**The department-call analogy.** You are drafting a proposal and need a legal
review. You don't teach yourself law — you call the legal team.

> `import legal` → "connect me to the legal team"
> `legal.review(contract)` → "hand this contract to the legal team's review person"

The `module.function()` notation is "department.person()". Because the
affiliation is explicit, you never confuse who is from where. For someone you
call often, you can get a direct line: `from legal import review` → now you
call `review(contract)` directly. Convenient — but if you bring in
same-named people from several departments (say, sales' review and legal's
review), you risk mixing them up.

A module we write ourselves is "our team's procedures binder." Collect the
frequently used procedures (functions) in the binder (utils.py) and any
document (main.py) can pull them out.

## 3. Core concepts

### 3.1 The three forms of import

```python
import math                      # the whole department: math.sqrt(2)
from math import sqrt            # a direct line to one person: sqrt(2)
from math import sqrt as rt      # with a nickname: rt(2)
```

The default recommendation is the first form. Seeing the origin, as in
`math.sqrt`, makes code review easy. `from module import *` (summon
everyone) is a breeding ground for name collisions — treat it as forbidden.

### 3.2 Making your own module — a file *is* a module

Create `utils.py` in the same folder, define functions in it, and in
`main.py` call `import utils` and use it like `utils.format_krw(1200000)`.
The rules are simple.

- The file name is the module name (`utils.py` → `import utils`; drop `.py`).
- Python searches for modules along a fixed list of paths (`sys.path`),
  including the folder the running script lives in.
- **The moment a file is imported, its code runs top to bottom, once.** So a
  module should contain only function and variable definitions, with any
  executable code under `if __name__ == "__main__":`. That idiom's true
  identity is revealed here — "act only when run directly; when imported,
  provide definitions only." Now you know why every main.py so far has used
  it.

### 3.3 Packages — bundling modules into folders

When modules multiply, group them in a folder. Put an `__init__.py` file in
the folder and Python recognizes it as a package, importable by path like
`from mypkg import utils`. The numpy and pandas you install with pip are, in
the end, just packages someone else built and published (installation and
virtual environments were covered in lecture01 level04).

### 3.4 Standard library tour — today's four departments

| Module | In charge of | Signature features |
|---|---|---|
| `datetime` | dates & times | today's date, date differences, countdowns, formatting |
| `math` | mathematics | square roots, ceil/floor, logarithms, pi |
| `random` | random numbers | draws, sampling, shuffles (seed for reproducibility) |
| `collections` | data-structure extensions | Counter (frequency tallies), defaultdict (default-value drawers) |

`collections.Counter` in particular finishes "count per category" in one
line and shows up constantly in real work. Beyond these, `csv` (level06),
`os`, `json`, `statistics` — the important habit is "when you need a
feature, search the standard library first."

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level08_modules_packages
python3 main.py
```

This level's folder contains two files: **utils.py** — our team's shared
module with currency formatting, VAT calculation, and business-day functions
— and **main.py**, which imports and uses it.

- **[1] Importing my module**: pulls shared functions with `import utils` and builds a sales summary. Observe that the print at the top of `utils.py` runs exactly once at import, and that utils' self-demo runs only when executed directly (the `__name__` distinction).
- **[2] datetime**: demos today's date (a fixed reference date), the countdown to the project deadline, and date formatting.
- **[3] math**: shows why budget allocation needs rounding up (`ceil`) — seating 17 people in 5-seat vans.
- **[4] random**: with a fixed seed, runs a prize draw (`sample`) and a duty-roster shuffle (`shuffle`).
- **[5] collections**: tallies order-category frequencies in one line with `Counter` and collects per-store sales with `defaultdict`.

First run the module **directly** with `python3 utils.py` to see the
self-demo, then run `python3 main.py` — the **importing side** — and compare
the two cases.

## 5. Try it yourself

1. **Add a shared function** — add a discount function
   `apply_discount(price, rate)` to utils.py and call it from main.py.
   (Hint: you can move over the function you built in level05. "Functions
   used by several files go into utils" is the professional instinct.)
2. **Compare direct imports** — add `from utils import format_krw` at the top
   of main.py and call it without the `utils.` prefix. What pros and cons do
   you notice? (Hint: shorter, but the origin is hidden.)
3. **Visit the statistics department** — after `import statistics`, compute
   the mean and median of the sales list in [1] with `statistics.mean` and
   `statistics.median`. (Hint: this is practice at looking up standard
   library docs. `help(statistics)` works too.)

## 6. Common mistakes

- **Naming your file after a module** — name your file `random.py` and Python
  imports *your* file instead of the standard library, producing baffling
  errors. Avoid standard module names.
- **Scattered imports** — convention is to gather imports at the top of the
  file. Sprinkled through the code, dependencies become hard to trace.
- **`from ... import *`** — you can no longer tell where names came from, and
  collisions are frequent.
- **Executable code loose in a module** — leave running code outside
  `if __name__ == "__main__":` and it fires on every import. Separate
  definition from execution.
- **Not fixing the random seed** — in work that needs reproducible tests and
  reports, an unfixed seed gives "worked yesterday, different today" results.
  Experimental reproducibility is a fundamental for all the AI exercises
  ahead.

## Next level preview

With functions and modules organizing our procedures, it is time for the tool
that **binds data and procedure into one body**. In level09 we build an
Employee → Manager inheritance structure with classes and object-oriented
programming, implementing the "cookie cutter and cookies" relationship in
code.
