# Lecture 02 · Level 11 — Type Hints, Testing, and Code Quality

> Type hints are the contract stating a function's input/output specs; tests are the quality-inspection checklist that automatically verifies "this code behaves like this in this case."
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level10 / **Estimated time** 60 min

## 1. Why learn this — the business angle

"It ran once on my machine" and "you can trust it" are different stories.
Production code gets handed to someone else, gets fixed by you six months
later, and must keep running after the data changes. What you need then are
this level's three things.

- **Type hints**: state in the code what a function takes and returns. Unlike
  comments, tools (mypy and friends) catch spec violations automatically.
- **Tests**: every time you change something, automatically confirm "did I
  break the existing behavior?" With tests, changing code stops being scary —
  that is the real payoff.
- **Docstrings and conventions**: the documentation habits that make code
  explain itself.

This level is something like lecture02's final exam. Whether these habits are
attached to the data-processing functions you build in the AI exercises ahead
is what separates "hobby code" from "work code."

## 2. Understanding through analogies

**The contract analogy.** When you outsource work, the contract states the
specs. "Your company delivers the transaction list (each item an integer
amount); our company delivers the summary report (a string)." Type hints are
exactly this.

```python
def summarize(amounts: list[int]) -> str:
```

The contract — "amounts arrives as a list of integers, and a string comes
back" — is engraved into the function signature. Without a contract? The
other side sends a list of strings and nobody notices until delivery day
(runtime), when the accident happens. With a contract, the inspector (mypy)
flags "spec mismatch" **before the work even starts**.

**The quality-checklist analogy.** A factory runs a checklist before shipping
a product. "Normal input gives normal output? Empty input handled safely? No
misbehavior at boundary values (0 KRW, the maximum)?" Test code automates
that checklist. Instead of a person eyeballing everything each time, one
`python3 main.py` inspects every item. Changed even one line? Run the
checklist again — that is a **regression test**.

## 3. Core concepts

### 3.1 Type hint syntax

```python
def calc_vat(amount: int, rate: float = 0.1) -> int:
    ...

price: int = 4500
stores: list[str] = ["Downtown", "Riverside"]
stock: dict[str, int] = {"Ballpoint pen": 37}
memo: str | None = None          # a string or None (a value that may be absent)
```

`arg: type` and `-> return-type` is all there is. For collections, spell out
the content type as in `list[int]`; a "may be absent" value is written
`| None`.

An important fact: **Python does not enforce type hints at runtime.**
Violate a hint and the program still runs. Hints take effect through three
channels — (1) conveying the spec to readers, (2) editor autocomplete and
warnings, (3) a **static checker** like mypy scanning the code without
running it and reporting violations. mypy is a separately installed tool, so
this exercise covers the concept only — main.py builds a **micro checker** so
you experience the principle (matching signatures against actual values).

### 3.2 assert — a one-line automatic check

```python
assert calc_vat(10000) == 1000, "VAT on 10,000 KRW must be 1,000 KRW"
```

`assert condition, message` passes silently when the condition is true and
raises an `AssertionError` with the message when false. The core of testing —
"compare expected against actual" — lives in this one line. Real teams use
dedicated tools like pytest, but at the root it is all assert.

### 3.3 The three-piece set of good tests

Even for one function, you can only relax after checking three families of
input.

1. **Normal cases**: representative input → expected output
2. **Edge cases**: 0, an empty list, the maximum, a value exactly on the threshold
3. **Error cases**: does bad input raise **the intended exception**?
   (Silently returning a wrong value is the worst outcome.)

Most bugs come from 2 and 3. The habit of asking "what if an empty list
arrives?" in a test reduces the 3 a.m. incident calls.

### 3.4 Docstrings — the function's user manual

```python
def net_price(price: int, discount_rate: float) -> int:
    """Compute the discounted price.

    Args:
        price: list price (KRW). 0 or more.
        discount_rate: discount rate. 0.0 to 1.0.
    Returns:
        Discounted price (KRW, floored to an integer).
    Raises:
        ValueError: if the discount rate is out of range.
    """
```

The triple-quoted string at the head of a function is the docstring. It shows
up instantly via `help(net_price)`, editors display it automatically, and it
carries what type hints cannot — units, ranges, exception conditions. The
division of labor: "type hints = the spec, docstring = the user manual."

### 3.5 A code-quality checklist

- Do the names alone convey the meaning? (the level01/05 principle)
- Does each function do one thing, staying under 20-30 lines?
- Named constants instead of magic numbers? (`VAT_RATE`, not `0.1`)
- Consistent formatting? — there is a community style guide (PEP 8) and
  automatic formatters
- And: never refactor without tests

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level11_typing_testing_quality
python3 main.py
```

A capstone demo: the tax/discount functions (level05's material) get type
hints and docstrings, then are verified by a home-grown test runner.

- **[1] Reading the contract**: prints the hinted functions' signatures and shows how Python remembers hints via `__annotations__`.
- **[2] Hints are not enforced**: shows a spec-violating call just running anyway, leading to the conclusion that a checker is needed.
- **[3] A micro type checker**: a 20-line `tiny_type_check` that matches a function's hints against actual argument types and catches the violations from [2] — a hands-on taste of what mypy does.
- **[4] A mini test runner**: writes 9 assert tests across the normal/edge/error trio, tallies pass/fail, and prints a `9 passed`-style report. A deliberately planted bug (boundary handling) is caught by the tests, and the fixed version passes everything.
- **[5] Using docstrings**: prints the manual that `help()` displays.

## 5. Try it yourself

1. **Add hints** — attach type hints and a docstring to a function from an
   earlier level (e.g., `format_krw` from level08's utils). (Hint:
   `def format_krw(amount: int) -> str:`)
2. **Add tests** — add two cases to [4]: "discount rate 1.0 (full discount)
   gives 0 KRW" and "a negative price raises ValueError". (Hint: exception
   tests wrap the call in try and invert the verdict — "pass if it raises" —
   the runner contains an example.)
3. **Bug-reproduction drill** — deliberately restore the bug fixed in [4],
   rerun the tests, and see which test flips to failure. (Hint: with tests,
   the "reproduce → fix → re-verify" cycle takes a minute.)

## 6. Common mistakes

- **Believing hints make you safe** — Python does not enforce hints at
  runtime. Run a checker (mypy), or validate critical values directly in code
  (raise).
- **Testing only the normal cases** — passing them proves little. Edge and
  error cases are the main event.
- **Tests that copy the implementation** — build the expected value by
  copying the function's formula and you copy its bugs too. Derive expected
  values by hand, independently.
- **Docstring/code drift** — fix the code without fixing the manual and you
  have a lying document. Whenever you edit a function, reread its docstring.
- **Using assert for input validation** — assert can be switched off entirely
  under the optimizer flag (-O). Validate user input with raise (level07);
  keep assert for tests — separate the uses.

## Next level preview

You have completed lecture02. From sequence, branching, and repetition all
the way to contracts and quality inspection — you are now ready to write not
just "code that runs" but "code you can trust." In lecture03 we stack NumPy
and pandas on top of these fundamentals and begin the journey of moving your
Excel data work into Python.
