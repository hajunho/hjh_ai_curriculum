# Lecture 02 · Level 07 — Exception Handling and Debugging

> try/except is the emergency-response manual — "if something goes wrong, respond like this" — and an error message is a precise incident report sent by the computer.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level06 / **Estimated time** 50 min

## 1. Why learn this — the business angle

The reconciliation batch that should have run all night stopped at 2 a.m. The
cause: one row out of 10,000 — someone had typed "TBD" in the amount column.
A program without exception handling dies right there on that one row, and
the other 9,999 never get processed.

Real-world data is always broken. Blank cells, typos, dates in the wrong
format, negative quantities. Exception handling is the defense system that
says "log the one bad row and skip it, but keep the whole job going." And
debugging is the skill of finding the cause of an error from evidence — the
error message and printed intermediate values — rather than gut feeling.
These two things separate "code that ran once on my machine" from "code fit
for production."

## 2. Understanding through analogies

**The fire-drill analogy.** Having extinguishers and an evacuation manual in
a building does not prevent fires. It keeps a fire from **bringing the whole
building down**. That is exactly the role of try/except.

> try: carry on with normal work (the zone where a fire might break out)
> except: if a fire starts, respond per this manual (where the extinguisher is)
> finally: fire or no fire, lock up before going home (always runs)

**The incident-report analogy.** A Python error message (traceback) is not
red text designed to scare you — it is an incident report written to the
who-what-when-where-why standard. **Which file, which line, which code, what
kind of accident, and why** — it is all in there. There is one reading trick:
**bottom to top**. The last line is the accident type and cause
(`ValueError: ...`); right above it is the accident site. Changing code at
random without reading the report is like repainting random walls without
visiting the accident scene.

## 3. Core concepts

### 3.1 The five exceptions you will meet most

| Exception | Meaning | Common cause |
|---|---|---|
| `ValueError` | value doesn't fit the format | `int("TBD")` |
| `TypeError` | operation on mismatched types | `"100" + 5` |
| `KeyError` | key missing from a dictionary | `row["amount"]` (typo in the column name) |
| `IndexError` | index out of range | `items[0]` on an empty list |
| `ZeroDivisionError` | division by zero | computing an average when the count is 0 |

Add `FileNotFoundError` (opening a missing file) and these six cover most
early real-world errors. The names themselves explain the cause — read the
name precisely and you are halfway to the fix.

### 3.2 try / except / else / finally

```python
try:
    amount = int(raw_value)          # the accident-prone zone
except ValueError:
    log_error(raw_value)             # response when a ValueError occurs
    amount = 0
else:
    process(amount)                  # runs only when there was NO accident (optional)
finally:
    close_connection()               # always runs, accident or not (optional)
```

- **Name the exception type.** Swallow everything with `except:` or
  `except Exception:` and you hide real bugs too (like a NameError from a
  typo).
- Group several types as `except (ValueError, KeyError):` or write multiple
  except clauses.
- Catch with `except ValueError as e:` and `e` carries the cause text you can
  write to a log.
- Use finally for cleanup that "must happen no matter what," like closing
  files and connections (this is exactly what `with` does for you internally).

### 3.3 Guard narrowly, respond specifically

Put all 100 lines inside one try block and you cannot tell where the accident
happened. The principle: wrap only **the smallest accident-prone stretch**
(type conversion, file opening, key lookup). And doing nothing in except —
just `pass` — is switching off the fire alarm. At minimum, record how many
rows failed and why, so you can later answer "why is the total short?"

### 3.4 raise — sounding the alarm yourself

```python
if amount < 0:
    raise ValueError(f"Amount cannot be negative: {amount}")
```

When letting a bad value pass silently is more dangerous than making noise
early, raise the exception yourself with `raise`. It is a contract
declaration: "this function refuses this kind of input."

### 3.5 print debugging — the most primitive and the most reliable

When there is no error but the result is wrong, stick prints into the suspect
stretch and inspect intermediate values.

```python
print(f"DEBUG: row={row!r}, amount={amount!r}")
```

The technique: print with the variable name attached, use `!r` so quotes and
whitespace become visible (hidden spaces like `"100 "` show up), and delete
the debug prints once the cause is found. Fancier tools (a debugger, logging)
come later, but print debugging you will use forever.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level07_errors_debugging
python3 main.py
```

We process 10 rows of deliberately contaminated monthly sales data — "TBD"
in the amount, a blank value, hidden whitespace, a missing key, a
division-by-zero setup.

- **[1] Observing exceptions**: deliberately triggers the five representative exceptions one by one, catching each with except and showing "how to read the last line."
- **[2] Unprotected processing**: reproduces how aggregating the dirty data with no exception handling drops dead on the 3rd row (wrapped once on the outside so the whole demo survives).
- **[3] Protected processing**: with try/except, logs each bad row with its reason and skips it, aggregating to the end. Out comes an operations report of the form "7 succeeded, 3 failed (reasons listed)."
- **[4] print debugging**: catches the case where nothing errors but the total is off (the hidden whitespace " 1000") using `!r` debug output.
- **[5] raise**: demonstrates "alert early and loudly" with a validation function that rejects negative amounts.

The contrast between [2] and [3] — same data, one program dies, one survives
— is the heart of this level.

## 5. Try it yourself

1. **Add a new contamination** — add a row whose amount is `"10,000"` (with a
   comma). Which exception occurs, and how does [3] handle it? (Hint:
   `int("10,000")` is a ValueError. Preprocessing with `replace(",", "")` can
   rescue it.)
2. **Failure-rate alarm** — add a condition to the [3] report that prints
   "Data quality check needed!" when the failure rate exceeds 20%. (Hint:
   `failed / total > 0.2`)
3. **Traceback reading drill** — produce the traceback from trying
   `items[0]` on an empty list `[]` and interpret the last two lines out
   loud. (Hint: last line = type and cause; the one above = file, line
   number, and the offending code.)

## 6. Common mistakes

- **Swallowing everything with `except:`** — hides real bugs too. Name only
  the types you expect.
- **Silently `pass`-ing in except** — the very fact that failures happened
  disappears. Record the count and the reason at minimum.
- **Wrapping the whole block in try** — you lose the accident site. Smallest
  stretch only.
- **Editing before reading the traceback** — the last line alone gives you
  half the cause. Searching the error text is a good habit too (someone has
  read the very same incident report before you).
- **Leaving debug prints behind** — forget to delete them and your production
  logs get polluted. A `DEBUG:` prefix makes them easy to find and remove
  later.

## Next level preview

As code grows, it no longer fits in one file. In level08 we learn to split
code into modules and assemble them with import, and we tour the standard
library Python ships with (datetime, math, random, collections).
