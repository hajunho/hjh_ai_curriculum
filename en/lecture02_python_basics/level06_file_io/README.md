# Lecture 02 · Level 06 — Reading and Writing Files

> Learn to read and write text and CSV files with open/with, and you can produce results that outlive the program — reports and data files.
**Difficulty** ⭐⭐ / **Prerequisites** level05 / **Estimated time** 50 min

## 1. Why learn this — the business angle

So far our exercises only printed results to the screen. When the program
ends, the variables and results all vanish. Real work is different: sales
data arrives as CSV files, and analysis results must be saved to files —
attached to emails, uploaded to shared folders.

File I/O is the gateway between your program and the outside world. Finish
this level and you can automate "read the transactions CSV every morning and
produce a summary report" — for many office workers, this is literally the
first thing they ever automate with Python. You will also learn to prevent,
in advance, the trap everyone handling international text eventually hits:
**encoding** problems.

## 2. Understanding through analogies

**The filing-cabinet analogy.** Working with a file follows the same
procedure as pulling a folder from the office cabinet.

1. **Take** the folder out of the cabinet → `open("filename")`
2. Read it or **add** to it → `read()` / `write()`
3. Always **put it back** → `close()`

Leave folders piled on your desk when you go home and the next person can't
use them — and things get lost. Files are the same: fail to close one and the
contents may not be saved, or another program may not be able to open it. The
problem is that people forget. So Python provides `with`, an "automatic
return system": when the `with` block ends — even if an error blows you out
mid-way — the file closes automatically. Think of it as the library's
self-return machine.

**Encoding is the storage standard for characters.** The same document, saved
under one standard (utf-8, cp1252, etc.), turns to garbage when opened under
another. It is like trying to push a plug into a foreign socket without an
adapter. Today's standard is utf-8, and **always spelling out
`encoding="utf-8"` when opening and writing** is the survival rule for any
text beyond plain ASCII — accented names, em dashes, emojis, Korean won signs.

## 3. Core concepts

### 3.1 with open — the standard pattern

```python
with open("report.txt", "w", encoding="utf-8") as f:
    f.write("Q1 summary\n")          # \n is a newline (not added for you!)

with open("report.txt", "r", encoding="utf-8") as f:
    content = f.read()               # the whole file as one string
```

Memorize just three modes for now.

| Mode | Meaning | Watch out |
|---|---|---|
| `"r"` | read (the default) | FileNotFoundError if the file is missing |
| `"w"` | write | **wipes the existing contents** and writes fresh |
| `"a"` | append | continues after the existing contents (good for logs) |

`"w"` overwriting the whole file is the most painful mistake in practice, so
it bears repeating.

### 3.2 Reading line by line

Reading a big file whole with `read()` eats all your memory. A file object
yields one line at a time when put in a loop, so this is the standard pattern:

```python
with open("data.txt", encoding="utf-8") as f:
    for line in f:                   # one line at a time (memory-friendly)
        line = line.strip()          # remove the trailing newline
        ...
```

Skip `strip()` and every line carries an invisible `\n` at the end, breaking
comparisons and conversions.

### 3.3 CSV — the lingua franca of tabular data

CSV (comma-separated values) is comma-delimited tabular text that Excel,
databases, and Python all read. You could `split(",")` by hand, but values
containing commas ("Portland, OR") would break, so we use the standard
library's `csv` module.

```python
import csv

with open("sales.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "item", "amount"])       # header
    writer.writerow(["2026-09-01", "Ballpoint pen", 24000])

with open("sales.csv", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):                     # each row arrives as a dictionary!
        print(row["item"], row["amount"])
```

`DictReader` uses the header as keys and returns each row as a dictionary —
exactly the "list of dictionaries" shape from level04. Every value read from
CSV is a **string**, so numbers need `int()`/`float()` conversion (the
level01 trap again). Passing `newline=""` when writing is the convention that
prevents blank lines sneaking in on Windows.

### 3.4 Paths and os.makedirs

Result files are collected in a fixed place like `outputs/` under the
exercise folder.

```python
import os
os.makedirs("outputs", exist_ok=True)     # create if missing, pass quietly if present
```

Relative paths depend on which folder you run the script from, so main.py
builds paths relative to "the folder this file lives in" via
`os.path.dirname(__file__)`.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level06_file_io
python3 main.py
```

A three-act structure: create a transactions CSV → read it → generate a
summary report. Every artifact is saved under `outputs/` and its path is
printed.

- **[1] Create the CSV**: writes 20 September transactions (seeded random) to `outputs/transactions.csv` with `csv.writer`. This plays the role of "the file the system spat out."
- **[2] Read & aggregate**: reads line by line with `csv.DictReader`, tallying per-category totals and the largest transaction. The point: never forget the string → number conversion.
- **[3] Save the report**: turns the tallies into a human-readable summary, saves it to `outputs/daily_report.txt`, then reads it back and shows it on screen.
- **[4] Append mode and encoding**: appends to a processing log with `"a"` mode, and safely reproduces what happens when a utf-8 file is read with the wrong encoding.

When the run ends, open the `outputs/` folder yourself and inspect the files.
The results surviving after the program ends — that is this level's
achievement.

## 5. Try it yourself

1. **Add a report line** — add an "average transaction" line to the summary
   report. (Hint: total and count are already tallied; one division does it.)
2. **Filtered save** — select only transactions of 100,000 KRW or more and
   save them to `outputs/big_transactions.csv`. (Hint: read with DictReader
   and write only matching rows with a writer.)
3. **Watch the log grow** — run main.py three times in a row, then open
   `outputs/run_log.txt`. Why do the lines keep multiplying?
   (Hint: the difference between `"a"` mode and `"w"` mode.)

## 6. Common mistakes

- **Overwriting a precious file with `"w"`** — write mode erases the existing
  contents instantly. Only ever read the source data file; always save results
  under a **different name**.
- **Omitting encoding** — omit it and the OS default is used, so a file made
  on Windows (cp1252) breaks on Mac/Linux. Make `encoding="utf-8"` a habit for
  both reading and writing.
- **Doing math on CSV values still in string form** — `row["amount"] + 100`
  is a TypeError or nonsense string concatenation. Always
  `int(row["amount"])`.
- **open without close** — write only `f = open(...)` and never close, and
  the buffer may never be saved. Always use `with` and the worry disappears.
- **Leaving the trailing `\n`** — `strip()` before comparing a read line.
  `"pen\n" == "pen"` is False.

## Next level preview

Real-world files always contain broken data. The amount column says "TBD",
an entire row is blank. In level07 we learn try/except exception handling —
building programs that don't grind to a halt over one bad row — and how to
read error messages and debug.
