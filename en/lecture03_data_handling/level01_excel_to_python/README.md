# Lecture 03 · Level 01 — From Excel to Python

> We move the sums, filters, and sorts you do in Excel every day into code, and experience how code overcomes Excel's three limits — reproducibility, scale, and automation.

**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 45 min

## 1. Why learn this — the business view

Excel is a great tool. It is easy to learn, you can fix things while looking at them, and it is the lingua franca of the office. But once data grows and work starts repeating, you hit three walls.

1. **Reproducibility**: ask how last month's report numbers were produced and you get an answer that relies on memory — "I applied a filter, copied it somewhere, added a formula...". The sequence of clicks is recorded nowhere. When the person in charge changes, it is common that the numbers can never be rebuilt.
2. **Scale**: an Excel sheet has a physical ceiling of about 1.04 million rows, and in practice it gets visibly slow from a few hundred thousand rows. "The file won't open" is a familiar cry from data practitioners.
3. **Automation**: opening the same file every Monday and building the same pivot is 30 minutes plus a chance of mistakes when a human does it — and 3 seconds plus identical output every time when code does it.

Moving to Python does not mean abandoning Excel. It means **turning a "procedure of clicks" into a "readable document (code)"**. Code is its own work manual: it can be reviewed, re-run, and scheduled.

## 2. Understanding through an analogy

Manual Excel work is like **cooking a stew by feel**. It tastes great that day, but "a splash of soy sauce, hmm... what did I add next?" makes the same taste hard to reproduce. Code is a **recipe with exact measurements**. With a recipe:

- Anyone who cooks it gets the same taste. → **Reproducibility**
- A one-serving recipe scales straight into a hundred-serving pot. → **Scale**
- Drop the recipe into a cooking machine and it cooks itself every morning. → **Automation**

And when the recipe says "2 spoons of salt", you know exactly what to fix when someone complains it is too salty. That means code review is possible. Excel's undo history (Ctrl+Z) evaporates the moment you close the file; code stays.

## 3. Core concepts

### 3-1. CSV — the bridge where Excel and Python meet

CSV (Comma-Separated Values) is a plain-text table with values separated by commas. It opens in Excel, it opens in Notepad, and Python can read it. There is no formatting, no formulas, no charts — **only values**. That is why it serves as the de facto standard for exchanging data between systems. In this exercise we save the sales table to CSV and read it back. The file you get from a "Download data" button at work is usually this format.

### 3-2. Excel task ↔ Python code mapping

| What you did in Excel | Python equivalent |
|---|---|
| `=SUM(G2:G1000)` | `sum(list_of_values)` |
| Auto-filter to show only "Downtown" | `[r for r in rows if r["store"] == "Downtown"]` |
| Data → Sort (descending) | `sorted(rows, key=..., reverse=True)` |
| SUMIF (totals per store) | Accumulate per store in a dictionary |
| Recording a macro | The script file itself |

Once this mapping settles in your head, the question "I could do it in Excel, but how do I do it in code?" turns into "How do I translate this Excel action into code?" — a far easier question to answer.

### 3-3. The CSV trap — everything reads back as a string

When you read a CSV with `csv.DictReader`, every value arrives as a string (str). `"512000"` looks like a number but is a string, so it cannot go straight into `sum()`. A conversion step with `int()`/`float()` right after reading is therefore mandatory. Empty cells (`""`) blow up during conversion, so they must be filtered out first. In level04 you will compare how much of this "type restoration" work Pandas does for you.

### 3-4. Code as a verifiable procedure

Manual Excel work leaves behind only "the numbers". Code leaves behind "the numbers + the procedure that produced them". In an audit or a handover, that difference is decisive. This level's main.py is also built to report, by itself, how many missing/negative rows it skipped. Good data code always tells you what it threw away.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level01_excel_to_python
python3 main.py
```

The flow of the script is "a day of Excel work" translated directly into code.

- **[1] Save CSV**: saves 60 days of sales (1,500 rows) to `outputs/sales.csv`. Feel free to open this file in Excel yourself.
- **[2] Read CSV**: reads it back with `csv.DictReader`. The output confirms that every value is a string at this point.
- **[3] Restore types + skip contamination**: converts `revenue` to an integer, but skips empty and negative values — and reports how many were skipped.
- **[4] SUM**: total revenue, the same as one `=SUM()` in Excel.
- **[5] SUMIF**: builds totals per store by accumulating in a dictionary.
- **[6] Auto filter**: picks only "Downtown & weekend" rows and compares the average revenue with weekdays.
- **[7] Top 5 by sort**: prints the top 5 rows by revenue in descending order.
- **[8] Reproducibility statement**: prints why running this script 100 times yields the same numbers (fixed seed + a recorded procedure).

In the output, check that per-store totals grow in the order Downtown < Riverside < Airport < University < Lakeside, and that the weekend average is above the weekday average. The patterns planted in the data generator surface precisely through code.

## 5. Try it yourself

1. **(Easy)** Using [5] as a reference, compute revenue totals "per category". Hint: just change the accumulation dictionary's key to `r["category"]`.
2. **(Medium)** Instead of the top 5, pull the "bottom 5" (excluding negatives). Hint: drop `reverse=True`, or take the front of the sorted result.
3. **(Challenge)** Save the summary (totals per store) as a new CSV file `outputs/summary.csv`. Hint: `hjh_data.to_csv` takes a list of dictionaries, so build something like `[{"store": s, "total": t} for ...]` and pass it in.

## 6. Common mistakes

- **Using values straight from the CSV in calculations**: `"500" + "300"` is `"500300"`. Always convert to numbers right after reading.
- **Feeding an empty string to `int()`**: missing values are stored in the CSV as `""`, and `int("")` raises a `ValueError`. Check for empty values before converting.
- **A CSV with non-ASCII text looks garbled in another program**: that is an encoding issue. This curriculum standardizes on UTF-8; level04 covers how to deal with it in detail.
- **Overwriting the source file with code**: build the habit of only reading the source and always saving results to a different file (`outputs/`). It is the same family of mistake as forgetting "Save As" over an original in Excel.

## Next level preview

The loop-based approach in this level slows down once data reaches millions of rows. In the next level we learn **NumPy arrays** and measure, first-hand, the speed of "vectorization" — computing millions of numbers at once with no loops.
