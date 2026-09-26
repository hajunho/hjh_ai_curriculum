# Lecture 03 · Level 11 — Strategies for Processing Large Datasets

> The survival skills for when data starts outgrowing memory — the dtype diet, chunked processing, categoricals — and knowing "the moment to leave pandas".

**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level04, level07 / **Estimated time** 70 min

## 1. Why learn this — the business view

The analysis script that ran fine yesterday dies one day with "MemoryError". The company grew, and the data grew tenfold with it. At that fork there are two roads. One is opening a months-long project — "we're big data now, let's adopt Spark". The other is a 30-minute tune-up that keeps it running on the same laptop. Surprisingly, the vast majority of "large data" problems you meet in practice are solved by the second road.

What this level teaches is what that 30-minute tune-up actually is: putting data types (dtype) on a diet to cut memory to less than half, reading files in chunks to process files bigger than memory, and compressing repetitive strings with categoricals. And finally, the criteria for telling when even these techniques won't do — when it truly is time to move to a database or Spark. Knowing a tool's limits is as much a senior skill as knowing the tool.

## 2. Understanding through an analogy

**Memory is your desk; disk is the archive.** pandas is the tool that "pulls all the books (data) out of the archive (files) and spreads them across the desk (memory) to work". When the books outgrow the desk, there are three strategies.

- **The dtype diet = switching to paperbacks.** The same content shelved as paperbacks (int32, 4 bytes) instead of hardcovers (int64, 8 bytes) takes half the desk space. If the range of your numbers fits in the paperback, there is no reason to insist on the hardcover.
- **categorical = a dictionary of abbreviations.** Instead of writing out the word "Downtown" 250,000 times, you write "1 = Downtown" once in a dictionary and put only 1 in the text. The more repetitive the strings, the more dramatic the shrinkage.
- **Chunked processing = fetching the book one chapter at a time.** If the complete works won't fit on the desk, bring one volume at a time, keep only your summary notes, and return it. The desk is small, yet you can summarize the whole collection. It only works, though, for jobs whose partial results can be combined — like "summaries" (aggregations).

Changing the archive itself is what a column-oriented format like Parquet does, and when the archive reaches library scale, you call in a librarian (a DB) or a distributed archive (Spark).

## 3. Core concepts

### 3-1. Measure first — memory_usage(deep=True)

Every diet starts at the scale. `df.memory_usage(deep=True)` reports each column's memory usage in bytes. The `deep=True` matters: without it, string (object) columns count only pointer sizes and report far smaller than reality. Measure, and it is almost always the string columns that are the memory hippos.

### 3-2. The dtype diet

| Conversion | Size change | Condition |
|---|---|---|
| int64 → int32 | 8B → 4B (½) | values within ±2.1 billion |
| float64 → float32 | 8B → 4B (½) | when ~7 significant digits suffice |
| object → category | down to a tiny fraction | unique values ≪ row count |

Two cautions. float32 carries only about 7 significant digits, so the last digits of big sums — like revenue totals — can shift (keep float64 for accounting reports). And category backfires when the unique count approaches the row count (IDs like order numbers): the dictionary's upkeep costs more than it saves.

### 3-3. read_csv(chunksize=...) — the streaming aggregation pattern

`pd.read_csv(path, chunksize=50_000)` returns, instead of a DataFrame, "an iterator that hands out 50,000 rows at a time". The working pattern is fixed:

```python
total = None
for chunk in pd.read_csv(path, chunksize=50_000):
    part = chunk.groupby("store")["revenue"].sum()
    total = part if total is None else total.add(part, fill_value=0)
```

The key requirement: it must be an operation whose **partial aggregates combine into the full aggregate** (sums, counts, max/min). The mean must be rewritten as "accumulate sums and counts separately, divide at the end", and statistics that need the whole dataset — like the median — cannot use this pattern.

### 3-4. Column-oriented formats — the Parquet concept

CSV is row-by-row text, so it ① stores numbers as characters (big), ② needs parsing every time, and ③ forces you to read everything even if you only need 2 columns. Parquet is a binary format that stores data column by column, with type information and compression built in. You can read only the columns you need, dtypes are preserved, and files typically come out a fraction of the CSV size. In pandas it is one line — `df.to_parquet(...)` / `pd.read_parquet(...)` — but it requires an engine like pyarrow, so this exercise covers it as a concept and commented code only. For data you read repeatedly, the habit "receive as CSV, save as Parquet, work from that" saves the whole team's time.

### 3-5. When do you leave pandas?

| Signal | Prescription |
|---|---|
| data ≤ 1/3 of memory | just use pandas |
| fits in memory, but barely | dtype diet + Parquet |
| bigger than memory, but the goal is aggregation | chunksize streaming, or Polars/DuckDB |
| several people query/update the same data | a database (taught in lecture04) |
| hundreds of millions of rows, multiple servers needed | distributed processing like Spark |

More important than "how many GB" is **the nature of the work**. A one-off solo analysis can survive on pandas up to astonishing sizes; conversely, even small data needs a DB the moment it requires concurrent access and updates.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level11_large_data_strategies
python3 main.py
```

The output flows as follows.

- **[1] Stepping on the scale** — builds 8 years of sales (73,000 rows) and prints per-column memory with `memory_usage(deep=True)`. See how many times heavier the string columns (date, weekday, store, category) are than the numeric ones.
- **[2] The dtype diet** — converts strings to category, int64 to int32, float64 to float32, then compares total memory before/after with the savings (%). It also verifies no values were altered.
- **[3] Chunked streaming aggregation** — saves the data to CSV, reads it back in `chunksize=10_000` pieces while accumulating per-store revenue totals, and verifies the result matches a single full-read aggregation. This is the step where "files bigger than memory yield to this pattern" gets into your muscle memory.
- **[4] Optimized reading comparison** — reads the same CSV ① with no options and ② with `usecols` + `dtype` specified, comparing elapsed time and the resulting DataFrame's memory. Reading only the columns you need is the most reliable optimization there is.
- **[5] Criteria summary** — prints the "when to leave pandas" decision table.

## 5. Try it yourself

1. **(Basic)** In [2], try int16 instead of int32 for `day_index`. Is that safe for this data? Hint: int16 tops out at 32,767. Compare with `day_index`'s maximum.
2. **(Applied)** Use the chunk pattern from [3] to compute "mean revenue per store". Hint: accumulate sums and counts separately per chunk, then divide at the end. Also think through why averaging the per-chunk means gives the wrong answer.
3. **(Challenge)** Time [3] with `chunksize` set to 1,000 / 10,000 / 50,000. Explain why chunks that are too small are actually slower (the fixed cost paid per chunk).

## 6. Common mistakes

- **Measuring memory without deep=True** — the true size of string columns stays hidden, producing the mystery "there's plenty of memory, why did it die?".
- **Converting ID columns to category too** — a column with as many unique values as rows makes category *increase* memory. Use it only on highly repetitive columns.
- **Reporting accounting figures after converting to float32** — the ~7-significant-digit limit can shift the last digits of large totals. If the report must be exact to the won, keep float64 or work in integers.
- **Computing a median with the chunk pattern** — statistics whose partial results cannot be combined cannot be streamed chunk-wise. You need approximate algorithms or DB/distributed tools.
- **Premature Spark adoption** — millions of rows are still pandas (or Polars/DuckDB) territory. Never forget that a distributed system is an operating cost in itself.

## Next level preview

lecture03 ends here. You have completed the craft of handling a single table — but "data that many people use at the same time" lives in a database, not a file. The next lecture, **lecture04 — Databases and SQL**, crosses into that world. The answer to this level's "when do I move to a DB?" lives there.
