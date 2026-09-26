# Lecture 03 — Working with Data (NumPy · Pandas)

> This lecture moves every table task you used to do in Excel into code — and then goes on to the things Excel simply cannot do.

## What you will learn

- Why data gets organized as a "table of rows and columns", and what changes once you handle that table with code
- The vectorized way of thinking: computing on hundreds of thousands of numbers at once with NumPy arrays (ndarray)
- The full pipeline of practical data analysis with a Pandas DataFrame: loading → cleaning → filtering → aggregating → merging → time series → pivoting
- Strategies for large data — saving memory and processing in chunks when data outgrows your laptop, and knowing when to move on to a database or Spark

Every exercise uses sales data from a fictional cafe chain with 5 stores
(Downtown, Riverside, Airport, University, Lakeside).
The data is generated directly in code by `common/hjh_data.py`, so no internet
connection is needed — and it comes pre-contaminated with missing values and
negative outliers, just like real-world data.

## Prerequisites

- **lecture01 — Computers and the Development Environment** and **lecture02 — Python Programming Basics** (variables, lists, dictionaries, loops, functions, reading and writing files)
- If you have ever used SUM, filters, or pivot tables in Excel, the analogies will land even better.

## Level overview

| Level | Title | Difficulty |
|---|---|---|
| [level00](level00_what_is_data/README.md) | What Is Data? — The Structure of a Table | ⭐ |
| [level01](level01_excel_to_python/README.md) | From Excel to Python | ⭐ |
| [level02](level02_numpy_basics/README.md) | NumPy Array Basics | ⭐⭐ |
| [level03](level03_pandas_dataframe/README.md) | Pandas — Series and DataFrame | ⭐⭐ |
| [level04](level04_loading_data/README.md) | Loading Data (CSV · Excel · JSON) | ⭐⭐ |
| [level05](level05_filter_sort_select/README.md) | Filtering, Sorting, and Selecting | ⭐⭐ |
| [level06](level06_missing_outliers/README.md) | Handling Missing Values and Outliers | ⭐⭐⭐ |
| [level07](level07_groupby_aggregation/README.md) | Grouping and Aggregation (groupby) | ⭐⭐⭐ |
| [level08](level08_merge_join/README.md) | Merging and Joining (merge · concat) | ⭐⭐⭐ |
| [level09](level09_time_series/README.md) | Working with Time Series Data | ⭐⭐⭐⭐ |
| [level10](level10_pivot_reshape_window/README.md) | Pivoting, Reshaping, and Window Operations | ⭐⭐⭐⭐ |
| [level11](level11_large_data_strategies/README.md) | Strategies for Processing Large Datasets | ⭐⭐⭐⭐⭐ |

## The fast track (if you are short on time, do just these 5)

1. **level03** — If you don't know the DataFrame structure, you can't do anything else.
2. **level05** — 80% of real-world questions boil down to "pick the rows and line them up".
3. **level06** — Real data is always contaminated. Always.
4. **level07** — Questions like "What are sales per store?" are answered by groupby.
5. **level08** — No real analysis ends with a single table. Merging is where practice begins.

After the fast track, add level09 if your job involves time series, or level10 if you write a lot of reports.

## Where this lecture shows up at work

- **Monthly performance report**: open the sales CSVs from 30 stores and pull per-store, per-category totals plus month-over-month growth in five minutes. (level04 · 07 · 09)
- **Data audits**: when a transaction log arrives with blank values and negative amounts mixed in, diagnose how many rows are contaminated and decide on a handling policy. (level06)
- **Target tracking**: combine the sales table, the store info table, and the target table into a ranking of target achievement by store. (level08 · 10)
- **Automation**: replace the Excel busywork you repeated every week with a single script that reproduces the exact same result every time. (level01 · 11)

## How to run

Run each level from inside its own folder, like below. Outputs such as figures and CSV files are created in each level's `outputs/` folder.

```bash
cd lecture03_data_handling/level03_pandas_dataframe
python3 main.py
```
