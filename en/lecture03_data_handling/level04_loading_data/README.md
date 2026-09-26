# Lecture 03 · Level 04 — Loading Data (CSV · Excel · JSON)

> Real-world data arrives as files — we conquer the read_csv options and the encoding problem.

**Difficulty** ⭐⭐ / **Prerequisites** level03_pandas_dataframe / **Estimated time** 45 min

## 1. Why learn this — the business view

The first gate of any analysis job is always "opening the file". The CSV that IT extracted for you, the Excel a business partner sent, the JSON a system spat out — bringing files of wildly varying formats and encodings safely into a DataFrame is step zero of every analysis. And an astonishing amount of time disappears at this step. Text gets mangled into `ê°•ë‚¨`-style gibberish, the number "1,234" reads in as text, and missing values written as "-" quietly wreck every average you compute.

Know a handful of `read_csv` options and 90% of these problems dissolve into a single option line. Not knowing them sends you back to opening the file in Excel and fixing it by hand — and at that moment reproducibility dies. The iron rule of this level: **solve it in code, never by touching the file**.

## 2. Understanding through an analogy

Reading a CSV file is like **clearing an international parcel through customs**. The parcel (the file) is packed according to the sender country's rules. The language (encoding) differs, the way the item list is separated (the delimiter) differs, and the convention for writing "none" (the missing-value marker) differs. Fill in the customs declaration (the read_csv options) precisely — "this parcel is written in cp949, items are separated by semicolons, and '-' means an empty slot" — and the parcel sails through. Fill it in carelessly? Customs (Pandas) misreads the contents or returns the parcel outright (an error).

For encoding specifically, the **lock and key** analogy is exact. A file locked with cp949 opens only with the cp949 key. Insert the utf-8 key and the lock doesn't break — you just get the error "wrong key" (`UnicodeDecodeError`). Swap keys and you're done.

## 3. Core concepts

### 3.1 A map of the essential read_csv options

```python
pd.read_csv(
    "sales.csv",
    sep=",",              # delimiter. use ";" for semicolon files
    encoding="utf-8",     # if text is garbled, try "cp949" (legacy Korean) etc.
    na_values=["-", ""],  # treat these markers as missing (NaN)
    thousands=",",        # turn "1,234" into the number 1234
    parse_dates=["date"], # parse the date column as real dates
    usecols=["date", "revenue"],  # only the columns you need
    nrows=1000,           # only the first 1000 rows (preview of large files)
    dtype={"zipcode": str},       # protect values with leading zeros, like zip codes
)
```

No need to memorize them all. As long as the map "this kind of problem is solvable with an option" is in your head, you can look them up when needed.

### 3.2 The great ambush — encoding

An encoding is the rule that converts characters into the computer's numbers. The international standard is utf-8, but plenty of legacy systems use something older — for our fictional Korean cafe chain, the old POS exports use cp949 (the euc-kr family), and old Windows systems in the West often used cp1252. The symptoms and prescriptions come in exactly two flavors.

| Symptom | Prescription |
|---|---|
| `UnicodeDecodeError` raised | re-read with the right key, e.g. `encoding="cp949"` |
| No error, but text looks like alien script | re-read with the opposite encoding |

When you save files yourself, always use `encoding="utf-8"` (`utf-8-sig` for files that will be opened in Excel) — a habit that will rescue your future colleagues.

### 3.3 The "numbers read as text" accident

When a system exports comma-grouped numbers like "1,234,000", Pandas reads them as text (object). From that moment `sum()` becomes string concatenation or an error. The `thousands=","` option is a one-line fix. Likewise, for files that write missing values as "-", "N/A", "none", and so on, hand the list of markers to the `na_values` option.

### 3.4 JSON — the parcel box between systems

JSON is the standard format web systems use to exchange data. Tabular data usually arrives in **records form** (a list where one row = one dictionary).

```json
[{"store": "Downtown", "revenue": 512000},
 {"store": "Riverside", "revenue": 430000}]
```

This shape becomes a table immediately via `pd.read_json(path)`, or `json.load()` followed by `pd.DataFrame()`. Nested JSON, on the other hand, must be flattened with `pd.json_normalize()`.

### 3.5 Excel files — concept only

You can read Excel too: `pd.read_excel("report.xlsx", sheet_name="January")` (requires the openpyxl package). But Excel files cause frequent accidents thanks to sheets, formatting, and merged cells, so the professional rule of thumb is: **ask for CSV whenever possible**. For report-style Excel with merged cells, you have to carve out just the table part with options like `header=2` and `skiprows`.

### 3.6 Saving — to_csv / to_json

```python
df.to_csv("out.csv", index=False, encoding="utf-8-sig")
df.to_json("out.json", orient="records", force_ascii=False)
```

Forget `index=False` and a nameless number column sneaks into the file, so the next person who reads it gets an `Unnamed: 0` column. `force_ascii=False` keeps non-ASCII text in the JSON human-readable.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level04_loading_data
python3 main.py
```

main.py **deliberately creates** problematic files and then solves them with options, in four scenarios. All generated files land in the `outputs/` folder.

- **[1]** Saves 30 days of `hjh_data.sales_table` as a utf-8 CSV, reads it back with `read_csv`, and checks how the dtypes changed compared with before saving.
- **[2]** Hand-crafts a troublemaker CSV mixing "1,234"-style comma numbers, "-" as missing, and semicolon separators, then compares the wreckage of reading it with no options (every dtype collapses to object) against the recovery with `sep=";", thousands=",", na_values="-"`.
- **[3]** Creates a CSV exported by the chain's legacy Korean POS system (Korean store names, saved as cp949), shows the `UnicodeDecodeError` safely inside try/except when reading it as utf-8, and fixes it with `encoding="cp949"`.
- **[4]** Saves records-form JSON and reads it back with `pd.read_json`, confirming the JSON ↔ DataFrame round trip.

What to look for in the output: the before/after `dtypes` comparison in [2], and the actual face of the error message in [3]. When you meet this message at work, you should be able to greet it warmly — "ah, an encoding problem".

## 5. Try it yourself

1. **(Basic)** Read the CSV saved in [1] with the options `usecols=["date", "store", "revenue"]` and `nrows=10` to fetch just 10 rows and 3 columns. Hint: put both options into the same read_csv call.
2. **(Applied)** Suppose the troublemaker CSV from [2] also used the marker "none" for missing values. Create such a file yourself and handle both at once with `na_values=["-", "none"]`. Hint: na_values accepts a list.
3. **(Challenge)** Save the JSON from [4] with the default settings instead of `orient="records"`, open the file, compare how the structure differs, and write down which shape is better for handing to another system — and why. Hint: the default is a column-oriented (columns) structure.

## 6. Common mistakes

- **Fixing garbled text by hand** — open the file in Excel and save it, and the original is altered on the spot. Never touch the source; solve it with options on the reading side.
- **Omitting `index=False`** — to_csv writes the index into the file by default, so reading it back produces an `Unnamed: 0` column.
- **Skipping the dtype check** — compute right after reading and numbers-read-as-text silently corrupt the result. Checking `df.dtypes` right after reading is the inspection after customs clearance.
- **Reading zip codes or employee IDs as numbers** — "01234" turns into 1234 and the leading zero is gone. Protect them with `dtype={"zipcode": str}`.
- **Assuming an Excel-produced CSV is utf-8** — CSVs saved by localized versions of Excel are often in a legacy encoding (Korean Excel: usually cp949). Never assume "it's a CSV, so it must be utf-8".

## Next level preview

The data has cleared customs; now it is time to answer questions. We learn to answer business questions like "What are the top 10 weekend Coffee sales at the Downtown store?" with boolean masks, loc/iloc, and sorting.
