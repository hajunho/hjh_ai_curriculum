# Lecture 03 · Level 08 — Merging and Joining (merge · concat)

> Sales table + store info table + targets table: we stitch scattered tables together to produce the answer called "target achievement rate".

**Difficulty** ⭐⭐⭐ / **Prerequisites** level07_groupby_aggregation / **Estimated time** 45 min

## 1. Why learn this — the business view

Almost no analysis ends with a single table. Sales live in the POS system, store information (region, opening year, manager) sits in the facilities team's roster, and this year's targets are in a planning-team document. One perfectly ordinary request — "show me target achievement by store, broken down by region" — already needs three tables.

In Excel this job is done with VLOOKUP. But VLOOKUP drags over one column at a time, silently takes only the first match when keys repeat, and scatters #N/A wherever it fails. Pandas' `merge` joins whole tables in one motion and even provides tools to **verify** unmatched and duplicate keys. It is the same concept as a database JOIN, so the intuition you build here carries straight into lecture04's SQL.

Merging is as accident-prone as it is powerful. Reports where rows silently multiplied or vanished are born from merge mistakes. Half of this level is about dodging those traps.

## 2. Understanding through an analogy

Think about assembling an alumni-reunion list. You hold two lists. **A: the yearbook roster** (name, homeroom) and **B: the contact list** (name, phone number). There are four ways to combine them, keyed on the name.

- **inner (intersection)** — only people on both lists. You keep just the "confirmed, reachable alumni".
- **left (keyed on the left)** — everyone from the yearbook stays; people without a contact get blanks. "The full roster + the numbers we know."
- **right (keyed on the right)** — everyone on the contact list. The mirror image of left.
- **outer (union)** — everyone who appears on either list. It surfaces "people not in the yearbook but in the contacts" (a transfer student?), which makes it great **for finding data inconsistencies**.

And one trap: what if "John Smith" appears **twice** on the contact list? The combined list ends up with two John Smith rows. Twice in the yearbook and twice in the contacts makes 2×2=4 rows. This is merging's biggest accident: the **duplicate-key explosion**.

`concat` is an entirely different job. It is **stacking homeroom 2's list below homeroom 1's** — piling downward, not joining sideways (merge).

## 3. Core concepts

### 3-1. The basic merge

```python
pd.merge(left_df, right_df, on="store", how="left")
```

- `on` : the key column that shares its name across both tables. If the names differ, specify separately: `left_on="store"`, `right_on="store_name"`.
- `how` : the join type. `inner` (the default), `left`, `right`, `outer`.

### 3-2. Choosing among the four how's

| how | Rows kept | Typical situation |
|---|---|---|
| inner | keys present on both sides | computing with only the confirmed data |
| left | all of the left table | **when there is a designated base table (the everyday default)** |
| right | all of the right table | symmetric with left (usually you swap the tables and use left) |
| outer | keys present on either side | auditing the mismatch between two tables |

The working intuition: when the base is clear — "**attach** store info to the sales table" — use left; when you are **auditing** whether two rosters agree, use outer.

### 3-3. indicator=True — make merge verification a habit

With `pd.merge(..., how="outer", indicator=True)` you get a `_merge` column marking where each row came from: `both` / `left_only` / `right_only`. One line of `value_counts()` instantly audits "how many keys have no partner". Make this check a post-merge habit and you prevent most of the accidents.

### 3-4. The duplicate-key trap — row-count explosion

If the same key appears k times in the right table, each matching left row multiplies k-fold (a Cartesian product). If Downtown accidentally has two rows in the targets table, its revenue gets **counted twice** after the merge and the total is inflated. Two preventive measures:

1. Before merging, check duplicates with `right_df["store"].duplicated().sum()`, and `drop_duplicates()` if needed.
2. Compare `len()` before and after the merge — if the row count is not what you expected, stop immediately and investigate.

### 3-5. concat — stacking

```python
pd.concat([first_half, second_half], ignore_index=True)
```

It stacks tables of the same structure vertically (combining 12 monthly files, and so on). Without `ignore_index=True` the original indexes remain and duplicate like 0,1,2,...,0,1,2,.... For reference, DataFrame also has a `join` method — a shorthand for merge keyed on the index, so the concept is identical.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture03_data_handling/level08_merge_join
python3 main.py
```

Scenario: merge the (aggregated) sales table + store info table + targets table into a ranking of target achievement. The store info table is deliberately **missing the Airport store**, and instead includes the brand-new **"Parkview" store** that has no sales yet.

- **[1] Per-store performance** — builds the annual total revenue table with level07's groupby (5 rows).
- **[2] Preparing the side tables** — creates store_info (4 known stores + Parkview) and targets as dicts.
- **[3] Comparing how's** — merges the same two tables with inner/left/outer and watches the row count change 4/5/6, then uses `indicator` to spot `left_only` (Airport) and `right_only` (Parkview).
- **[4] Achievement rate** — merges the targets, computes `achievement = actual/target`, and prints the descending ranking.
- **[5] Duplicate-key trap demo** — injects a duplicate Downtown row into the targets table, merges, and watches the rows grow 5→6 with the total inflated — then prevents it with `drop_duplicates`.
- **[6] concat** — stacks the H1/H2 aggregates vertically to contrast with merge.

The heart of the code is the audit pattern in [3].

```python
audit = pd.merge(perf, store_info, on="store", how="outer", indicator=True)
print(audit["_merge"].value_counts())
```

## 5. Try it yourself

1. **Achievement by region** — merge store_info's region into the result of [4] and compute the "average achievement rate per region". (Hint: after the merge, `groupby("region")["achieve_pct"].mean()`.)
2. **Category targets** — build your own per-category targets table (Coffee 1.2 billion, Beverage 0.9 billion, ...) and compute achievement per category. (Hint: the same flow as [1]~[4], just with category as the key.)
3. **The validate argument** — apply `pd.merge(..., validate="one_to_one")` to the duplicate situation in [5]. What happens? (Hint: it throws an error when duplicates exist — a safety device that turns a "silent accident" into a "loud error".)

## 6. Common mistakes

- **Omitting how and not knowing the default is inner** — unmatched rows vanish without a sound. Always compare row counts before and after a merge.
- **Inconsistent spellings in the key column** — `"Downtown"` vs `"Downtown "` (whitespace), `"DOWNTOWN"` vs `"downtown"` (case) are different keys. Normalize with `str.strip()` and friends before merging.
- **Leaving duplicate keys in place** — the #1 cause of row explosions and double counting. Check with `duplicated()` or use the `validate=` argument.
- **Using concat where merge belongs (or vice versa)** — attaching columns sideways is merge; stacking rows downward is concat. Concatenating sideways aligns by index and can shuffle things into the wrong places.
- **Ignoring suffixes** — when both tables have a non-key column of the same name, `_x` and `_y` get appended. Being explicit, like `suffixes=("_actual", "_target")`, keeps it clear which side is which.

## Next level preview

Now that you can join tables, it is time to handle the time axis. In level09 we convert dates into real datetime values (to_datetime) and compute weekly aggregates (resample), moving averages (rolling), and week-over-week growth (shift).
