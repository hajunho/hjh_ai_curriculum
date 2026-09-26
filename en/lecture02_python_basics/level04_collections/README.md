# Lecture 02 · Level 04 — Lists, Tuples, and Dictionaries

> Four containers for holding many values — the ordered list, the unchangeable tuple, the labeled drawer cabinet (dictionary), and the no-duplicates pouch (set) — and how to choose between them.
**Difficulty** ⭐⭐ / **Prerequisites** level03 / **Estimated time** 50 min

## 1. Why learn this — the business angle

Real-world data arrives in bundles, not single values: this month's list of
sales records, a per-product inventory table, a list of vendor tax IDs. Which
container you put a bundle in completely changes your code's length, speed,
and bug rate.

For example, "look up stock quantity by product code" written with a list
means scanning the whole thing every time; written with a **dictionary**, it
is a single grab. "De-duplicate this week's visitors" is one line with a
**set**. Container choice looks trivial, but it is the same problem as how
you design a table in Excel — it determines your work efficiency. In this
level we use a mini inventory-management example to try all four containers
and internalize the selection criteria.

## 2. Understanding through analogies

Let's use office supplies management as the analogy.

- **List = the goods-received ledger.** Entries are written line by line in
  arrival order. Order matters, the same item can appear multiple times, and
  lines can be added, edited, and deleted.
- **Tuple = the laminated spec sheet.** A bundle like "A4 paper, 500 sheets,
  25,000 KRW" that never changes once set. Being unmodifiable is not a
  weakness — it is a **safety guard: you can't change it even by accident**.
- **Dictionary (dict) = the labeled drawer cabinet.** Open the "ballpoint pen"
  drawer and out come 37 units of stock. Know the drawer's label (the key) and
  you grab the contents (the value) instantly, no top-to-bottom searching.
- **Set = the pouch that refuses duplicates.** Put the same thing in twice and
  only one remains. Use it when all that matters is **membership and
  distinct kinds**, like "the kinds of items ordered this week."

## 3. Core concepts

### 3.1 Lists — ordered sequences

```python
sales = [45000, 120000, 89000]
sales.append(56000)       # add to the end
sales[0]                  # first item (numbering starts at 0!)
sales[-1]                 # last item
sales[1:3]                # slice: from index 1 up to (not including) 3
len(sales), sum(sales), sorted(sales)   # count, total, sorted copy
```

Remember: indexes start at 0, and a slice's end index is not included (the
same rule as range).

### 3.2 Tuples — unchangeable bundles

```python
product = ("P001", "Ballpoint pen", 1200)   # (code, name, unit price)
code, name, price = product                  # unpacking: take all at once
```

Once created, a tuple's items cannot be changed (it is immutable). Use tuples
for "values that travel as a set" — coordinates, settings, a function's
multiple return values. When level03 did `for name, qty in orders:`, that was
tuple **unpacking**.

### 3.3 Dictionaries — finding values by key

```python
stock = {"Ballpoint pen": 37, "A4 paper": 12, "Stapler": 4}
stock["Ballpoint pen"]        # 37 (instant lookup by key)
stock["Highlighter"] = 20     # add a new entry
stock.get("Eraser", 0)        # returns 0 if missing (avoids KeyError)
for item, qty in stock.items():   # iterate key-value pairs
    ...
```

Looking up a missing key with square brackets raises a `KeyError`. If you are
not sure the key exists, check with `in` or use `get(key, default)` — that is
the professional habit.

### 3.4 Sets — de-duplication and membership tests

```python
visited = {"P001", "P002", "P001"}    # -> {"P001", "P002"} duplicates removed
"P001" in visited                      # membership test, very fast
a & b, a | b, a - b                    # intersection, union, difference
```

List-matching chores like "customers on both lists" (intersection) or "items
ordered but not yet received" (difference) finish with a single operator.
One caveat: sets have no order.

### 3.5 Selection criteria, summarized

| Situation | Container | Why |
|---|---|---|
| Ordered records, needs add/edit | list | receiving ledger |
| A bundle of values that must not change | tuple | spec sheet |
| Fast lookup by name (key) | dictionary | drawer cabinet |
| De-duplication, membership, list matching | set | pouch |

And the standard shape of real-world data is a **"list of dictionaries."**
One row is a dictionary; the whole table is a list — an Excel sheet moved
straight over — and it leads naturally into pandas in lecture03.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level04_collections
python3 main.py
```

A mini inventory-management example for a stationery wholesaler.

- **[1] Lists**: builds today's receiving records in a list and demos indexing, slicing, and sorting.
- **[2] Tuples**: keeps the product spec sheet (code, name, unit price) as a tuple and safely shows that an edit attempt errors out (wrapped in try, so the program survives).
- **[3] Dictionaries**: builds a product-name → stock-quantity drawer cabinet, applies inbound/outbound moves, and looks up a missing product safely with `get`.
- **[4] Sets**: matches this week's ordered items against received items and finds "ordered but not yet received" via set difference.
- **[5] Putting it together**: builds the inventory table as a "list of dictionaries" and prints a reorder report for items below safety stock, using a loop.

The data shape in [5] — `[{"name": ..., "qty": ..., ...}, {...}]` — is the
standard form you will meet in every data lecture from here on, so let it
sink in.

## 5. Try it yourself

1. **Add stock** — add 15 "Scissors" to the drawer cabinet in [3], and ship
   out (subtract) 10 "Ballpoint pen". (Hint: `stock["Scissors"] = 15`,
   `stock["Ballpoint pen"] -= 10`)
2. **List matching** — create one more set for last week's ordered items and
   find "items ordered two weeks in a row." (Hint: intersection `&`)
3. **Sort by scarcity** — print the inventory table in [5] sorted by lowest
   quantity first. (Hint: `sorted(inventory, key=lambda row: row["qty"])`.
   lambda is covered properly in level10; for now just typing it along is fine.)

## 6. Common mistakes

- **Indexes start at 0** — `items[1]` is the *second* item. For the first,
  use `items[0]`.
- **Looking up a missing key** — `stock["Eraser"]` halts the program with a
  `KeyError` if there is no eraser. Use `stock.get("Eraser", 0)` or
  `if "Eraser" in stock:`.
- **The list-copy illusion** — `b = a` is not a copy; it sticks a second
  label on the same list. `b.append(...)` changes a too. A real copy is
  `b = a.copy()`.
- **Trying to modify a tuple** — `product[2] = 1500` is a `TypeError`. If a
  value needs to change, put it in a list or dictionary in the first place.
- **Expecting order from a set** — sets do not preserve insertion order. If
  you need order, get a sorted list with `sorted(the_set)`.

## Next level preview

We can now store data, loop over it, and make decisions. But once the same
calculation logic starts getting copy-pasted around the codebase, maintenance
hell opens up. In level05 we learn to name logic with **functions** and reuse
it — organizing code like a work procedure manual.
