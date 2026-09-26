# Lecture 02 · Level 01 — Variables and Data Types

> A variable is a labeled box you put data in, and a data type is the kind of data inside the box (integer, float, string, boolean).
**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 40 min

## 1. Why learn this — the business angle

When you write `=B2*C2` in Excel, B2 and C2 are the addresses of the cells
holding "unit price" and "quantity." But give it a few weeks and you forget
what B2 was. In Python, instead of an address, you can give the cell a
**meaningful name** like `unit_price` or `quantity`. That is a variable.

Variables and types are the raw material of all code. Sales calculations,
customer list cleanup, even training an AI model — all of it is a chain of
"put a value in a variable → process it → put the result in another
variable." Understanding types precisely also prevents one of the most common
accidents at work: mixing up the number `1000` with the text `"1000"` and
getting a nonsense total.

## 2. Understanding through analogies

**The labeled-box analogy.** Imagine boxes in the office storage room, each
with a label: "March sales", "customer list", "VAT rate". When you need
something, you find the box by its label, take out the contents, and you can
also swap in new contents.

- `march_sales = 12000000` → put 12,000,000 into the box labeled "March sales".
- `print(march_sales)` → open the box and check what is inside.
- `march_sales = 13000000` → replace the contents with a new value. (That is why it is called a *variable* — it can vary.)

**The data type is the kind of contents inside the box.** What you can do
depends on whether the box holds cash, documents, or an "approved"/"rejected"
stamp. You can add cash to cash, but "add cash to a document" makes no sense.
Python is the same: ask it to compute "number + text" and it raises an error.

## 3. Core concepts

### 3.1 Creating a variable — assignment

```python
unit_price = 4500      # "put 4500 into the unit_price box"
```

`=` is not the mathematical "equals" — it is a command: **"put the value on
the right into the box named on the left."** That is why
`count = count + 1` (take the current value, add 1, put it back) works.

Naming rules: lowercase letters and underscores, chosen so the meaning shows.
`total_revenue` and `vat_rate` beat `a` and `x1`. Code is read far more often
than it is written.

### 3.2 The four basic data types

| Type | Name | Example | Business use |
|---|---|---|---|
| `int` | integer | `120`, `-3` | quantities, headcount, counts |
| `float` | floating point | `0.1`, `4500.0` | rates, unit prices, averages |
| `str` | string | `"Americano"` | names, addresses, notes |
| `bool` | boolean | `True`, `False` | approved or not, member or not |

You can check a type with `type(value)`. Whenever in doubt, just ask.

### 3.3 Type conversion

Values read from a CSV file or user input are always strings. To do math,
you must convert them.

```python
int("1000")      # string -> integer 1000
float("3.3")     # string -> float 3.3
str(4500)        # integer -> string "4500"
```

`"1000" + "500"` gives `"1000500"` (string concatenation). This is the number
one accident at work, so always stay conscious of "is this value currently a
number or text?"

### 3.4 f-strings — building report sentences

Put an `f` before a string and you can slot variables inside curly braces.

```python
name = "Americano"
total = 9000
print(f"{name} revenue is {total} KRW")      # Americano revenue is 9000 KRW
print(f"{total:,} KRW")                      # 9,000 KRW  (thousands separator)
print(f"{0.0525:.1%}")                       # 5.2%       (percentage, 1 decimal)
```

The `{variable:format}` pattern is something you will use constantly when
producing numeric reports. Memorize just three for now: `,` (thousands
separator), `.2f` (two decimal places), `.1%` (percentage).

### 3.5 Arithmetic operators at a glance

The calculation symbols are almost identical to Excel; only two are new.

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `+` `-` `*` `/` | basic arithmetic | `7 / 2` | `3.5` |
| `//` | quotient (floor division) | `7 // 2` | `3` |
| `%` | remainder | `7 % 2` | `1` |
| `**` | exponent | `2 ** 10` | `1024` |

`//` and `%` come up in allocation problems like "moving 17 people with
5-seat cars — how many cars, and how many people are left over?" Also
remember: the result of `/` is always a float.

### 3.6 The tiny inaccuracy of floats

Computers store decimals in binary, so `0.1 + 0.2` is not exactly 0.3 but
`0.30000000000000004`. This is not a bug — it is a limitation of how numbers
are stored. When comparing monetary amounts, use rounding (`round`); for
precise accounting, computing in whole integer units (e.g., whole won) is
safest.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level01_variables_types
python3 main.py
```

The example calculates a cafe's daily revenue. Stage by stage:

- **[1] Creating variables**: put the menu name, unit price, and cups sold into variables and check their types with `type()`.
- **[2] Calculation**: revenue = unit price x quantity, then VAT and the total. Reconfirms that `=` means "put it in."
- **[3] Type conversion**: compares adding the string `"120"` as-is versus after `int()` conversion, reproducing the "text number" accident.
- **[4] f-string report**: prints a human-friendly revenue summary using thousands separators and percentage formatting.
- **[5] Float error**: sees the `0.1 + 0.2` problem firsthand and fixes it with `round`.

The key line is `revenue = unit_price * cups_sold`. It does the same job as
Excel's `=B2*C2`, but notice how the names make the formula self-explanatory.

## 5. Try it yourself

1. **Add a menu item** — add cafe latte variables (unit price 5000 KRW, 80
   cups sold) to main.py and include it in the total. (Hint: create a
   `latte_revenue` variable and add it into the `day_total` calculation.)
2. **Formatting practice** — change the total output to look like
   "Total revenue: 1,234,567 KRW (VAT included)". (Hint: `f"{day_total:,} KRW"`)
3. **Type quiz** — what is `type(10 / 2)`? Predict first, then print to check.
   (Hint: the result of `/` is always the same type, even when it divides evenly.)

## 6. Common mistakes

- **Confusing `=` and `==`** — `=` is assignment (put in), `==` is comparison
  (are they equal?). Revisit this when comparison appears in level02.
- **Doing math on string numbers** — `"100" * 2` is not 200 but `"100100"`.
  Convert external values with `int()`/`float()` first.
- **Typos in variable names** — write `total_revenu` instead of
  `total_revenue` and you get
  `NameError: name 'total_revenu' is not defined`. The misspelled name appears
  right in the error message, so read it carefully and the fix is immediate.
- **Meaningless names** — `a`, `b`, `data2` are ciphers to the you of next
  week. Prefer longer names whose meaning is visible.

## Next level preview

Now that values live in variables, it is time to make the computer
**decide**. In level02 we build an "automatic travel-expense approval judge"
with `if/elif/else` conditionals and learn branching properly.
