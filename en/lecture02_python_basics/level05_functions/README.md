# Lecture 02 · Level 05 — Functions: Reusable Work Procedures

> A function names a frequently used procedure so you can delegate — "handle this for me." You hand over the work as arguments and get the result back as a return value.
**Difficulty** ⭐⭐ / **Prerequisites** level04 / **Estimated time** 50 min

## 1. Why learn this — the business angle

Suppose the VAT calculation logic is copy-pasted in six places across the
company's code. When the tax rate changes, you have to find and fix all six —
miss even one and different departments start computing different taxes. A
large share of real production bugs are exactly this: "copy-paste logic
drifting out of sync."

A function defines logic **in one place** and lets you call it by name
wherever needed. When the rate changes, you fix one line inside the function
and every caller is updated. Better still, a name like `calc_vat(total)` is
its own documentation — the code reads like a work procedure manual. In this
exercise we run "before refactoring" code, plastered with copied tax/discount
logic, side by side with the "after" version organized into functions, and
compare.

## 2. Understanding through analogies

**The delegation analogy.** Picture a team lead who doesn't do expense
reconciliation herself but hands it to the specialist.

> Team lead: "Could you reconcile this stack of receipts (input)?"
> Specialist: (processes it at their own desk, their own way) "The total comes to 320,000 KRW (output)."

Three things matter here.

1. **The name** — you only need to know the role, "reconciliation specialist." → the function name
2. **The input** — you hand over the stack of receipts. → the arguments
3. **The output** — you get the resulting amount back. → the return value

And crucially: the team lead does not need to know **how** the specialist
calculates. The specialist's desk (the variables inside the function) is
their own territory and never mixes with the team lead's desk. That is
**scope**. Just as a well-delegated organization keeps desks separate, a good
function talks to the outside world only through arguments and return values.

## 3. Core concepts

### 3.1 Defining and calling

```python
def calc_vat(amount):          # definition: merely registers "this procedure exists"
    return int(amount * 0.1)   # return: hand the result back to the caller

vat = calc_vat(50000)          # call: the moment work actually happens (-> 5000)
```

`def` only defines; it does not execute. The body runs only when you **call**
the function with parentheses after its name. On hitting `return`, the
function ends immediately and comes back carrying the value. A function
without `return` gives back `None`.

### 3.2 Arguments — several, by name, with defaults

```python
def calc_discount(amount, grade, rate=0.05):     # rate has a default value
    ...

calc_discount(100000, "VIP")                     # rate uses the default 0.05
calc_discount(100000, "VIP", rate=0.1)           # specify only when needed
calc_discount(amount=100000, grade="VIP")        # keyword args: order-free, self-explanatory
```

A default argument is like "absent special instructions, follow the standard
policy." **Arguments with defaults must come after those without.** Keyword
arguments — naming arguments at the call site — dramatically improve
readability once you pass more than three.

### 3.3 Return values — returning several at once

```python
def summarize(sales):
    return sum(sales), len(sales), max(sales)    # returned bundled as a tuple

total, count, best = summarize(data)             # received via unpacking
```

Level04's tuple unpacking shines here. Results that come as a set — "total,
count, max" — can be returned in one go.

### 3.4 Scope — variables inside a function stay inside

```python
def process():
    temp = 100        # local variable: exists only on this function's desk

process()
print(temp)           # NameError! invisible from outside
```

Variables created inside a function vanish when it ends (local variables).
You *can* read variables outside the function (globals) from within, but
**do not get into the habit of modifying them.** You end up with code where
nobody can trace where a value changed. Take what you need as arguments,
return the result — that is the "everyone keeps their own desk" principle.

### 3.5 What makes a good function

- It does **one thing**. A "compute tax plus save file plus send email"
  function gets split into three.
- Its **name is a verb phrase**: `calc_vat`, `apply_discount`, `format_receipt`.
- A **docstring** (the explanatory string on the first line) states purpose,
  arguments, and return in a line or two.
- The moment you find yourself **copying the same code a second time** is
  the moment to extract a function (the DRY principle: Don't Repeat Yourself).

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level05_functions
python3 main.py
```

We compute the payment (grade discount + VAT) for 3 orders, two ways.

- **[1] Before refactoring**: the discount and tax logic is copy-pasted per order. Twenty-odd lines repeat, and one copy carries a **deliberately planted inconsistency bug** (a tax-rate typo), so one order's result is subtly off.
- **[2] After refactoring**: the same calculation via three functions — `get_discount_rate`, `calc_vat`, `calc_payment`. With the logic in one place, inconsistency is impossible by construction.
- **[3] Before/after comparison**: puts both sets of results side by side and prints which order diverged and by how much.
- **[4] Function syntax recap**: short demos of default arguments, keyword arguments, multiple returns, and scope (reproducing a NameError).

Also note how `calc_payment` calls two other functions internally — the
specialist delegating to sub-specialists. Functions stack into layers like
this.

## 5. Try it yourself

1. **Tax-change drill** — suppose VAT rises from 10% to 12%. Count how many
   lines you must change in the function version in [2]. How many would it be
   in the copy-paste version in [1]? (Hint: the function version is one line
   inside `calc_vat`.)
2. **Add a grade** — add the rule "VVIP gets 15% off" to
   `get_discount_rate`, create a VVIP order, and verify. (Hint: add one
   condition at the top of the if ladder.)
3. **Receipt function** — write `format_receipt(name, amount)` that takes a
   name and amount and **returns** the string "NAME's payment: N KRW". Do not
   print — return. (Hint: a returning function is free to reuse anywhere —
   screen output, file save, email body.)

## 6. Common mistakes

- **Confusing print with return** — print only shows things on screen; it
  returns nothing. If `result = show_total(...)` holds `None`, the function
  printed instead of returning. Split the roles: "calculate with return,
  report with print."
- **Forgetting the call parentheses** — `calc_vat` alone is the function
  itself (an object); `calc_vat(50000)` is what executes it.
- **Argument order mix-ups** — `calc_discount("VIP", 100000)` with the order
  swapped computes nonsense. When in doubt, call with keyword arguments.
- **Modifying globals** — sneakily changing an outer variable inside a
  function makes bug-hunting a nightmare. Take arguments, return results.
- **A list as a default value** — `def f(items=[])` is a famous trap where
  calls share one list. Use `def f(items=None)` and inside the function do
  `if items is None: items = []`.

## Next level preview

Time to store our results somewhere that survives after the program ends — a
file. In level06 we read and write text and CSV with open/with, and build an
exercise that auto-generates a summary report from a transactions CSV.
