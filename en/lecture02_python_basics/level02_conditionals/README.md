# Lecture 02 · Level 02 — Conditionals: Making the Computer Decide

> if/elif/else is your company's approval policy translated into code: "if X, do A; otherwise if Y, do B; otherwise, C."
**Difficulty** ⭐ / **Prerequisites** level01 / **Estimated time** 45 min

## 1. Why learn this — the business angle

A large share of office work is "deciding by policy." Travel expenses up to
KRW 150,000 get the team lead's sign-off; above that, the division head must
approve. Inventory drops below the safety level — reorder. Customer grade is
VIP — extra discount. Have people make these calls and they tire out after
hundreds a day, and the criteria drift from person to person.

Learn conditionals and you can move the rulebook into code and **automate the
decision itself**. Write it precisely once and it judges 1 case or 100,000 by
the same standard, instantly. This level's exercise is exactly that: an
"automatic travel-expense approval judge." If you have used Excel's `IF`
function you already know half of this — and Python lets you write it far
more readably.

## 2. Understanding through analogies

**The approval-policy analogy.** Suppose the company expense policy reads:

> 1. If the amount is KRW 50,000 or less: auto-approve
> 2. Otherwise, if the amount is KRW 150,000 or less and there is a receipt: team lead sign-off
> 3. Otherwise, if the amount is KRW 500,000 or less: division head approval
> 4. If none of the above applies: reject and request justification

That document structure *is* a conditional. "If" — "otherwise if (elif)" —
"otherwise (else)". And one important rule: just as you read the policy from
the top and **stop at the first clause that applies**, Python checks from the
top, executes only the first branch that is true, and exits. A KRW 70,000
expense hits clause 2, and clauses 3 and 4 are never even read.

## 3. Core concepts

### 3.1 The basic if / elif / else skeleton

```python
if amount <= 50000:
    result = "Auto-approved"
elif amount <= 150000 and has_receipt:
    result = "Team lead sign-off"
elif amount <= 500000:
    result = "Division head approval"
else:
    result = "Rejected"
```

- A condition ends with a colon (`:`), and the lines to execute are written
  inside with 4 spaces of **indentation**. Python uses indentation to mark
  "how far this condition's jurisdiction extends." Indentation — not braces —
  being the syntax is a defining trait of Python.
- You can chain as many `elif` as you like; `else` comes last, at most once,
  and may be omitted.
- Checking proceeds from the top and **only the first true branch** runs.
  So the order of the conditions is the priority order of the policy.

### 3.2 Comparison operators — the tools that produce True/False

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `==` | equal | `grade == "VIP"` | True/False |
| `!=` | not equal | `status != "cancelled"` | True/False |
| `<` `<=` `>` `>=` | size comparison | `amount <= 150000` | True/False |

The result of a comparison is always a boolean — `True` or `False`.
**Never confuse `=` (assignment) with `==` (comparison).** Using `=` in a
condition is a syntax error.

### 3.3 Logical operators — combining conditions

Policies are usually combinations of several conditions.

- `and` : true only if both are true. "150,000 or less **and** has a receipt"
- `or` : true if either is true. "an executive **or** an overseas trip"
- `not` : flips true/false. "if there is **no** receipt" → `not has_receipt`

`and` is evaluated before `or`, so when mixing them, make your intent
explicit with parentheses. `(a or b) and c` and `a or (b and c)` are entirely
different policies.

### 3.4 Handy Python-only expressions

```python
if 50000 < amount <= 150000:     # chain a range comparison like math
if grade in ("VIP", "VVIP"):     # is it one of several values?
if has_receipt:                  # no need to write == True
```

`in` asks "is this value in the list?" and is especially useful for grade and
category checks.

### 3.5 Nesting and readability

You can nest an if inside an if, but stack three or more levels and the
rulebook becomes a maze. It reads better flat: combine conditions with `and`,
or filter out the exceptional cases first (early return — covered in
level05). The bar is: "could a new hire seeing this policy for the first time
understand it?"

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level02_conditionals
python3 main.py
```

An automatic travel-expense approval judge. Stage by stage:

- **[1] Single judgment**: walks one expense through the policy (auto-approve / team lead sign-off / division head approval / reject), condition by condition.
- **[2] Comparison & logic lab**: see the True/False of `amount <= 150000`, `and`, `or`, `not` with your own eyes.
- **[3] Why order matters**: judges with the same conditions in a different order and compares the results — reproducing the trap where "putting a loose condition first means the strict one never runs."
- **[4] Batch judgment**: processes 6 expense claims with one rule set and tallies approved / escalated / rejected counts. (A sneak preview of loops, next level.)

The heart of it is the if/elif/else ladder in `judge_expense`. Confirm that
the policy document and the code correspond almost 1:1, and that condition
order = policy priority.

## 5. Try it yourself

1. **Policy revision** — an order comes down: "lower the auto-approve limit
   from KRW 50,000 to KRW 30,000." Change one number in main.py and find which
   cases now get a different verdict. (Hint: find and change `50000`. A policy
   change being a one-character code edit is the charm of automation.)
2. **New clause** — add a clause with **top priority**: "overseas trips
   (`is_overseas` is True) require division head approval regardless of
   amount." (Hint: put `if is_overseas:` at the top of the ladder so it is
   checked before everything else.)
3. **Weekend judge** — write new code that judges a weekday string
   (`"Sat"` or `"Sun"` → "Closed", otherwise "Open"). (Hint: `in ("Sat", "Sun")`
   does it in one line, no `or` needed.)

## 6. Common mistakes

- **Confusing `=` and `==`** — `if grade = "VIP":` is a `SyntaxError`.
  Decisions always use `==`.
- **Inconsistent indentation** — 4 spaces on one line and 3 on another in the
  same block gives an `IndentationError`. Trust your editor's auto-indent, and
  never mix tabs and spaces.
- **Wrong condition order** — put `if amount <= 500000` at the top and even a
  KRW 50,000 claim gets "division head approval." Narrow (strict) conditions
  go on top, broad ones below.
- **Leaving a policy hole with `else`** — the final `else` means "none of the
  clauses above applied." Put "approve" there and every unforeseen case gets
  approved. The principle: `else` holds a safe default (reject, hold, or
  route to a person).
- **Comparing floats with `==`** — because of the level01 error,
  `0.1 + 0.2 == 0.3` is False. Use range comparisons or compare after `round`.

## Next level preview

Now that we have a decision rule, we need to apply it automatically to
hundreds or thousands of cases. In level03 we aggregate data on 100 customers
with loops (for/while) and feel, physically, the payoff of automation versus
hand-calculation.
