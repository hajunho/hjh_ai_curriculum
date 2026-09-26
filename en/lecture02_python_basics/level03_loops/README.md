# Lecture 02 · Level 03 — Loops: Delegating the Boring Work

> A for/while loop is the instruction "perform the same procedure for every item in the list" — the syntax where the payoff of automation is most tangible.
**Difficulty** ⭐ / **Prerequisites** level02 / **Estimated time** 45 min

## 1. Why learn this — the business angle

Imagine summing the purchases of 100 customers, computing the average, and
picking out the top customers. In Excel you would use SUM and AVERAGE, but
when a new file arrives every day, the format shifts slightly, and the
aggregation rules get complicated, the manual work piles up. When a person
processes 100 cases, the phone rings mid-way, focus breaks, and somewhere one
case gets missed.

A loop is a one-line instruction: "apply this procedure to every item in the
list." Whether it is 100 cases or a million, the code is the same and so is
the quality. Most of the practical benefit of learning to program —
automating repetitive work — comes from exactly this syntax. In this
exercise we aggregate data on 100 customers side by side, "by hand" and
"with a loop," and see the difference for ourselves.

## 2. Understanding through analogies

**The mail-merge analogy.** Suppose you hand a temp the job of mailing a new
product flyer to 100 customers. You would not write the instructions like
this:

> "Write Ms. Smith's address on an envelope and insert the flyer. Write Mr. Jones's address on an envelope and insert the flyer. …" (100 lines)

You would write this:

> "**For each customer on the list**: write the address on an envelope, insert the flyer, and stamp it."

That one line is a `for` statement. "For each customer on the list" =
`for customer in customers:`, and the indented procedure below it is the body
that runs each time.

A `while` statement is a slightly different instruction: "keep sticking
stamps **until you run out**" — repeat, not a known number of times, but
**as long as a condition holds**. If the count is fixed, use for; if a
condition decides the end, use while. That instinct is all you need.

## 3. Core concepts

### 3.1 The for statement — walking a list

```python
purchases = [120000, 45000, 230000]
total = 0
for amount in purchases:      # take items from the list one at a time into amount
    total = total + amount    # the body, executed each time
print(total)                  # 395000
```

`amount` is the **loop variable** — it changes to the next item every lap.
A variable like `total`, initialized to 0 before the loop and built up each
lap, is called an **accumulator**. It is the basic pattern of aggregation, so
get it into your fingers.

### 3.2 range — repeating by count

```python
for i in range(5):        # 0, 1, 2, 3, 4  (5 is NOT included!)
for i in range(1, 6):     # 1, 2, 3, 4, 5
for i in range(0, 10, 2): # 0, 2, 4, 6, 8  (step by 2)
```

In `range(start, stop, step)`, **the stop value is not included**. "Start at
0 and go up to just before the end" is the convention throughout Python. When
you need both the item and its position, use
`for i, item in enumerate(items):`.

### 3.3 The while statement — as long as a condition is true

```python
budget = 1000000
day = 0
while budget > 0:          # repeat while budget remains
    budget -= 87000        # daily spend
    day += 1
```

A while loop's body must change state so the condition **eventually becomes
false**. If it never does, you have an infinite loop. (Force-quit with Ctrl+C
in the terminal.)

### 3.4 break and continue — flow control

- `break` : ends the loop immediately and exits. "Found it — no need to look further."
- `continue` : skips just this lap and moves to the next item. "Skip the missing value and keep going."

```python
for row in rows:
    if row is None:
        continue          # skip broken data
    if row == target:
        break             # found it, end the loop
```

Real-world data always contains blanks and contaminated values, so the
continue-to-filter pattern comes up constantly.

### 3.5 Nested loops

Put a loop inside a loop and you get a grid traversal like "every store x
every month." The inner loop runs completely for each lap of the outer loop,
so 100 x 100 means the body executes 10,000 times. Convenient — but also the
spot that slows down as data grows (lecture03 improves this with vectorized
operations).

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level03_loops
python3 main.py
```

An example that aggregates purchase data for 100 customers. The random seed
is fixed, so everyone who runs it gets the same data.

- **[1] The limits of hand-calculation**: mimics, for just 3 customers, what code looks like without a loop — `total = c1 + c2 + c3 + ...`. Confirms that 100 customers would need 100 lines.
- **[2] for aggregation**: four lines of loop compute the total, the average, and the top customer among 100.
- **[3] Combining with conditions**: puts level02's if inside the loop to count only "top customers with 300k+ KRW," using `continue` to skip dormant customers (0 KRW).
- **[4] while simulation**: subtracts daily spending from a 1,000,000 KRW marketing budget until it runs out, counting "how many days it lasts." A 60-day cap is enforced with `break`.
- **[5] range and enumerate**: prints the top 5 purchasers with rank numbers.

The "line count comparison" at the end of the output is this level's
conclusion: aggregating 100 customers takes 100+ lines by hand, 4 lines with
a loop.

## 5. Try it yourself

1. **Change the threshold** — raise the top-customer bar from 300,000 to
   500,000 KRW and see how many remain. (Hint: change the single
   `VIP_THRESHOLD` value.)
2. **Bucketed counts** — split purchases into three bands — under 100k /
   100k–300k / 300k and above — and count each band. (Hint: put if/elif/else
   inside the loop and keep 3 accumulators.)
3. **Times-table report** — print the multiplication tables from 2 to 9 with
   a nested loop. (Hint: outer loop `for dan in range(2, 10):`, inner loop
   `for n in range(1, 10):`)

## 6. Common mistakes

- **Initializing the accumulator inside the loop** — put `total = 0` inside
  the for and it resets to 0 every lap, leaving only the last item. Initialize
  **before** the loop, always.
- **Misreading range's stop value** — `range(1, 100)` goes up to 99. To loop
  100 customers, use `range(1, 101)` or `range(100)`.
- **Infinite loops** — forget to update the condition variable in a while
  body and it spins forever. Before writing a while, ask yourself: "when does
  this condition become false?"
- **Modifying the list you are iterating** — delete items from a list while
  walking it and some items get skipped. Collect results into a new list
  instead (level04, level10).
- **Wrong indentation, wrong ownership** — a print that belongs inside the
  loop but sits outside prints only the final value, once. Check with
  indentation: "does this line run every lap?"

## Next level preview

So far our variables held a single value each. To hold 100 customers' data we
need containers that group many values. In level04 we learn the four
containers — lists, tuples, dictionaries, and sets — when to use each, and
build a mini inventory-management system.
