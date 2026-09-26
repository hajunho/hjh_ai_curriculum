# Lecture 04 · Level 08 — Modifying Data and Transactions

> Learn to change data with INSERT · UPDATE · DELETE, and to bundle several changes into "all succeed or all cancelled" with transactions (BEGIN/COMMIT/ROLLBACK). A stock-deduction scenario demonstrates both success and failure-with-rollback.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level07 / **Estimated time** 50 min

## 1. Why this matters — the business view

The moment someone who only read data starts **changing** it, the weight of
responsibility changes. One UPDATE line can change the grade of 200,000
customers. That's why this level teaches the syntax (INSERT/UPDATE/DELETE)
together with the **safety gear**. Real work — processing an order, deducting
stock, crediting loyalty points — comes as bundles of several changes. What if
one in the middle fails? Data where "the order saved but the stock never went
down" comes back at you as an accounting audit and customer complaints. A
transaction (transaction) is the mechanism that guarantees the bundle is
**all-success or never-happened**, and it is the beating heart of every bank,
online store, and ERP system.

## 2. Understanding by analogy

**A bank transfer** is the best analogy. Taking 50,000 out of my account
(change 1) and putting 50,000 into yours (change 2) must be a single body. If
the withdrawal succeeds and the deposit fails, the money evaporates. So the
bank puts both actions in one envelope, seals the envelope only when both
succeed (COMMIT), and shreds the whole envelope if either fails (ROLLBACK).

Another analogy: **writing in pencil, inking over it later.** Changes after
BEGIN are pencil notes. They look changed to me, but they're not the original
ledger yet. COMMIT is inking them in for good; ROLLBACK is erasing them all.
Even in a heart-stopping moment like "I forgot the WHERE on my UPDATE!" —
before the commit, you still have the eraser.

## 3. Core concepts

### 3-1. INSERT — adding rows

```sql
INSERT INTO customers (customer_id, name, city, grade, joined_at)
VALUES (204, 'Paige Winters', 'New York', 'BASIC', '2025-12-20');
```

Make a habit of spelling out the column list. If the table's structure changes
later, the statement doesn't break. A duplicate primary key gets refused by
the DB, just as you saw in level00 — the safety gear at work.

### 3-2. UPDATE — modifying rows

```sql
UPDATE customers
SET    grade = 'GOLD'
WHERE  customer_id = 204;
```

**The WHERE is the life of an UPDATE.** Without it, every row changes. The
working rule: **run a SELECT with the same WHERE first** to see how many rows
it catches, then fire the UPDATE. Aim → shoot, in that order.

### 3-3. DELETE — removing rows

```sql
DELETE FROM customers WHERE customer_id = 204;
```

Again, the WHERE is everything. In practice, data often isn't deleted at all —
just flagged, like `status='closed'` (a soft delete). If the data needs an
audit trail, check the policy before any DELETE.

### 3-4. Transactions — BEGIN / COMMIT / ROLLBACK

```sql
BEGIN;                       -- open the envelope (pencil notes from here on)
UPDATE inventory SET stock = stock - 2 WHERE product_id = 1;
INSERT INTO orders ...;
COMMIT;                      -- both succeeded -> ink it in
-- if something went wrong: ROLLBACK;  -> shred the envelope, back to before BEGIN
```

The guarantees a transaction provides are called ACID. Two are worth
memorizing. **Atomicity** — all or nothing. **Consistency** — no
rule-breaking half-state ever lands in the ledger. (The remaining two,
isolation and durability, guarantee "many people can write at once, and even a
power cut is fine.")

### 3-5. The verify-then-commit pattern

The skeleton of a real-world transaction:

```
BEGIN
  change 1, change 2, ...
  verify: did stock go negative? is the balance short?
  no problem -> COMMIT / problem found -> ROLLBACK
```

"Deduct first, cancel everything if it goes negative" is possible precisely
because everything before the commit is pencil. This exact pattern appears in
main.py's step [5].

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

main.py starts by creating a stock table, `inventory`, alongside products.

- **[1] INSERT**: add a new customer — check the row counts before and after,
  and the new row.
- **[2] UPDATE**: demonstrates the aim (confirm the target with SELECT) →
  shoot (UPDATE) sequence.
- **[3] Reenacting the missing-WHERE UPDATE accident**: inside a transaction,
  a WHERE-less UPDATE deliberately runs, turning every customer into a VIP —
  and ROLLBACK restores everything. The moment the eraser works.
- **[4] Stock-deduction transaction (success)**: order creation + stock
  deduction go into one envelope, verification passes → COMMIT. Confirm the
  stock actually went down.
- **[5] Stock-deduction transaction (failure → rollback)**: order more than
  the stock on hand → verification after deduction finds negative stock →
  ROLLBACK. Confirm **both the orders and the stock are exactly as before the
  attempt**. The key output is that no "half-processed state" exists.

## 5. Try it yourself

1. **(Easy)** Mimic [1]: INSERT a customer with your own name, then INSERT
   once more with the same customer_id and watch the DB refuse.
   *Hint: wrap it in try/except sqlite3.IntegrityError to see the error
   message gracefully.*
2. **(Medium)** "Promote every Boston customer to SILVER" using the aim→shoot
   pattern. Before the UPDATE, check the target count with SELECT COUNT(*).
   *Hint: the point is making the WHERE identical in both statements.*
3. **(Challenge)** Build the bank-transfer scenario yourself. Create an
   accounts(name, balance) table and write a transaction that transfers A→B
   but ROLLBACKs if the balance is short.
   *Hint: carry [5]'s structure (deduct → verify → commit/rollback) straight
   over to accounts.*

## 6. Common mistakes

- **UPDATE/DELETE without WHERE**: the most common accident of all. The
  "aim (SELECT) first" rule plus working inside a transaction is your double
  seatbelt.
- **Forgetting to commit**: in Python's sqlite3, closing the connection after
  changes without commit() throws the changes away. The classic cause of "my
  UPDATE isn't showing up."
- **Holding a transaction open too long**: while your envelope is open, other
  users' work may be kept waiting. Open the envelope briefly, close it fast —
  it's good manners.
- **Not ROLLBACKing on error**: rollback on failure may not be automatic. Call
  rollback() explicitly in your try/except.
- **Practicing on the production DB**: practice always happens on a copy. An
  environment you can simply rebuild — like this practice DB — is your
  scratchpad.

## Next level preview

When the data grows to hundreds of thousands of rows, "the right answer" isn't
enough — you need "the fast answer." In **level09 — Indexes and Query
Performance** you'll see how an index works, EXPLAIN QUERY PLAN, and a
measured before/after speed difference.
