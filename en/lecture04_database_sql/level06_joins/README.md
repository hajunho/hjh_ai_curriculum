# Lecture 04 · Level 06 — JOIN: Connecting Multiple Tables

> Learn JOIN, which reattaches split tables by key. Covers the difference between INNER JOIN and LEFT JOIN, how to choose join keys, and the fan-out trap that inflates aggregates.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level05 / **Estimated time** 50 min

## 1. Why this matters — the business view

As you saw in level01, a DB stores tables split by the "each fact in one
place" principle. So real questions almost always span several tables: "who
(customers) bought what (products), how much (order_items), when (orders)?" —
four tables have to be connected before an answer exists. Without JOIN, SQL
stays a one-table toy; the moment you can JOIN, the entire company DB becomes
one analyzable whole. One more thing: **questions about what's missing** —
"customers who never ordered" (dormant customers, clients with unpaid
invoices) — simply cannot be expressed without LEFT JOIN.

## 2. Understanding by analogy

The order book holds only numbers, like "member #7, product #3." To build a
report, the clerk **reads a line of the order book → carries the member number
over to the customer register and finds that row → copies the name across.**
That "carrying a number to another book" is JOIN, and the number you carry is
the **join key** — usually one side's foreign key and the other side's
primary key.

The difference between INNER JOIN and LEFT JOIN is **what happens to rows
without a partner**. Think of a speed-dating event: attaching the attendee
list (customers) to the match records (orders):

- **INNER JOIN**: only matched people remain in the result. Attendees with no
  match record disappear.
- **LEFT JOIN**: the left list **keeps everyone**. People with no match get
  NULL (blank) in the record columns. → "find the unmatched" = find the rows
  with blanks.

To find "customers with no orders," put customers on the left, LEFT JOIN, and
pick the rows where the order column is NULL.

## 3. Core concepts

### 3-1. INNER JOIN, the basic form

```sql
SELECT o.order_id, c.name, o.ordered_at
FROM   orders AS o
JOIN   customers AS c ON c.customer_id = o.customer_id;
```

- `AS o`, `AS c`: table aliases. Once several tables appear, prefix columns
  with `alias.` to say which table each column comes from.
- `ON`: which keys pair the rows. **foreign key = primary key** is the
  standard shape.
- A bare `JOIN` is shorthand for INNER JOIN.

### 3-2. Joining three or more tables

JOINs chain like links: line items → orders → customers, and line items →
products. Keep level01's relationship diagram beside you and write the ON
clauses **by following the foreign-key arrows** — you won't get lost.

```sql
FROM order_items AS oi
JOIN orders    AS o ON o.order_id    = oi.order_id
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products  AS p ON p.product_id  = oi.product_id
```

### 3-3. LEFT JOIN — everyone on the left survives

```sql
SELECT c.name, o.order_id
FROM   customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id;
```

Customers with no orders stay in the result with a NULL order_id. Add
`WHERE o.order_id IS NULL` and only "customers with no orders" remain — the
canonical pattern for dormant-customer analysis. (RIGHT JOIN just swaps sides,
so in practice nearly everyone standardizes on LEFT.)

### 3-4. The fan-out trap — joins multiply rows

Join across a 1:N relationship and **each row on the 1 side gets replicated N
times**. If one order has 3 products, the moment you attach orders to
order_items that order becomes 3 rows. So far, so normal. The accident comes
next: count "number of orders" with `COUNT(*)` on the inflated result and you
get something like 2,000 instead of 1,000 — **inflated**. Attach the customer
table to a revenue sum the wrong way and the same thing happens.

Two defenses:

1. Count by the **unique key** of the thing being counted:
   `COUNT(DISTINCT o.order_id)`.
2. **Join only the tables you need**: don't attach tables the aggregate
   doesn't use.

Whenever a post-join aggregate comes out "bigger than it feels like it should
be," suspect fan-out first.

### 3-5. Division of labor between ON and WHERE

ON is "the pairing rule"; WHERE is "the filter on the joined result." In an
INNER JOIN they give the same answer either way, but in a LEFT JOIN, putting a
right-table condition in WHERE knocks out the NULL rows and **quietly turns it
into an INNER JOIN**. Right-table conditions on a LEFT JOIN belong in ON.

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

- **[1] Two tables**: orders + customers — your first join, where numbers
  become names.
- **[2] Four tables**: "who bought what, how many, for how much" — a chain
  join around the line items, through orders, customers, and products.
- **[3] Join + aggregate**: top 5 customers by total purchases — JOIN and
  GROUP BY combined.
- **[4] LEFT JOIN**: every customer + their order count. Confirm customers
  with no orders show 0.
- **[5] Anti-join**: `WHERE o.order_id IS NULL` lists "customers who never
  ordered" — the dormant-customer report.
- **[6] Fan-out demo**: the same "how many orders?" question counted three
  ways — (a) orders alone, (b) COUNT(*) after joining order_items,
  (c) COUNT(DISTINCT order_id) — and only (b) comes out inflated. This one
  table is the most valuable output of the level.

## 5. Try it yourself

1. **(Easy)** Modify [1] to attach orders + employees and build a "which
   employee handled this order" table.
   *Hint: ON e.employee_id = o.employee_id.*
2. **(Medium)** Print total quantity sold per product, largest first.
   *Hint: order_items JOIN products → GROUP BY p.name → SUM(oi.quantity).*
3. **(Challenge)** Find "customers whose only orders are cancelled ones" —
   exclude anyone with even one completed/shipping order.
   *Hint: one way — LEFT JOIN with an extra ON condition o.status <>
   'cancelled', filter with IS NULL, then compare against all orders. A
   subquery (next level) solves it more cleanly.*

## 6. Common mistakes

- **Missing ON (Cartesian product)**: join without ON and you get every row ×
  every row. 1,000 × 200 = 200,000 rows. When a result explodes, check ON
  first.
- **Joining on the wrong key**: join on a duplicable column like name and
  people who share a name get tangled together. Join keys are primary key ↔
  foreign key.
- **SUM/COUNT while fanned out**: see section 3-4. Before aggregating, ask
  yourself, "did this join multiply my rows?"
- **LEFT JOIN + right-table condition in WHERE**: the NULL rows get cut and it
  becomes an INNER. Right-table conditions go in ON.
- **Identical column names without aliases**: use the name column that exists
  in both tables bare and you get an ambiguity error — or worse, the wrong
  one gets picked. In join statements, alias every column, always.

## Next level preview

Questions remain like "customers who bought more than average" — where **one
query's result is another query's ingredient**. In **level07 — Subqueries**
you'll learn queries inside queries, and the WITH syntax that keeps them
readable.
