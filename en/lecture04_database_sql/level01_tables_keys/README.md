# Lecture 04 · Level 01 — Tables, Rows, Columns, and Primary Keys

> Learn what the inside of a database looks like: the structure of tables, rows, and columns, the unique number attached to every row (primary key), and the reference number that links tables together (foreign key) — all by querying the practice DB's schema directly.

**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 35 min

## 1. Why this matters — the business view

Ask the person who runs your company's systems for "the order data" and you'll
get an answer like this: "The orders table doesn't have amounts. You need to
join order_items and products." If you can't parse that sentence, every data
request turns into a game of ping-pong. Data in a DB is not one big
Excel-style sheet — it is stored as **several small tables connected by
numbers**. Once you can read that structure map (the schema), you can decide
for yourself where each piece of data lives and what needs to be joined to what
to produce the table you want. It's map-reading, and it comes before SQL
grammar.

## 2. Understanding by analogy

Picture the owner of a small stationery shop keeping the books. At first,
everything goes into one notebook: "March 2, James Smith (a regular, lives in
New York), 2 fountain pens, 29,000 each…" But when James moves house, there's
a problem: the owner has to **find every line** in the notebook that mentions
James and fix the address. Miss one line and the books contradict each other.

A smart owner splits the books: a **customer register** (each customer gets a
member number), a **product list** (each product gets a product number), and an
**order book** (a date plus "member #7 bought product #9, quantity 2"). Now
when an address changes, exactly one line in the customer register gets fixed.

Every core DB concept just appeared:

- One book = a **table**
- One line in a book = a **row** — one customer, one order
- A field in the book (name, address…) = a **column**
- The unique number on each line, like the member number = the **primary key (PK)**
- The "member #7" written in the order book = a **foreign key (FK)** pointing at another book

## 3. Core concepts

### 3-1. Primary key — a name tag that never repeats

A name cannot be a primary key: two people can share one. So every table
designates one value that is unique per row (customer ID, order ID, employee
ID…). A primary key has two requirements: **no duplicates, never empty
(NULL).** The DB refuses any save that breaks these rules (you already watched
that happen in level00).

### 3-2. Foreign key — a finger pointing at another book

The customer_id column of the orders table holds not a customer's name but
**the primary key value from the customers table**. It's a reference: "this
order belongs to row 7 of the customer register." This way, (1) editing a
customer's info happens in one place, and (2) ghost data — like an order from
a customer who doesn't exist — can be blocked.

### 3-3. The five tables of the practice DB

Here is the structure of `hjh_shop.db`, the online-store DB we'll use for the
whole lecture.

| Table | Contents | Primary key | Foreign keys |
|---|---|---|---|
| customers | 200 customers (name · city · grade · join date) | customer_id | — |
| products | 10 products (name · category · price · cost) | product_id | — |
| employees | 20 employees (name · dept · salary · manager) | employee_id | manager_id → employees |
| orders | 1,000 orders (who · when · status) | order_id | customer_id → customers, employee_id → employees |
| order_items | order line items (which product, how many) | (order_id, product_id) | order_id → orders, product_id → products |

As a text relationship diagram:

```
customers ──< orders >── employees ┐
                │              └───┘ (manager_id self-reference)
                └──< order_items >── products
```

`──<` means a one-to-many (1:N) relationship. One customer has many orders.

### 3-4. Why doesn't the order book have amounts? — a taste of normalization

If orders stored the product name and price too, the same information would
repeat a thousand times, and a price change would mean fixing a thousand rows.
**Record each fact in exactly one place** — that principle is called
normalization. The trade-off is that reading the data back requires stitching
tables together, and that technique (JOIN) is level06. "Store it split, read
it joined" is the basic rhythm of a relational DB.

### 3-5. order_items has *two* primary key columns?

The primary key of order_items is the **pair** (order_id, product_id). It
means the combination "product 3 in order 17" can appear only once. This is
called a composite key. You'll see it often in bridge tables that resolve a
many-to-many relationship between two books (orders ↔ products).

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

main.py builds `hjh_shop.db` and then asks the DB itself, "what do you look
like?" SQLite keeps every table's creation statement in a special table called
`sqlite_master`, so the schema can be queried with SQL.

- **[1] Table list**: `SELECT name FROM sqlite_master WHERE type='table'`
- **[2] Each table's structure**: `PRAGMA table_info(table_name)` prints each
  column's name, type, and primary-key flag as a table. A pk value of 1 or
  more means primary key.
- **[3] Row counts**: `SELECT COUNT(*)` per table to gauge the data's size.
- **[4] Relationship diagram**: follows the foreign-key links and prints a
  text diagram.
- **[5] Reference check**: queries both tables side by side to confirm with
  your own eyes that customer_id=7 in orders really points at row 7 of
  customers.

In the output, watch for **columns flagged pk** and **columns ending in
`_id`**. An `_id` column is almost always a foreign key pointing at some
table's primary key. This same routine (table list → each table's keys → draw
the relationships) works verbatim when you meet an unfamiliar DB at work.

## 5. Try it yourself

1. **(Easy)** Using the output of step [2], write down the primary key of each
   of the five tables on paper. Why does order_items show two pk rows?
   *Hint: reread the composite-key explanation in section 3-5.*
2. **(Medium)** Mimic step [5]: add a query that checks which row of the
   employees table the employee_id in orders points at.
   *Hint: copy the customer_id code and change only the table names.*
3. **(Challenge)** Take data from your own job (say: clients, contracts,
   account owners) and design it as 2–3 tables — which columns go in which
   table, and what are the primary and foreign keys?
   *Hint: "each fact in one place" — information you keep writing repeatedly is
   the signal to split off a table.*

## 6. Common mistakes

- **Using names as keys**: they collapse under duplicate names, legal name
  changes, and typos. A meaningless unique number is the safest key.
- **The habit of cramming everything into one table**: that's Excel thinking.
  When you see repeating information, think "split off a table."
- **Reading a foreign-key column as 'just a number'**: customer_id=7 is not a
  quantity or an amount — it is an address meaning **row 7 of another table**.
  It must never be averaged or summed.
- **Writing queries before checking the schema**: on any unfamiliar DB, always
  start with the table list and key structure. Steps [1]–[4] of this level are
  that exact order.

## Next level preview

You've read the map — time to start talking. In **level02 — SELECT Basics**
you finally learn SQL's first sentence: "show me these columns from this
table."
