# Lecture 04 · Level 00 — Why Do We Need Databases?

> Live through the three little hells of passing Excel files around (concurrent edits, consistency, permissions) in a hands-on simulation, then see how a database solves each one.

**Difficulty** ⭐ / **Prerequisites** none (being able to read lecture02-level Python is enough) / **Estimated time** 30 min

## 1. Why this matters — the business view

You have probably received an email like this at work:
"Please ignore customer_list_final.xlsx — use customer_list_final_REALLY_final_v3.xlsx."
The moment files start being passed around, **nobody knows which one is the
truth anymore.** Two people edit the same file at the same time and one
person's work silently disappears; the same customer is spelled differently
on every sheet, so the revenue total comes out different every time; and a
file containing salary data spreads to the whole company with a single
"FYI" email.

A database (Database, DB) is the tool built to solve exactly these problems.
The moment you stop thinking of it as "something harder than Excel" and start
thinking of it as "the thing that prevents Excel-sharing accidents," you begin
to see why there is a DB behind every system your company runs.

## 2. Understanding by analogy

**Passing Excel files around = photocopying the ledger and handing out copies.**
Imagine the accounting ledger being photocopied so Sales and Marketing can each
take one home and make their own edits. A month later, the numbers in the three
ledgers all disagree. Finding out who is right means a late night comparing
them line by line.

**A database = the original ledger in a vault + a clerk at the counter.**
There is exactly one original ledger, and it stays in the vault. Nobody touches
it directly; instead, everyone **asks** the clerk (the DBMS, Database
Management System): "please change customer 3's address," "please show me this
month's revenue." The clerk lines the requests up and handles them one at a
time (solving concurrent edits), refuses any entry that breaks the ledger's
rules (consistency), and checks each requester's authorization before showing
anything (permissions).

The standard language for talking to this clerk is SQL (Structured Query
Language), and it is what this entire lecture teaches.

## 3. Core concepts

### 3-1. The concurrent-edit problem (Lost Update)

If Kim and Park each open a copy of the same Excel file, edit different cells,
and save, **whoever saves last wins — the whole file.** The first person's
edits vanish without a trace. This is called a lost update.
The scary part is that **no error occurs.** Data disappears and nobody notices.

A DB processes change requests in one place, in order, so two requests that
touch different rows both go through, and requests that touch the same row are
serialized with a lock.

### 3-2. The consistency problem (Integrity)

Excel lets any cell hold anything. "Three-ish" in a quantity cell, "sometime
Monday" in a date cell, the same client registered twice as "Acme Corp." and
"Acme Corporation" — there is effectively no way to stop it. That is why your
totals come out different every time you aggregate.

A DB **declares the rules first**, when the table is created. Data that breaks
a rule (a constraint) — "quantity must be an integer," "customer IDs cannot
repeat," "an order's customer ID must actually exist in the customer list" —
is refused at the moment of saving.

### 3-3. The permission problem (Access Control)

An Excel file escapes your control the instant it is copied. Because of one
salary column you either cannot send the file at all, or you end up making the
unrealistic request "please pretend you didn't see that column." A DB lets you
set permissions at the counter, per user: "read-only on this table," "this
column is invisible to you." The original never leaves the vault.

### 3-4. So when do you use a DB?

| Situation | Excel | DB |
|---|---|---|
| One-off calculation, just you | ✅ perfect | overkill |
| Several people editing the same data | ❌ accident waiting to happen | ✅ |
| Hundreds of thousands of rows or more | ❌ slows down / hits limits | ✅ |
| Data with rules (orders, members, inventory) | ❌ gets polluted | ✅ |
| Contains sensitive information | ❌ leak risk | ✅ access control |

This does not mean Excel is a bad tool. It means Excel is the wrong **system of
record** for data that must be shared by many people, kept for a long time, and
held to rules. The strongest real-world combo is: "the original lives in a DB;
analysis and reports are pulled out with SQL into Excel/Python."

## 4. Hands-on — main.py

Run it:

```bash
python3 main.py
```

You can read this level's main.py without knowing any SQL yet. It replays the
"Excel way" as a Python simulation in four steps and contrasts it with the DB
way at the end.

- **[1] Concurrent-edit accident**: Kim and Park each copy the same 'file'
  (a list of dictionaries), edit different entries, and 'save' one after the
  other. Watch Kim's edit disappear in the output. There is not a single
  error message.
- **[2] Consistency pollution**: totaling a ledger where the same client was
  entered twice with different spellings splits one client into two, so the
  report is wrong.
- **[3] No permissions**: shows how handing over the whole file hands over the
  salary column with it.
- **[4] The DB way**: builds a real database, `hjh_shop.db`, with Python's
  built-in sqlite3, then shows that two connections (two users) editing
  different rows **both get saved safely**, and that bad data (a text
  quantity) is **refused at the point of saving**.

What to look for in the output: the final table in [1] keeps only Park's edit,
while the table in [4] keeps both people's edits. Same scenario, different
outcome — that difference is the starting point of this entire lecture.

## 5. Try it yourself

1. **(Easy)** In step [1] of main.py, swap the save order (Park first, Kim
   last). Whose edit disappears this time?
   *Hint: just swap the two save lines.*
2. **(Medium)** Add one more duplicated client with a different spelling to the
   ledger in step [2] and watch the totals split again.
   *Hint: add one dictionary line to the dirty_ledger list.*
3. **(Challenge)** Think of one dataset your team currently manages by passing
   files around, and write a paragraph on which of this level's three problems
   (concurrent edits, consistency, permissions) could actually happen to it.
   If you learn the coming levels while imagining that data moving into a DB,
   everything will stick much faster.

## 6. Common mistakes

- **Thinking "a DB is just a big Excel"**: the key difference is not size but
  the *counter* (request handling). Rule checks, ordering guarantees, and
  permissions all happen at the counter.
- **Trying to throw Excel away entirely**: for one-off analysis and reports,
  Excel is still king. Only the system of record and concurrent editing belong
  in the DB.
- **"We're fine, we use a shared drive"**: cloud spreadsheets with live
  co-editing still don't give you integrity rules, fine-grained permissions,
  or large-scale processing. That solves only half the problem.
- **Googling how to install sqlite3**: SQLite ships inside Python. One line —
  `import sqlite3` — and you're done. Nothing in this lecture needs installing.

## Next level preview

Now that you know why a DB is needed, it's time to look at what one looks like
inside. In **level01 — Tables, Rows, Columns, and Primary Keys** we open up the
schema of the practice online-store DB and draw the map of how its tables
connect to each other.
