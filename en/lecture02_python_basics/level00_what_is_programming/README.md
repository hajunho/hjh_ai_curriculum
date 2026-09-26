# Lecture 02 · Level 00 — What Is Programming?

> Programming is writing down a procedure so a computer can follow it without any room for misunderstanding — and every procedure can be expressed with just three things: sequence, branching, and repetition.
**Difficulty** ⭐ / **Prerequisites** none (finishing lecture01 level03 helps) / **Estimated time** 30 min

## 1. Why learn this — the business angle

At work we create and follow "procedures" every day. The new-hire onboarding
checklist, the month-end closing sequence, the customer-complaint playbook —
they are all procedure documents. Programming is writing that procedure
document in a form a computer, rather than a person, can read and execute.

There is one difference. A person will fill in the gaps of a slightly vague
manual using common sense; a computer does exactly what is written and nothing
more. That is why learning to program gives you two abilities. First, you can
hand entire repetitive tasks over to the computer. Second, you develop the
habit of designing procedures without ambiguity — which improves even the
plain business documents you write that contain no code at all.
In this level, before memorizing any syntax, we build the big picture:
"what is a program?"

## 2. Understanding through analogies

**The recipe analogy.** Think of a recipe for instant noodles. "Boil 550 ml of
water → add the noodles and seasoning → boil for 4 minutes 30 seconds." It
runs top to bottom, in order. That is **sequence**.

Recipes also have forks in the road. "If you like it spicy, add chili flakes;
otherwise, skip them." Taking a different path depending on a condition —
that is a **branch**.

And doing the same action several times, like "keep stirring until the
noodles are cooked," is a **loop**.

Remarkably, every program in the world is a combination of these three. A
payroll system, a smartphone game, a chatbot — all of them are sequence,
branching, and repetition stacked very high and very precisely. Whenever you
meet a new piece of syntax, ask "which of the three does this tool make more
convenient?" and you will never get lost.

**The work-manual analogy.** Picture the order-handling manual you get on your
first day as a barista. "Take the order → calculate the payment → if the
customer is a rewards member, add points → make the drink → buzz the pager
until the customer picks it up." Translate that manual into a computer
language and you have a program. The main.py in this exercise does exactly
this job.

## 3. Core concepts

### 3.1 Program = data + procedure

A program consists of two things. The raw material it works on — **data** —
such as order quantity, unit price, customer name; and the steps that process
that material — the **procedure** — calculating, deciding, printing. In Excel
terms, the values in the cells are the data, and the formulas and macros are
the procedure.

### 3.2 Sequence — top to bottom

Unless told otherwise, Python code runs from the first line to the last, in
order. Change the order and you change the result — just as "apply the
discount, then compute sales tax" and "compute sales tax, then apply the
discount" produce different amounts.

### 3.3 Branching — different paths for different conditions

"If X, do A; otherwise, do B." In Python this is written with the word `if`.
The detailed syntax comes in level02; today you only need the concept: this is
the point where the program makes a decision on its own.

### 3.4 Repetition — the same job, many times

"For every order in the list, do X." In Python this is written with the word
`for`. A person doing something 100 times gets tired and makes mistakes; a
computer does it a million times at identical quality. Most of the payoff of
automation comes from here.

### 3.5 Why you must write precisely for a computer

Tell a new hire "tidy this up and report back, roughly" and you will get some
result — they fill in the blanks with experience and common sense. A computer
has no such common sense. Write only "aggregate the sales" and it stops,
because it does not know which file, which column, or by what rule to add.
Half of programming is therefore "spelling out a procedure you already know,
completely and without ambiguity." Get into the habit of writing the
procedure first as numbered sentences in plain English (this is called
**pseudocode**) — do that, and the design of your program is finished before
you know any syntax.

### 3.6 Programming languages and Python

A notation for procedures that a computer understands is called a
**programming language**. Among them, Python reads closest to plain English
sentences, making it the easiest for non-majors to learn, and it is the
standard language of data analysis and AI — which is why this entire
curriculum uses it.

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level00_what_is_programming
python3 main.py
```

main.py is a demo that translates a cafe's "americano order handling"
procedure into code. The output comes in four stages.

- **[1] Sequence**: shows order intake → amount calculation → receipt printing running top to bottom.
- **[2] Branch**: the loyalty-points message changes depending on whether the customer is a rewards member. The `if is_member:` line in the code is the fork in the road.
- **[3] Loop**: processes 3 orders with a single `for` loop. The key point: even with 300 orders, the code stays exactly the same.
- **[4] Summary**: prints today's conclusion — "every program is a combination of sequence, branching, and repetition."

You have not learned the syntax yet, so do not try to understand every line.
If a part like `cup_price * quantity` makes you go "wait — I can read this,"
that is success. Just follow the comments (the explanations starting with `#`)
and spot which lines are sequence, branch, and loop.

## 5. Try it yourself

1. **Change a value** — open main.py, change `cup_price = 4500` to a different
   number, and run it again. How do the printed amounts change? (Hint: change
   the data and the same procedure produces a different result.)
2. **Experience the branch** — change `is_member = True` to `False`, run it,
   and see which line disappears from the output. (Hint: the indented lines
   under `if` no longer execute.)
3. **Break a daily task into a procedure** — pick one task you do every week
   (e.g., writing the weekly report) and write down on paper where the
   sequence, branches, and loops are. (Hint: phrases like "repeat until ..."
   and "only if ..." mark the spots.)

## 6. Common mistakes

- **Starting by memorizing syntax** — syntax is just a tool. The far more
  important habit is stating in words "which procedure am I trying to
  automate?" first.
- **Expecting the computer to figure it out** — the computer does exactly what
  is written. Even if "obviously that includes sales tax" to you, if it is not
  in the code, it is not calculated.
- **Trying to build the perfect program in one go** — real-world programs are
  built by making a small procedure, running it, checking, and growing it bit
  by bit. Press the run button often.
- **Treating errors as failure** — an error message is a question the computer
  is asking you. You will learn to read them in level07; for now, not being
  scared of them is enough.

## Next level preview

Procedures need ingredients. In level01 we learn about **variables** — the
boxes that hold data — and **types** — the kinds of data — and write our
first "real code" with a sales calculation example.
