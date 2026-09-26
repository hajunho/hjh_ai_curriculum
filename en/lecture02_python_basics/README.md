# Lecture 02 — Python Programming Basics

An introduction to Python for business professionals who are comfortable with
Excel but have never written code. We start from the question "what even is
programming?", and by the end of the 12 levels you will be able to write a
small work-automation program of your own — one that reads a data file, makes
decisions and aggregates by rule, and produces a report automatically.

## What you will learn

- The three principles behind every program — sequence, branching, repetition
- Python's backbone syntax: variables, data types, conditionals, loops, collections, functions
- The fundamentals of "code that doesn't break at work": file I/O, exception handling, modularization
- Pythonic expressions such as classes and OOP, comprehensions and generators
- The habit of protecting code quality with type hints and tests

## Prerequisites

- lecture01 (Computers and the Development Environment) — being able to run `python3 main.py` in a terminal is enough.

## Level table

| Level | Title | Difficulty | Key keywords |
|---|---|---|---|
| [level00](level00_what_is_programming/) | What Is Programming? | ⭐ | sequence·branch·loop, the language of procedure |
| [level01](level01_variables_types/) | Variables and Data Types | ⭐ | int/float/str/bool, f-string |
| [level02](level02_conditionals/) | Conditionals — Making the Computer Decide | ⭐ | if/elif/else, comparison & logical operators |
| [level03](level03_loops/) | Loops — Delegating the Boring Work | ⭐ | for/while/range/break |
| [level04](level04_collections/) | Lists, Tuples, and Dictionaries | ⭐⭐ | choosing the right data structure |
| [level05](level05_functions/) | Functions — Reusable Work Procedures | ⭐⭐ | arguments, return values, scope |
| [level06](level06_file_io/) | Reading and Writing Files | ⭐⭐ | open/with, CSV, utf-8 |
| [level07](level07_errors_debugging/) | Exception Handling and Debugging | ⭐⭐⭐ | try/except, reading error messages |
| [level08](level08_modules_packages/) | Modules and Packages | ⭐⭐⭐ | import, the standard library |
| [level09](level09_classes_oop/) | Classes and Object-Oriented Programming | ⭐⭐⭐ | attributes, methods, inheritance |
| [level10](level10_comprehensions_generators/) | Comprehensions, Generators, and Advanced Syntax | ⭐⭐⭐⭐ | lambda, a taste of decorators |
| [level11](level11_typing_testing_quality/) | Type Hints, Testing, and Code Quality | ⭐⭐⭐⭐ | type hints, assert-based tests |

## Fast track (if you are short on time)

If you only want the essentials, study these 5 levels in order.

1. **level01 Variables and Data Types** — the raw material of all code
2. **level02 Conditionals** — the start of automated decision-making
3. **level03 Loops** — the syntax where the payoff of automation is most tangible
4. **level05 Functions** — how to organize code like a work procedure manual
5. **level06 Reading and Writing Files** — the gateway that connects code to real data

Later, backfill level04 (data structures) and level07 (exception handling)
and you are fully ready to move on to lecture03 (Working with Data).

## Where this lecture shows up at work

- Opening the transaction CSV you receive every morning and auto-generating a per-item totals and anomaly summary report (level03·04·06)
- Turning a company rule like "auto-approve travel expenses up to KRW 150,000" into code and eliminating repetitive approvals (level02)
- Unifying tax/discount calculation logic into a single function when every department does it slightly differently, preventing mistakes (level05)
- Stopping an overnight batch job from dying because of one corrupted data row, using exception handling (level07)
- Managing similar-but-slightly-different rules — like employees vs. managers — cleanly with class inheritance (level09)
- Adding type hints and tests so that others (or future you) can trust and reuse your code (level11)

Every exercise uses only the standard library and runs with a single line —
`python3 main.py` — no internet connection required.
