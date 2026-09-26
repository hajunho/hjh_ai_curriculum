# Lecture 01 · Level 08 — Automating Repetitive Work with Shell Scripts

> Learn the automation mindset that turns the weekly manual file-sorting-and-renaming chore into a single script run, and practice bulk file cleanup with Python.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level02 / **Estimated time** 45 min

## 1. Why learn this — the business view

Picture this task: every Monday, you download the dozens of per-store sales files piling up in the shared folder, fix the file names to match the convention, sort them into monthly folders, and move old files to the archive. Twenty minutes each time — seventeen hours a year. And humans make mistakes: one file lands in the wrong folder, one date in a name gets mistyped.

The cost-benefit math of automation is simple. **If "repetitions × time saved > time to build the automation," it is a guaranteed win** — with zero mistakes thrown in for free. This level takes "bulk file cleanup" as your first real case. File sorting is the greatest common denominator of office busywork, and the pattern you learn here (collect the list → apply the rules → execute in bulk → report the results) transfers unchanged to every automation that follows: data preprocessing, report generation, sending emails, all of it.

## 2. Understanding through an analogy

**A script is the work manual you hand a new hire.**

A capable senior colleague doesn't just tell the new hire what to do — they write a manual. "1) Open the inbox folder. 2) Prefix each file name with the date. 3) File it into the store's folder. 4) Report the completed list." A good manual produces the same result no matter who runs it or how many times.

A script is exactly this manual. Except the executor is a computer, not a new hire, so: (1) if the instructions are even slightly ambiguous it stops or does something absurd, and (2) in exchange, write it correctly once and you can assign the job ten thousand times with no complaints and no mistakes.

There are two languages for writing the manual. A **shell script (.sh)** is the terminal commands from level02 (cp, mv, etc.) written into a file in order — "a manual written in terminal language." A **Python script (.py)** is the same job written in Python. For a short sequence of commands the shell is more concise; the moment conditions, exceptions, or data manipulation get involved, Python wins decisively. This curriculum's rule of thumb: "two or three lines, use the shell; anything more, use Python."

## 3. Core concepts

### 3-1. Minimum shell-script syntax

```bash
#!/bin/bash                      # line 1: this manual's executor is bash
mkdir -p archive                 # write commands in order
mv report_*.csv archive/         # the * wildcard: every file matching the pattern
echo "Cleanup done: $(date)"     # progress report
```

Run it with `bash cleanup.sh`. **Wildcards** like `*` (any string) and `?` (one character) grab "everything matching the pattern" — the heart of bulk processing.

### 3-2. The four-stage pattern of an automation script

Every automation shares the same skeleton:

1. **Collect**: get the list of targets — `Path.glob("*.csv")`
2. **Decide**: apply the rules to each target — parse the file name, choose the destination
3. **Execute**: move / rename / create — `rename`, `mkdir`
4. **Report**: record what was done — log output, a summary table

The most important habit here is the **dry run** — locking stage 3 and printing only "what *would* be done" as a rehearsal. Verifying the plan with your own eyes before moving anything beats regretting 500 wrongly moved files afterwards. That is how professionals work.

### 3-3. Python's core tools for file handling

| Tool | What it does |
|---|---|
| `Path.glob("*.csv")` / `rglob` | Collect files by pattern (rglob includes subfolders) |
| `path.rename(new_path)` | Move and/or rename (the shell's mv) |
| `shutil.copy2(src, dst)` | Copy (the shell's cp, preserving timestamps) |
| `path.mkdir(parents=True, exist_ok=True)` | Create folders (the shell's mkdir -p) |
| `re.match(pattern, name)` | Parse file names with a regular expression |

A **regular expression** is a "text-pattern search language." For example, `r"(\d{4})-(\d{2})"` means "4 digits, a dash, 2 digits" (year-month). It is powerful for pulling information like dates and store names out of file names; for now, knowing "this tool exists" is plenty.

### 3-4. Designing the safety nets

Automation is fast — and so are its accidents. Three minimum safety nets: **(1) make dry-run mode the default**, **(2) move instead of delete** (set things aside in a quarantine folder), **(3) keep an execution log** (what moved where, and when). Just these three prevent the majority of automation disasters.

## 4. Hands-on — main.py

```bash
python3 main.py
```

The script recreates **a messy shared folder** inside a temporary directory and shows the full process of an automatic cleanup script tidying it up.

- **[1] Make the mess**: creates 14 files with wildly inconsistent naming — `SalesReport_Downtown_2024-03.csv`, `sales_lakeside_2024-04.csv`, `memo.txt`, `slides.pptx`, and so on — with seed-fixed random contents.
- **[2] Collect & decide**: parses year-month and store from each file name with regular expressions and drafts a classification plan — "where does each file go?" Unparseable files are bound for the `_needs_review` folder.
- **[3] Dry run**: prints only the move plan before executing — the rehearsal of "what is about to happen."
- **[4] Real execution**: creates the folders and bulk-moves the files, unifying names into the `store_YYYY-MM.csv` format.
- **[5] Report**: shows the tidied folder tree, a processing summary (n sorted, n needing review), and compares against the time the job would have taken by hand.

The key piece of code is the separation of `plan_moves()` and `execute_moves()`. **Splitting planning and execution into separate functions** gives you the dry run for free and makes testing easy. This structure is the standard design for real-world automation.

## 5. Try it yourself

1. **(Easy)** Add `sales_university_2024-05.csv` to the file list in step [1] and confirm a new University store folder appears. *(Hint: one line added to the `MESSY_FILES` list.)*
2. **(Medium)** Right now pptx files go to `_needs_review`. Add a classification rule that creates a `presentations` folder and sends pptx files there. *(Hint: add one branch inside `plan_moves()` that checks the `suffix`.)*
3. **(Challenge)** Build a **dry-run-only** version aimed at your computer's actual Downloads folder, printing only "if we cleaned up, what would go where." *(Hint: call only the planning function, never the execution function. Move things for real only after thoroughly verifying the plan!)*

## 6. Common mistakes

- **Bulk-executing without a dry run**: undoing hundreds of wrongly moved files is many times harder than moving them. Keep the order: print the plan → verify by eye → execute.
- **Overwriting files with the same name**: if the destination already holds a file with that name, it can be silently overwritten and the data lost. Check existence first (`exists()`) or append a number to the name.
- **Hard-coding absolute paths in an automation script**: bake in `/Users/myname/...` and the script dies instantly on a colleague's machine. Take the target folder as an argument or a setting.
- **Over-investing in throwaway code**: a job that will never repeat is faster done by hand. Do the automation cost-benefit math first.

## Coming up next

As automation grows, so does the **secret information** your scripts touch (accounts, API keys). What catastrophe follows from writing a password into your code? In level09 we learn to handle secrets safely outside the code, with environment variables and the .env pattern.
