# Lecture 01 · Level 09 — Handling Environment Variables and Secrets Safely

> Understand why writing API keys and passwords into code causes disasters, and build the habit of managing secrets outside the code with environment variables and the .env pattern.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level06 / **Estimated time** 40 min

## 1. Why learn this — the business view

There is an incident pattern that keeps repeating in real life. A developer pushes code to a public GitHub repository with a cloud access key written inside. The internet is patrolled 24/7 by harvesting bots that comb public repositories, so **within minutes** the key is in a thief's hands. The thief rents a fleet of high-end servers with it to mine cryptocurrency, and next month the company receives a bill for tens of thousands of dollars. Not an exaggeration — this exact pattern has played out over and over across the industry.

In the AI era the risk has grown. One LLM API key allows unlimited paid calls, and a leaked internal database password leads to customer-data breaches. That is why "never put secrets in code" is rule number one of development security, and the very first thing security audits check. Fortunately the fix is not hard: **inject secrets via environment variables, and let the code refer to them by name only.** Internalizing that one sentence is this entire level.

## 2. Understanding through an analogy

**A password written in code is a note with the safe combination taped to the front door.**

Code gets copied around, shared, and permanently recorded in the repository (as level06 taught, Git history does not disappear). Writing the safe combination into such a document is like writing it on the front-door sticky note that moves with you every time you relocate.

Think of **environment variables as the company's locker-key system**. The work manual (the code) says only "the safe key is in each person's locker." The actual key (the secret value) sits in each employee's locker (each computer's environment variables). Even if the manual leaks, the key is safe — and each employee (dev server, production server) can use a different key.

An environment variable is one of the "name=value" notes the operating system hands to each program. You are already using them — the PATH from level03 was an environment variable. Add an entry like `OPENWEATHER_API_KEY=abc123`, and the code fetches the value by **calling the name only**: `os.environ["OPENWEATHER_API_KEY"]`. The actual value appears nowhere in the code.

## 3. Core concepts

### 3-1. Reading and writing environment variables

```bash
# set in the shell (valid only while this terminal window lives)
export MY_API_KEY="sk-demo-1234"
python3 app.py
```

```python
import os
key = os.environ.get("MY_API_KEY")        # None if missing (the safe read)
key = os.environ["MY_API_KEY"]            # error if missing (for required values)
```

Values vanish when the terminal closes, so for values you use all the time, put the export in your shell config file (`~/.zshrc` etc.); per project, use the .env pattern below.

### 3-2. The .env file pattern

Create a file named `.env` in the project folder and gather the secrets there:

```
# .env — this file NEVER goes into Git!
DB_PASSWORD=s3cret!
LLM_API_KEY=sk-demo-abcd1234
```

And **always add `.env` to `.gitignore`**. Those two lines are a matched set. A tool that reads this file at startup and promotes it into environment variables (the python-dotenv package) is used practically as a standard, but the format is so simple that in this level's exercise we build the parser ourselves to see the mechanics. For colleagues, commit a template with the values blanked out — `.env.example` (`DB_PASSWORD=fill_in_here`) — sharing only "which named secrets are needed."

### 3-3. The three lines of defense against leaks

1. **Prevention**: secrets live only in .env/environment variables. Never write the value itself in code, commit messages, or chat.
2. **Detection**: scan for hard-coded secrets before committing. In practice a commit hook or CI checks automatically; in this level's exercise we build a mini scanner to see the mechanics. Long strings starting with `sk-`, `password = "..."` patterns, and the like are the clues.
3. **Response**: if a secret leaked, **revoke and reissue the key immediately** — the only real remedy. Even after deleting the commit, assume it already survives in history, forks, and bot harvests.

### 3-4. Mask before you print

Never print a secret whole to logs or screen either. When you need to verify one, the convention is **masking** — showing only a few leading and trailing characters, like `sk-d****1234`. Separate "is the value set?" from "what is the value?", and reveal only the former.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Three tools are demonstrated in turn. Every value is a practice fake, and nothing connects to the internet.

- **[1] Reading environment variables**: sets a practice variable and reads it via `os.environ`, including the safe `.get()` read of a missing variable and a look at an existing one like PATH.
- **[2] Masked output**: the `mask()` function hides a secret as `sk-d****5678`, reporting only "it is set."
- **[3] A .env parser**: a mini implementation that parses `name=value` .env text and promotes it into environment variables — the mechanics behind the dotenv tool.
- **[4] A hard-coding scanner**: with a deliberately bad example (secrets embedded in code) and a good one (names only) prepared, a scanner using regex rules (API-key shapes, password= patterns) flags only the bad code.
- **[5] Rules recap**: prints the three lines of defense as a checklist.

The key code is `scan_for_secrets()`. It translates the observation that "secrets have a distinctive **shape** (pattern)" into a few lines of regular expressions — which is exactly the basic principle behind real security tools.

## 5. Try it yourself

1. **(Easy)** In the terminal, run `export PRACTICE_KEY="hello123"` and then `python3 main.py`, and confirm step [1] discovers the variable. *(Hint: you must run it in the same terminal window where you exported.)*
2. **(Medium)** Add the line `token = "ghp_aaaa1111bbbb2222"` to the bad example code in [4] and see whether the scanner catches it. If not, add a pattern rule for strings starting with `ghp_` to `SECRET_PATTERNS`. *(Hint: model it on the existing `sk-` pattern.)*
3. **(Challenge)** Modify `mask()` so that short secrets under 8 characters display as `****` only — short values can be guessed from just their leading and trailing characters. *(Hint: branch on `len(value)`.)*

## 6. Common mistakes

- **"It's a private repository, so it's fine"**: repositories get opened up eventually, and job changes or outsourcing widen access. Private or not, the principle is: secrets live outside the code.
- **Creating .env but forgetting .gitignore**: the most common road to catastrophe. Creating `.env` and adding it to `.gitignore` must always happen as one set.
- **Only deleting the commit after a leak**: history, forks, and bots already have copies. Reissuing the key is the only remedy.
- **Secrets slipping into error messages and logs**: debug output like `print(config)` dumps the whole configuration, secrets included. Make masking a habit.
- **Committing the template (.env.example) with real values filled in**: templates hold placeholder text only.

## Coming up next

Secrets are under control, but one problem remains: "it works on my machine but not on the server" — because the environments themselves differ. In level10 we learn Docker, which **packs the OS, Python, packages, and settings into one standardized shipping container and reproduces them identically anywhere**.
