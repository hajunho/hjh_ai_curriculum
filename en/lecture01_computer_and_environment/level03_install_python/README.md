# Lecture 01 · Level 03 — Installing Python and Running It for the First Time

> Check the Python on your machine, understand the difference between interactive (REPL) execution and running a script, and inspect your environment with a self-diagnosis script.

**Difficulty** ⭐ / **Prerequisites** level02 / **Estimated time** 35 min

## 1. Why learn this — the business view

Python is the common language of this entire curriculum. Data analysis, automation, AI — everything we will learn runs on Python. The reason Python became the world standard for workplace automation and AI is simple: its syntax reads close to English sentences, and the collection of ready-made tools (packages) built by people around the world is overwhelming.

Yet surprisingly many beginners stumble in their first session not on code but on **installation and execution**. "I typed python and it says not found," "I definitely installed it but a different version runs," "the company laptop has two of them and I can't tell which is which" — all completely common. This level is where you clear that entry barrier. Once you can diagnose for yourself which Python is on your machine, where it lives, and what version it is, environment problems will never fluster you again.

## 2. Understanding through an analogy

Installing Python is **hiring a skilled interpreter for your company**. The Python code we write is a human-friendly language, while the computer only understands 0s and 1s (machine code). An interpreter — the Python interpreter — translates our instructions into machine code one line at a time.

There are two ways to put the interpreter to work:

- **Interactive (REPL)** — sitting across from the interpreter and having **an on-the-spot conversation, one line at a time**. Type just `python3` in the terminal and the `>>>` prompt appears; type `2 + 3` and it instantly answers `5`. Great for light, calculator-style experiments. REPL stands for Read–Eval–Print–Loop.
- **Script execution** — writing the instructions up **as a document and handing it over whole**. Put the instructions in a file like `main.py`, pass it over with `python3 main.py`, and everything runs from top to bottom. Because it can be stored, re-run, and shared, real work is almost always done this way.

On-the-spot conversation (REPL) for trying ideas; written instructions (a script) for official work — that distinction is all you need to remember.

## 3. Core concepts

### 3-1. Check before you install

Before installing anything new, check whether it is already there. In the terminal:

```bash
python3 --version    # e.g.: Python 3.12.9
which python3        # macOS/Linux: location of the executable (Windows: where python)
```

Version 3.10 or later is plenty for this curriculum. If it is missing, download the installer from the official python.org site; on a Mac, Homebrew works too (`brew install python`); on Windows, the key point is to **check "Add Python to PATH"** in the official installer.

### 3-2. Python 2 vs. 3, and multiple Pythons

For historical reasons, two commands exist: `python` and `python3`. Old Python 2 is end-of-life, so **we always use `python3`**. Also, having several Pythons installed on one computer is common and perfectly normal (one bundled with the OS, one you installed yourself, and so on). What matters is "**which** Python does my command run right now?" — and the answers are `which python3` and `sys.executable`. The proper way to keep multiple Pythons from tangling is the virtual environment, covered in the next level.

### 3-3. PATH — the order in which the computer looks up commands

When you type `python3` in the terminal, the shell searches **a list of folders called PATH**, front to back, for a program with that name. In company terms, it is "the ordered list of departments you visit when looking for the interpreter." "command not found" means the name appears nowhere in that list; if a wrong version runs, another Python sits earlier in the list. PATH is one of the environment variables, covered in more depth in level09.

### 3-4. Script conventions

- Python files use the `.py` extension.
- The paragraph wrapped in triple quotes (`"""..."""`) at the top of a file is the **docstring** — documentation explaining what the file is.
- `if __name__ == "__main__":` is the conventional marker meaning "run the following only when this file is executed directly." For now, "this marks the start of the main body" is understanding enough; the mechanics are covered in the modules lecture (lecture02).

## 4. Hands-on — main.py

```bash
python3 main.py
```

The script prints **a health-check report for your Python environment**.

- **[1] Verify the interpreter's identity**: Python version, the executable's actual location (`sys.executable`), and a pass/fail check against version 3.10+.
- **[2] Check the workplace**: operating system (macOS/Windows/Linux), processor type, and the current working directory.
- **[3] A look at PATH**: prints the front of the folder list the shell searches, confirming that "the command lookup order" really exists.
- **[4] REPL vs. script**: shows how the same calculation would have been typed in the REPL, contrasted with the fact that a script is running right now.
- **[5] Overall verdict**: a diagnosis summary with a pass/warning mark per item.

The key code is the use of the `sys` and `platform` modules. `sys.version_info` lets you compare the version numerically, and `sys.executable` gives the exact location of "the Python running this very code." These two lines are the all-purpose diagnostic tool you will reach for whenever an environment problem appears — commit them to memory.

## 5. Try it yourself

1. **(Easy)** Type `python3` in the terminal to enter the REPL, compute `3 * 7` and `"hi" * 3`, then leave with `exit()`. *(Hint: seeing `>>>` means you made it. Ctrl+D also exits.)*
2. **(Medium)** In step [1] of main.py, change the required version from (3, 10) to (3, 99) and run it. Watch the verdict flip to a warning, then put it back. *(Hint: look for the `REQUIRED` variable. This experiment proves the diagnostic logic is really working.)*
3. **(Challenge)** Add a line to step [2] printing whether the installed Python is a 32-bit or 64-bit build. *(Hint: look up `platform.architecture()`.)*

## 6. Common mistakes

- **Mixing `python` and `python3`**: on a Mac, `python` may be missing or a different version. Standardize on `python3` (inside a virtual environment, plain `python` becomes safe too — next level!).
- **Installing but not restarting the terminal**: PATH changes often apply only to new terminal windows. If a command fails right after installing, open a fresh terminal.
- **Typing a file-execution command into the REPL**: typing `python3 main.py` at the `>>>` prompt is an error. Files are run from the shell, after leaving the REPL. Use the prompt to tell where you are: shell (`$`) or REPL (`>>>`).
- **Reinstalling before reading the error**: "command not found" and "SyntaxError" have completely different causes. Reading just the first and last lines of the message points you in the right direction.

## Coming up next

The interpreter is hired. But what if every project needs a different set of specialist tools (packages) — how do you manage that? In level04 we learn the **virtual environment (venv)**, an independent workshop per project, and **pip**, the tool-procurement desk. It is the key level for preventing "but it works on my machine" incidents.
