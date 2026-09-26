# Lecture 01 · Level 02 — First Steps in the Terminal

> Learn the basic terminal commands (pwd, ls, cd, mkdir, cp, mv, rm) that let you put the computer to work with the keyboard alone — no mouse required.

**Difficulty** ⭐ / **Prerequisites** level01 / **Estimated time** 40 min

## 1. Why learn this — the business view

Every hands-on exercise in this curriculum starts by typing `python3 main.py` in the terminal. When you collaborate with developers, connect to a cloud server, or install an AI tool, what appears on screen is not a button to click but a single blinking cursor. If the terminal scares you, you will stall in front of it every single time.

The terminal is powerful because **work you request in words can be recorded and repeated**. Creating 100 folders with the mouse means 100 clicks; in the terminal it is one line. Save that one line, and next month — or your successor — can run it exactly the same way. This is the starting point of the repetitive-work automation you will meet in level08.

## 2. Understanding through an analogy

**A GUI (graphical screen) is giving instructions by gesture; the terminal is giving instructions in words.**

Suppose you ask a junior colleague to tidy up some documents. By gesture, you must walk to the cabinet yourself and point at every item — "this one, over there" (mouse clicking and dragging). In words, a single sentence does it: "please move all the 2023 documents from the third-floor cabinet to the archive" — and you can write the instruction down and have it repeated every month.

The structure of a terminal command mirrors a real instruction:

```
cp  report.txt  backup/
verb   object     place
"Copy report.txt into the backup folder"
```

The command is the verb; what follows are the objects and modifiers. Anything starting with `-`, like the `-l` in `ls -l`, is an **option** — an adverb meaning "in detail," "all of it," and so on.

One more thing: the terminal always has a notion of "where I am standing right now" — the **current working directory (CWD)** you learned in level01. Every command executes relative to the room you are standing in. That is why the number-one habit of terminal work is asking "where am I right now?"

## 3. Core concepts

### 3-1. The seven survival commands

| Command | Meaning | Analogy |
|---|---|---|
| `pwd` | print current location (print working directory) | "Which floor and room is this?" |
| `ls` | list the current folder's contents (list) | "What's in this room?" |
| `cd folder` | change location (change directory) | "Move to that room" |
| `mkdir folder` | create a folder (make directory) | "Install a new cabinet" |
| `cp source target` | copy | "Make a copy and put it over there" |
| `mv source target` | move or rename (move) | "Move it. Within the same room, just swap the label" |
| `rm file` | delete (remove) | "Into the shredder — **immediately, no recycle bin**" |

Frequent combinations: `cd ..` (one level up), `cd ~` (to your home folder), `ls -l` (detailed listing), `mkdir -p a/b/c` (create intermediate folders in one go).

### 3-2. macOS vs. Windows

- **macOS**: just open the built-in "Terminal" app. macOS is in the same family as Linux (Unix), so the commands above work as-is. Since most development servers run Linux, Mac users carry their skills straight over to servers.
- **Windows**: the traditional "Command Prompt (cmd)" uses different commands (`dir` instead of `ls`, `copy` instead of `cp`). These days **PowerShell** supports many of the familiar ones like `ls` and `cd`, and for development the standard choice is to install **WSL (Windows Subsystem for Linux)** and run a Linux terminal inside Windows. The commands in this curriculum assume macOS/Linux (including WSL).

| Purpose | macOS/Linux/WSL | Windows cmd |
|---|---|---|
| List | `ls` | `dir` |
| Copy | `cp` | `copy` |
| Move | `mv` | `move` |
| Delete | `rm` | `del` |
| Path separator | `/` | `\` |

### 3-3. The shell, your interpreter

The program that actually parses and executes your commands inside the terminal window is called the **shell**. The default shell on macOS is zsh; on Linux it is usually bash. Remember it as: "the terminal is the window; the shell is the interpreter inside who understands your instructions." When you later meet a `.sh` file (a shell script), it is a bundle of instructions written for this interpreter.

### 3-4. A warning about dangerous commands

`rm` deletes immediately, bypassing the recycle bin. In particular, `rm -rf folder` erases a folder and everything inside it irreversibly. In practice, the accident-prevention habits are: run `ls` to inspect the target before deleting, and read the line one extra time whenever a wildcard pattern (like `*.txt`) is involved.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Before touching the real terminal, you practice safely with a **mini shell simulator** written in Python. The script sets up a small practice office inside a temporary folder and plays a pre-written scenario one command at a time, as "input → execution → result."

- **[1] Set up the practice area**: a few sample files are laid out in a temporary folder. Mistakes cannot affect your computer.
- **[2] Run the scenario**: `pwd → ls → mkdir backup → cp → cd → mv (rename) → rm`, handling a real chore (backing up and tidying reports) through commands. Each command comes with a plain-English explanation of what it did.
- **[3] Check the final state**: shows the tidied folder structure and compares how many mouse clicks the same job would have taken.

The key piece of code is the `MiniShell` class. It holds `cwd` (the current location) as state, while methods like `cmd_ls` and `cmd_cd` implement each command with pathlib. In other words, you get to verify in code that **what a shell really does is "remember the current location and call file-system functions."** Once the simulator has shown you the flow, open a real terminal and type the same commands yourself.

## 5. Try it yourself

1. **(Easy)** Open a real terminal and type `pwd`, `ls`, `cd Documents` (or any folder you have), then `cd ..` in order. After each command, run `pwd` to see how your location changes. *(Hint: if you get lost, `cd ~` brings you home.)*
2. **(Medium)** Add two commands to the `SCENARIO` list in main.py — `mkdir archive` and `mv meeting_notes.txt archive` — so the meeting notes get moved into an archive folder. *(Hint: follow the format of the existing scenario entries exactly.)*
3. **(Challenge)** Add a `cat file` command (print a file's contents) to `MiniShell`. *(Hint: write a `cmd_cat` method and use `read_text()`. Observe how the other `cmd_` methods get registered.)*

## 6. Common mistakes

- **Running commands without checking your location**: the main culprit behind creating or deleting files in the wrong folder. Type `pwd` and `ls` first, as a reflex.
- **Typing names with spaces as-is**: `cd My Documents` is parsed as two objects, "My" and "Documents." Wrap it in quotes: `cd "My Documents"`.
- **Mistaking `rm` for the recycle bin**: terminal deletion is immediate and permanent. In important folders, a safe trick is to `mv` things aside somewhere else instead of `rm`.
- **Panicking when a command "doesn't work"**: it is almost always a typo (commands are strict about spelling) or a macOS/Linux command typed into Windows cmd. Reading the error message word by word is the best teacher there is.

## Coming up next

Now that you can talk to the computer through the terminal, it is time to **summon Python from that window**. In level03 we check whether Python is installed, learn the difference between the REPL and running a script, and run a script that self-diagnoses the Python environment on your machine.
