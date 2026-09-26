# Lecture 01 · Level 06 — Git — The Idea of Version Control

> Understand the core concepts of Git — snapshots, commits, branches — through approval-stamp and time-travel analogies, then practice with real git commands. This is the tool that ends "final_REALfinal.xlsx".

**Difficulty** ⭐⭐ / **Prerequisites** level02 / **Estimated time** 45 min

## 1. Why learn this — the business view

You have seen a folder like this: `proposal_v1.docx`, `proposal_v2_edits.docx`, `proposal_final.docx`, `proposal_final_REALfinal(boss_comments).docx`. Nobody knows which one is truly final, what changed in v2, or who changed it. Overwrite a file by accident, and the past is gone forever.

The software world solved this problem twenty years ago. The tool is **Git**. Git records the full change history of your files, lets you return to any point in the past, and lets you trace what changed, when, by whom, and why — in under a minute. Not just for code: documents, configuration, data pipelines — it is today's standard for tracking the history of "anything that gets made." AI projects are no exception; the history of model code and experiment settings all lives on Git. Without Git, half of every collaboration conversation with a development team goes over your head.

## 2. Understanding through an analogy

**A commit is an approval stamp, the repository is the approval filing cabinet, and a branch is parallel-universe time travel.**

- **Repository**: the invisible approval cabinet attached to the project folder (the `.git` folder). `git init` installs the cabinet, declaring "history tracking starts for this folder."
- **Commit**: taking a photo of the entire folder's state at this instant (a snapshot) and filing it in the cabinet with an approval stamp. The stamp carries a serial number (the hash), a date, the person responsible, and **the reason for approval (the commit message)**. Instead of "making separate v1, v2 files," you "keep the same file's state at each point in time."
- **Staging**: before submitting for approval, **the step of choosing which papers go on the approval tray**. With `git add` you select "only these files go into this round of approval." Only the chosen papers get stamped, not everything on the desk.
- **Branch**: opening a parallel universe. The main storyline (the main branch) stays intact while you run an experiment — "what if we changed the pricing policy?" — in the forked universe. If the experiment succeeds you merge it into the main story; if it fails you discard just that universe and the main story is unharmed.
- **Checkout / time travel**: you can wind the folder's state back to any stamp in the cabinet. "Let's go back to last Tuesday's state" is a single command.

## 3. Core concepts

### 3-1. The three spaces

Git work is movement between three spaces:

```
working folder (desk)  --git add-->  staging (approval tray)  --git commit-->  repository (cabinet)
```

| Space | Analogy | State |
|---|---|---|
| Working directory | The desk you are editing at | Freely editable, not yet recorded |
| Staging area | The approval tray | Candidates for the next commit |
| Repository | The approval cabinet | Stamped, permanent records |

`git status` is the all-purpose check: "what is on the desk and the tray right now?"

### 3-2. The seven essential commands

```bash
git init                       # install the cabinet (once per folder)
git status                     # check the current state (constantly!)
git add file or .              # put papers on the approval tray
git commit -m "reason"         # stamp it (message required)
git log --oneline              # browse the approval history
git branch name / git switch name   # create / move to a parallel universe
git diff                       # compare what changed
```

On a machine you use for the first time, register the name to engrave on the stamp: `git config --global user.name "Name"`, `git config --global user.email "email"`.

### 3-3. What makes a good commit

- **Small and often**: don't batch a whole day into one stamp; stamp each meaningful unit (one feature, one fix). Undoing becomes easy.
- **Messages state the "why"**: a message like "edits" is worse than none. Write a reason your future self can read and understand, like "fix discount-rate formula (10%→15%)."
- **A commit is an indestructible safety net**: anything committed can be recovered no matter what mistake follows. "Commit first" is the best backup habit there is.

### 3-4. .gitignore — what not to record

Temporary files, the `.venv` folder, and secret-key files (level09) do not belong in the cabinet. List the exclusions in a `.gitignore` file at the project root and Git pretends those files don't exist.

## 4. Hands-on — main.py

```bash
python3 main.py
```

If git is installed on your computer, the script **creates a real repository in a temporary folder and walks the entire flow automatically**. (If git is missing, it shows the same flow as a concept simulation and points you to installation instructions.)

- **[1] Install the cabinet**: creates a repository with `git init` and registers a practice name and email.
- **[2] First approval**: creates a price-list file and stamps the first commit via `add → commit`.
- **[3] Edit and diff**: changes a price, checks "what changed" with `git diff`, and makes a second commit.
- **[4] Browse the history**: views the accumulated stamps with `git log`.
- **[5] Branch experiment**: changes prices boldly on an `experiment` branch (a parallel universe), then returns to main and confirms **the file contents look untouched** — the moment time travel becomes real.

The key code is the `run_git()` function. It executes real git commands via Python's `subprocess.run()` and captures their output — Python typing the commands you would type, with commentary attached. In the output, the `$ git ...` lines are "the commands you would type yourself," so repeating the same sequence by hand in an empty folder afterwards makes the lesson complete.

## 5. Try it yourself

1. **(Easy)** In step [5] of the output, check what the price list contained after returning to the main branch. Did the experiment branch's edits affect main? *(Hint: they should not have — that is the parallel universe.)*
2. **(Medium)** In a real terminal, make a scratch folder and type `git init → create a file → git add → git commit → git log` yourself. *(Hint: follow the `$ git ...` lines from the main.py output in order.)*
3. **(Challenge)** Add a third commit after step [3] in main.py (say, adding a new menu item) and confirm three stamps appear in the log. *(Hint: repeat the edit-file → add → commit flow once more, using `demo_edit_and_diff()` as a guide.)*

## 6. Common mistakes

- **Committing without adding**: an empty approval tray leaves nothing to stamp. When "nothing to commit" appears, start by checking the tray with `git status`.
- **Committing without a message**: omit `-m "message"` and an unfamiliar editor screen opens and panics you (escape: `:q!` then Enter). Make attaching `-m` a habit.
- **Blindly `git add .` for everything**: the express lane to filing secret keys and huge files into the cabinet. Look at what is going in with `git status` before you add.
- **Deleting or moving the `.git` folder**: that folder *is* the cabinet. Delete it and the entire history is gone.

## Coming up next

Your own cabinet (the local repository) is complete. But what about **sharing** it with teammates? In level07 we learn remote repositories and the GitHub concepts (clone/push/pull/PR), and practice push/pull against a "fake remote" built locally — no internet required.
