# Lecture 01 · Level 05 — Code Editors and Jupyter

> Learn to tell apart the editor (VS Code) — the tool for *writing* code — from the Jupyter notebook — the tool for *experimenting* — and experience the notebook's cell-execution model through a simulation.

**Difficulty** ⭐⭐ / **Prerequisites** level04 / **Estimated time** 35 min

## 1. Why learn this — the business view

You *can* write code in Notepad. The reason nobody does so at work is that good tools catch mistakes ahead of time and multiply your speed — the same reason nobody closes the books with a calculator and paper instead of Excel.

Python work tools split into two branches: the **code editor**, for building programs (scripts), and the **Jupyter notebook**, for poking at data exploratively. A data analyst's screen almost always shows a notebook; a developer's shows an editor. Once you can tell the two apart, you can judge "which tool fits this task" on your own, and you will not freeze when a collaborator sends you an `.ipynb` file. The notebook's "cell execution order" concept in particular is where first-time users trip most often, so this level teaches you the mechanics in advance.

## 2. Understanding through an analogy

**The editor is the tool for writing the official report; the notebook is the lab notebook.**

- **An editor like VS Code** is like writing a formal report in a word processor. You produce the finished document (the whole script) and submit it for approval (execution). Spell check (syntax-error highlighting), table-of-contents navigation (jump to function), and autocomplete are powerful. The deliverable is a `.py` file that runs top to bottom, in order.
- **A Jupyter notebook** is the lab notebook of a research bench. "Add this reagent? → record the result → now raise the temperature? → record the result" — you **run one step, look at the result, and then decide the next step**. Code snippets (cells), their outputs, and explanatory prose all live together in one document.

There is one crucial difference. In a lab notebook, **the page order and the experiment order can differ**. You might run the experiment on page 3 first and come back to page 1. A notebook likewise lets you run cells in any order, and all cells **share the memory of a single workbench (the kernel)**. So if a cell higher on the screen was actually executed *after* a lower one, the results look wrong to anyone reading the document top to bottom. That is 90% of all notebook accidents.

## 3. Core concepts

### 3-1. Choosing the right tool

| Situation | Fitting tool | Why |
|---|---|---|
| First exploration of new data | Notebook | Decide direction step by step, watching results |
| Analysis with charts, drafting report material | Notebook | Code + results + prose in one document |
| An automation script to run weekly | Editor (.py) | Top-to-bottom, guaranteed re-runs |
| Developing a multi-file program | Editor | Cross-file navigation, search, debugging |
| Final code shared with the team | Editor (.py) | Execution order is always identical, hence trustworthy |

The common real-world flow: "experiment in a notebook → once the method is settled, consolidate into a `.py` script."

### 3-2. Three first steps in VS Code

VS Code (Visual Studio Code) is free and the de-facto industry standard editor. After installing, only three things need doing:

1. **Install the Python extension** — search "Python" in the extensions menu on the left and install it. You get syntax checking and a run button.
2. **Select the interpreter** — open the command palette (Cmd/Ctrl+Shift+P), run "Python: Select Interpreter," and point it at **the project's `.venv`**. This is how you tell the editor "which toolbox to use," as learned in level04.
3. **Open as a folder** — open the whole project folder (File > Open Folder), not a single file, so relative paths and search behave properly.

### 3-3. Notebook anatomy: cells + a kernel

- **Cell**: a box holding a few lines of code. Shift+Enter runs just that cell. The number like `[3]` to the cell's left is the **execution ordinal** — "run 3rd in this kernel."
- **Kernel**: the live Python process behind the scenes. Every cell shares this single memory (the variables). A variable created in one cell is visible in every other.
- **Restart**: restarting the kernel wipes the memory clean. "Restart & Run All" is the canonical procedure for verifying reproducibility.

An `.ipynb` file is a JSON document storing the cells and their outputs. It opens inside VS Code, and in the browser via the `jupyter lab` command (both installable with pip: `pip install jupyterlab`).

### 3-4. The anatomy of an out-of-order accident

Say cell A runs `price = 1000` and cell B runs `price * 2`. Delete cell A afterwards, and cell B still works — because `price` lingers in the kernel's memory. A colleague who receives this notebook and runs it top to bottom hits an error: `price` does not exist. **The code on screen and the kernel's memory can drift apart** — remember just this, and you understand most notebook accidents.

## 4. Hands-on — main.py

```bash
python3 main.py
```

So you can experience how notebooks work without installing Jupyter, we built a **mini notebook simulator**.

- **[1] Notebook setup**: defines four cells of a sales-analysis scenario (prepare data → total → average → report).
- **[2] Run in order**: executes top to bottom like Run All, showing for each cell its execution ordinal `[n]`, its output, and **the kernel memory (variable list) accumulating cell by cell**.
- **[3] Reproduce the out-of-order accident**: restarts the kernel and runs from cell 3, deliberately triggering the "skip the cells above and get a NameError," then runs cells in a different order from the screen and shows how the results go askew.
- **[4] Lessons recap**: prints why Restart & Run All is the canonical procedure.

The key code is the `NotebookKernel` class. A single dictionary `self.memory` is the kernel's memory, and `run_cell()` executes cell code on top of that memory with Python's built-in `exec()`. It is a real Jupyter kernel's mechanics scaled down to one dictionary.

## 5. Try it yourself

1. **(Easy)** Install VS Code, open this curriculum folder via "Open Folder," then open and run this main.py. *(Hint: the ▶ button at the top right. Don't forget to select `.venv` as the interpreter.)*
2. **(Medium)** Add a "find the top-revenue store" cell to `CELLS` in main.py so it appears in the Run All output. *(Hint: use `max(revenues)`, and feel free to reuse variables made by earlier cells — that is the kernel's shared memory.)*
3. **(Challenge)** Imitating step [3], pick another execution order (say 4→2→3), predict what error or skewed result appears, then verify with code. *(Hint: only the order list inside `simulate_out_of_order()` needs changing.)*

## 6. Common mistakes

- **Saving and sharing a notebook whose cells ran in a jumble**: the recipient cannot reproduce it. Before sharing, always confirm it runs cleanly start-to-finish with "Restart & Run All."
- **Not selecting the editor's interpreter**: if things work in the terminal but imports fail only in VS Code, nine times out of ten the interpreter is not the `.venv`.
- **Doing everything in notebooks**: notebooks are for experiments. Keeping your daily automation in one makes execution order and version control both painful. Move settled logic into `.py` files.
- **Blaming the code when the kernel died**: if output stops appearing, first check the kernel's state (does it need a restart?). A kernel with wiped memory needs the upper cells re-run first.

## Coming up next

The tools are in place — now, **how to protect your work**. In level06 we learn Git, the version-control tool that ends the tragedy of "final_REALfinal_edit2.xlsx." Through hands-on git commands you will see that a commit is an approval stamp, and a branch is time travel.
