# Lecture 01 · Level 04 — Virtual Environments and Package Management

> Learn to create an independent Python workshop (venv) per project and to procure the tools you need (packages) with pip.

**Difficulty** ⭐⭐ / **Prerequisites** level03 / **Estimated time** 40 min

## 1. Why learn this — the business view

There is an old joke in the software industry: "but it works on my machine!" — developer A's code refuses to run on developer B's computer. The cause, most of the time, is that the two have installed **different sets or versions of packages**. A was using library version 2.0, B was on 1.5.

The problem hits individuals too. Project X needs an old version of a library while new project Y needs the latest — and if only one can be installed machine-wide, one of the two projects breaks. The solution is the **virtual environment**: one independent Python installation space per project. Real-world Python projects begin with a virtual environment without exception, and this curriculum's repository has one too, named `.venv`. Finish this level and you will always be able to answer "which environment am I in right now, and with which tools?"

## 2. Understanding through an analogy

**Packages are specialist tools; a virtual environment is a project-specific toolbox.**

Installing Python comes with the basic tools (the standard library) — hammer-and-screwdriver essentials. But data analysis calls for a power drill (pandas), and charting for painting equipment (matplotlib). The procurement officer who fetches such tools from the worldwide shared warehouse (PyPI, the Python Package Index) is **pip**.

Trouble starts when all tools go into **one company-wide toolbox**. The renovation team needs the old drill standard, the new-construction team needs the new one — and the box can hold only one kind. The moment one team swaps the tool, the other team's work grinds to a halt.

So a smart company kits out **a dedicated toolbox per project**. That is the virtual environment. You create a box named `.venv` inside the project folder and put only that project's Python and packages in it. Boxes are fully independent, so they cannot break each other, and when the project ends you simply throw the box away. "Activating" is **strapping on that project's toolbox before starting work**.

## 3. Core concepts

### 3-1. The four-stage life cycle of a virtual environment

```bash
# 1) Create — once, in the project folder
python3 -m venv .venv

# 2) Activate — every time you start working
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (PowerShell)

# 3) Install tools — while activated
pip install requests

# 4) Deactivate — when you finish
deactivate
```

Once activated, `(.venv)` appears in front of your prompt. That marker is the visual signal of "which toolbox am I wearing right now." While activated, both `python` and `pip` switch to the ones inside that box.

### 3-2. Essential pip commands

| Command | What it does |
|---|---|
| `pip install name` | Install a package from the shared warehouse (PyPI) |
| `pip install name==2.1.0` | Install a specific version |
| `pip list` | List packages installed in this environment |
| `pip uninstall name` | Remove |
| `pip freeze > requirements.txt` | Record the current tool list to a file |
| `pip install -r requirements.txt` | Rebuild from that list in one go |

`requirements.txt` is **the toolbox's inventory sheet**. Hand over just this file and a colleague can reproduce a box with the identical contents on their own computer. It is the first vaccine against "but it works on my machine" (the second vaccine is Docker, in level10).

### 3-3. How to know which environment you are in

There is a reliable way to check from inside Python code:

- `sys.prefix` — where Python is currently looking for its tools
- `sys.base_prefix` — where the original (system) Python lives
- If the two **differ, you are inside a virtual environment**; if they match, you are on the system Python.

In the analogy, `sys.base_prefix` is the address of the company's central tool warehouse, and `sys.prefix` is the address of the box strapped to your waist. Different addresses mean you are wearing a dedicated box.

### 3-4. Conventions to follow

- By convention, name the virtual-environment folder `.venv` and put it directly under the project folder.
- Never commit the `.venv` folder to Git (it is large and differs per machine). Commit `requirements.txt` instead.
- Don't scatter `pip install` onto the system Python. Recent operating systems may block it outright (the externally-managed error). Build the habit of installing only inside a virtual environment.

## 4. Hands-on — main.py

```bash
python3 main.py
```

The script prints **a diagnosis of the environment it is running in**. Run it with the repository's `.venv` Python (`../../.venv/bin/python main.py`, or `python main.py` after activating) and the verdict says "inside a virtual environment"; run it with the system `python3` and it says "system Python" — compare the two yourself.

- **[1] Which toolbox?**: prints `sys.prefix` and `sys.base_prefix` side by side and compares them to decide whether you are in a virtual environment.
- **[2] Toolbox contents**: lists installed packages and versions via `importlib.metadata` — the same information `pip list` shows, obtained from code.
- **[3] Standard tools check**: confirms a few basics that need no installation (json, csv, datetime, and friends) are already present.
- **[4] requirements.txt preview**: prints the current environment's package list in `name==version` form, unmasking the inventory sheet that `pip freeze` produces.

The key code is the one-line comparison `sys.prefix != sys.base_prefix` and the loop over `importlib.metadata.distributions()`. Python can inspect its own environment from its own code — which is exactly why environment problems are nothing to fear.

## 5. Try it yourself

1. **(Easy)** Run the same main.py (a) with the system `python3` and (b) with the repository `.venv` Python, and compare the verdict in [1] and the package count in [2]. *(Hint: you can point at the exact Python directly, e.g. `.venv/bin/python main.py`.)*
2. **(Medium)** In any scratch folder, create a fresh virtual environment with `python3 -m venv test_env`, activate it, confirm with `pip list` that the box is nearly empty, then `deactivate` and delete the whole folder. *(Hint: boxes are supposed to be cheap to create and throw away.)*
3. **(Challenge)** In the output of [2], sort the packages by name length instead of alphabetically. *(Hint: change the key function in `sorted(..., key=...)`.)*

## 6. Common mistakes

- **Installing while forgetting to activate**: `pip install` without the `(.venv)` marker installs into the wrong place (the system). Check the prompt before installing.
- **Creating a venv but pointing the IDE at a different Python**: editors (VS Code and friends) need the interpreter selected separately (covered next level). This is the classic cause of "I installed it but import fails."
- **Copying or moving the `.venv` folder wholesale**: virtual environments are tied to the location where they were created and break easily when moved. The proper way to move one is to rebuild it from `requirements.txt`.
- **Forgetting to refresh requirements.txt**: after adding a package, regenerate the inventory with `pip freeze` so your colleagues' environments can follow along.

## Coming up next

The workshop and tools are ready — now for the **instruments on your desk**. In level05 we learn when a code editor (VS Code) fits and when a Jupyter notebook does, and we imitate the notebook's "cell execution" concept in plain Python.
