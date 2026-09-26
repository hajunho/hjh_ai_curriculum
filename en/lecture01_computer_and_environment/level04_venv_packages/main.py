"""
Lecture 01 / Level 04 — Virtual Environments and Package Management

Diagnoses the Python environment this very code is running in.
Compares sys.prefix with sys.base_prefix to decide whether we are inside a
virtual environment, lists the installed packages (the toolbox contents)
via importlib.metadata, and unmasks the requirements.txt that pip freeze
produces.
"""

import importlib.metadata
import importlib.util
import sys


def check_which_toolbox():
    """[1] Which toolbox?: decide between a virtual environment and the system Python."""
    prefix = sys.prefix            # where Python is looking for its tools right now
    base = sys.base_prefix         # where the original (system) Python lives
    in_venv = prefix != base       # different = wearing a dedicated toolbox
    print("[1] Which toolbox am I wearing?")
    print(f"    sys.prefix      (space in use)   : {prefix}")
    print(f"    sys.base_prefix (original space) : {base}")
    if in_venv:
        print("    verdict: inside a virtual environment — the project's own toolbox is on")
    else:
        print("    verdict: system Python — using the company-wide shared toolbox")
        print("    (run with the repository's .venv/bin/python and the verdict changes)")
    print(f"    executable: {sys.executable}")
    return in_venv


def list_installed_packages(limit=15):
    """[2] Toolbox contents: list installed package names and versions."""
    dists = sorted(
        ((d.metadata["Name"] or "?", d.version) for d in importlib.metadata.distributions()),
        key=lambda pair: pair[0].lower(),
    )
    print(f"\n[2] Packages installed in this environment: {len(dists)} total (same info as pip list)")
    for name, version in dists[:limit]:
        print(f"    - {name:<28} {version}")
    if len(dists) > limit:
        print(f"    ... and {len(dists) - limit} more")
    return dists


def check_standard_tools():
    """[3] Basic tools (standard library) check: already there, no installation needed."""
    basics = ["json", "csv", "datetime", "sqlite3", "pathlib", "random"]
    print("\n[3] Basic tools check — the standard library works out of the box")
    for name in basics:
        found = importlib.util.find_spec(name) is not None
        print(f"    {'[OK]' if found else '[missing]'} {name}")
    print("    -> lectures 01–02 of this curriculum run entirely on these basic tools")


def preview_requirements(dists, limit=8):
    """[4] requirements.txt preview: the inventory sheet that pip freeze produces."""
    print("\n[4] requirements.txt preview — an inventory sheet in 'name==version' form")
    if not dists:
        print("    (no packages installed, so the list is empty)")
        return
    for name, version in dists[:limit]:
        print(f"    {name}=={version}")
    if len(dists) > limit:
        print(f"    ... and {len(dists) - limit} more lines")
    print("    -> hand over just this file and a colleague runs 'pip install -r requirements.txt'")
    print("       to reproduce a toolbox with the identical contents on their own machine")


def main():
    print("=" * 60)
    print("Virtual environment & package diagnosis — which toolbox am I wearing?")
    print("=" * 60 + "\n")

    in_venv = check_which_toolbox()
    dists = list_installed_packages()
    check_standard_tools()
    preview_requirements(dists)

    # [5] Summary
    print("\n[5] Diagnosis summary")
    print(f"    virtual environment : {'yes (a .venv-style box)' if in_venv else 'no (system Python)'}")
    print(f"    installed packages  : {len(dists)}")
    print("\nRecap: kit out a dedicated toolbox (venv) per project,")
    print("       and share its makeup via the pip freeze inventory (requirements.txt).")
    print("       Half of all 'but it works on my machine' incidents are prevented right here.")


if __name__ == "__main__":
    main()
