"""
Lecture 01 / Level 01 — Files, Folders, and Paths

Build and explore the document folder tree of a fictional company,
'Hanbit Trading', using pathlib. We practice the difference between absolute
and relative paths, count files per extension, and dissect a path
(parent/stem/suffix) — all safely inside a temporary folder that is
deleted automatically when the script ends.
"""

import os
import tempfile
from collections import Counter
from pathlib import Path


def build_company_tree(root: Path):
    """[Build the tree] Create department folders and sample document files."""
    # Folder structure: (relative paths) — mkdir(parents=True) creates intermediate folders in one go
    folders = [
        "general_affairs",
        "sales/contracts",
        "sales/reports",
        "dev/code",
    ]
    for name in folders:
        (root / name).mkdir(parents=True, exist_ok=True)

    # Files: (relative path, contents) — the extension is the 'document-type stamp'.
    files = [
        ("general_affairs/supply_request.txt", "10 ballpoint pens, 2 boxes of A4 paper"),
        ("general_affairs/parking_rules.txt", "Level B2 is reserved for visitors."),
        ("sales/contracts/AcmeCo_contract.txt", "Party A: AcmeCo / Party B: Hanbit Trading"),
        ("sales/contracts/BetaCo_contract.txt", "Party A: BetaCo / Party B: Hanbit Trading"),
        ("sales/reports/march_results.csv", "store,revenue\nDowntown,1512000\nRiverside,1098000"),
        ("sales/reports/april_results.csv", "store,revenue\nDowntown,1620000\nRiverside,1150000"),
        ("dev/code/hello.py", "print('hello')"),
    ]
    for rel_path, content in files:
        (root / rel_path).write_text(content, encoding="utf-8")
    return len(folders), len(files)


def draw_tree(folder: Path, indent: int = 0):
    """[Draw the tree] Print the folder structure as an indented diagram via recursion."""
    marker = "📁" if indent == 0 else "└─"
    print("    " + "   " * indent + f"{marker} {folder.name}/")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            draw_tree(child, indent + 1)  # if it is a folder, call ourselves again (recursion)
        else:
            print("    " + "   " * (indent + 1) + f"└─ {child.name}")


def main():
    print("=" * 60)
    print("Files, folders, and paths — a tour of the Hanbit Trading archive")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "hanbit_corp"
        root.mkdir()

        # [1] Build the folder tree
        n_folders, n_files = build_company_tree(root)
        print(f"\n[1] Tree built: {n_folders} folders (+intermediates), {n_files} files created")
        print(f"    location (temporary folder): {root}")

        # [2] Draw the tree — a Python imitation of the terminal's tree command
        print("\n[2] Full structure of the archive:")
        draw_tree(root)

        # [3] Absolute vs. relative paths
        target = root / "sales" / "reports" / "march_results.csv"
        print("\n[3] Two addresses pointing at the same file:")
        print(f"    absolute path: {target.resolve()}")
        print(f"    relative path (from the company entrance): {target.relative_to(root)}")
        # '..' means one level up (the parent folder).
        sibling = target.parent.parent / "contracts" / "AcmeCo_contract.txt"
        print(f"    from the reports folder, following '../contracts/AcmeCo_contract.txt' leads to:")
        print(f"      -> {sibling.relative_to(root)} (exists? {sibling.exists()})")
        print(f"    current working directory (CWD): {os.getcwd()}")
        print("      -> relative paths are always interpreted from this CWD (or a stated base)")

        # [4] File census per extension — rglob sweeps every subfolder
        counts = Counter(p.suffix for p in root.rglob("*") if p.is_file())
        print("\n[4] Census by extension (the document-type stamp):")
        for ext, count in sorted(counts.items()):
            print(f"    {ext:<6} {count} file(s)")

        # [5] Path anatomy — dissect one path into its parts
        print("\n[5] Path anatomy: sales/reports/march_results.csv")
        print(f"    parent (containing folder)  : {target.parent.name}/")
        print(f"    name   (full file name)     : {target.name}")
        print(f"    stem   (name w/o extension) : {target.stem}")
        print(f"    suffix (extension)          : {target.suffix}")

    print("\n[6] Done: the temporary folder was deleted automatically — no trace left on your computer")
    print("Recap: an absolute path is the full address from the front entrance; a relative path is shorthand from where you stand.")


if __name__ == "__main__":
    main()
