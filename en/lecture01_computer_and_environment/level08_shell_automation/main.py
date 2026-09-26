"""
Lecture 01 / Level 08 — Automating Repetitive Work with Shell Scripts

A hands-on exercise that automatically tidies a messy shared folder
(recreated in a temporary directory). Following automation's standard
pattern — 'collect -> decide -> (dry run) -> execute -> report' — we parse
file names with regular expressions, bulk-move files into per-store,
per-month folders, and rename them to a unified convention.
"""

import random
import re
import tempfile
from pathlib import Path

# Files recreating the messy shared folder — naming conventions all over the place!
MESSY_FILES = [
    "SalesReport_Downtown_2024-03.csv", "SalesReport_Downtown_2024-04.csv",
    "SalesReport_Riverside_2024-03.csv", "sales_lakeside_2024-03.csv",
    "sales_lakeside_2024-04.csv", "SalesReport_Airport_2024-04.csv",
    "sales_airport_2024-03.csv", "SalesReport_Riverside_2024-04.csv",
    "memo.txt", "lunch_poll.txt", "slides.pptx", "old_backup.zip",
    "SalesReport_Downtown_2024-05.csv", "sales_lakeside_2024-05.csv",
]

# File-name parsing rules: two regular expressions extracting the 'store' and 'year-month'
PATTERNS = [
    re.compile(r"SalesReport_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
    re.compile(r"sales_(?P<store>.+)_(?P<ym>\d{4}-\d{2})\.csv"),
]
# lowercase store names -> unified spelling
STORE_CANONICAL = {"lakeside": "Lakeside", "airport": "Airport"}


def create_mess(folder: Path):
    """[1] Make the mess: create the inconsistent files with seed-fixed random contents."""
    rng = random.Random(42)  # fixed seed — the same contents on every run
    for name in MESSY_FILES:
        revenue = rng.randint(500, 2000)
        (folder / name).write_text(f"revenue,{revenue}000\n", encoding="utf-8")


def plan_moves(folder: Path):
    """[2] Collect & decide: for each file, plan only 'where to, under what name'.

    Keeping this separate from execution gives us the dry run (rehearsal) for free."""
    plans = []  # (source Path, destination Path, classification reason)
    for path in sorted(folder.glob("*")):        # collect: the target list
        if path.is_dir():
            continue
        for pattern in PATTERNS:                 # decide: apply the rules
            matched = pattern.match(path.name)
            if matched:
                store = STORE_CANONICAL.get(matched["store"], matched["store"])
                ym = matched["ym"]
                # Destination: storefolder/store_YYYY-MM.csv — name unified too
                dest = folder / store / f"{store}_{ym}.csv"
                plans.append((path, dest, f"sales file -> {store}/{ym}"))
                break
        else:  # matches no rule -> into the review bin for a human to look at
            dest = folder / "_needs_review" / path.name
            plans.append((path, dest, "no rule matched -> review bin"))
    return plans


def execute_moves(plans):
    """[4] Execute: create the folders and bulk-move according to the plan."""
    moved = 0
    for src, dest, _ in plans:
        dest.parent.mkdir(parents=True, exist_ok=True)  # the shell's mkdir -p
        if dest.exists():                                # prevent overwrite accidents!
            dest = dest.with_name(dest.stem + "_dup" + dest.suffix)
        src.rename(dest)                                 # the shell's mv
        moved += 1
    return moved


def draw_tree(folder: Path):
    """Print the cleanup result as a tree."""
    for p in sorted(folder.rglob("*")):
        depth = len(p.relative_to(folder).parts) - 1
        tag = "/" if p.is_dir() else ""
        print("      " + "  " * depth + f"- {p.name}{tag}")


def main():
    print("=" * 60)
    print("File cleanup automation — collect -> decide -> dry run -> execute -> report")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        shared = Path(tmp) / "shared_folder"
        shared.mkdir()

        create_mess(shared)
        print(f"\n[1] The mess, recreated: {len(MESSY_FILES)} files with clashing naming conventions")
        print("    " + ", ".join(MESSY_FILES[:6]) + " ...")

        plans = plan_moves(shared)
        print(f"\n[2] Collect & decide: file names parsed with regexes — {len(plans)} moves planned")

        # [3] Dry run: verify the plan by eye before executing — automation's #1 safety net
        print("\n[3] Dry run (rehearsal) — nothing has been moved yet")
        for src, dest, reason in plans:
            print(f"    {src.name:<36} -> {dest.relative_to(shared)}  ({reason})")

        moved = execute_moves(plans)
        print(f"\n[4] Real execution: {moved} files moved and renamed")

        print("\n[5] Report — folder structure after cleanup:")
        draw_tree(shared)
        review = sum(1 for _, dest, _ in plans if "_needs_review" in str(dest))
        print(f"\n    summary: auto-sorted {moved - review} / needing human review {review}")
        print("    By hand: 30 seconds per file x 14 files = about 7 minutes.")
        print("    The script: 0.1 seconds — and re-running it every week costs nothing.")

    print("\nRecap: separate planning (plan) from execution (execute) and the dry run comes free.")
    print("       Bulk operations always go 'verify the plan -> execute'!")


if __name__ == "__main__":
    main()
