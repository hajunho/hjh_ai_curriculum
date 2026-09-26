"""
Lecture 01 / Level 06 — Git — The Idea of Version Control

Creates a real Git repository in a temporary folder and automatically walks
the init -> add -> commit -> diff -> log -> branch flow. The analogies —
commit = approval stamp, branch = parallel universe — are verified through
actual command output.
If git is not installed, we fall back to installation guidance plus a
concept simulation.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run_git(args, cwd, note=""):
    """Run a git command and show it as '$ command -> output'."""
    print(f"\n  $ git {' '.join(args)}")
    if note:
        print(f"    note: {note}")
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:12]:  # only the head if it gets too long
        print(f"    | {line}")
    return output


def real_git_demo():
    """When git is installed: practice the whole flow with real commands in a temp folder."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "cafe_project"
        repo.mkdir()
        price_file = repo / "price_list.txt"

        # [1] Install the cabinet: init + a practice identity (valid inside this repo only)
        print("\n[1] Installing the cabinet — git init")
        run_git(["init", "-b", "main"], repo, "start tracking this folder's history (creates the .git cabinet)")
        run_git(["config", "user.name", "Trainee"], repo, "the name engraved on the stamp (practice-only, this repo only)")
        run_git(["config", "user.email", "student@example.com"], repo, "the email engraved on the stamp")

        # [2] First approval: create file -> add (tray) -> commit (stamp)
        print("\n[2] First approval — add and commit")
        price_file.write_text("Americano KRW 4000\nLatte KRW 4500\n", encoding="utf-8")
        print(f"    (file created: {price_file.name})")
        run_git(["status", "--short"], repo, "?? = a new file the cabinet doesn't know yet")
        run_git(["add", "price_list.txt"], repo, "placed on the approval tray (staging)")
        run_git(["commit", "-m", "Draft the price list"], repo, "stamp! this moment is preserved forever")

        # [3] Edit and diff: check what changed, then a second commit
        print("\n[3] Edit and diff — inspecting the changes")
        price_file.write_text("Americano KRW 4200\nLatte KRW 4500\n", encoding="utf-8")
        print("    (Americano price changed 4000 -> 4200)")
        run_git(["diff"], repo, "- is the old content, + is the new")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "Raise Americano price (4000->4200)"], repo, "the trick is putting the 'why' in the message")

        # [4] Browse the history: the accumulated stamps
        print("\n[4] Browsing the history — git log")
        run_git(["log", "--oneline"], repo, "the short code up front is the commit serial number (hash)")

        # [5] Branch: experiment in a parallel universe, then return to the main story
        print("\n[5] Branch — a parallel-universe experiment")
        run_git(["switch", "-c", "experiment"], repo, "create the experiment universe and move into it")
        price_file.write_text("Americano KRW 9900 (experiment!)\nLatte KRW 4500\n", encoding="utf-8")
        run_git(["add", "."], repo)
        run_git(["commit", "-m", "Experiment: premium pricing policy"], repo)
        print(f"    price list in the experiment universe: {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        run_git(["switch", "main"], repo, "back to the main story (main) — time travel!")
        print(f"    price list in the main universe       : {price_file.read_text(encoding='utf-8').splitlines()[0]}")
        print("    -> same file, yet switching branches restored the contents.")
        print("       If the experiment fails, discard just that universe — the main story is unharmed.")


def fallback_simulation():
    """When git is missing: installation guidance + a commit-concept simulation."""
    print("\n[Notice] The git command was not found.")
    print("  Install: on a Mac, 'xcode-select --install' or 'brew install git';")
    print("           on Windows, we recommend Git for Windows from git-scm.com.")
    print("  Re-run this script after installing to practice with real commands.")
    print("\n[Concept simulation] commit = a snapshot of the folder's state + an approval stamp")
    history = [
        ("a1f9c02", "Draft the price list", {"price_list.txt": "Americano KRW 4000"}),
        ("b7e3d11", "Raise Americano price (4000->4200)", {"price_list.txt": "Americano KRW 4200"}),
    ]
    for i, (commit_hash, message, snapshot) in enumerate(history, start=1):
        print(f"\n  commit {i}: [{commit_hash}] \"{message}\"")
        for fname, content in snapshot.items():
            print(f"    stored snapshot: {fname} -> '{content}'")
    print("\n  Git's power is that the folder can be wound back to any of these stamps.")


def main():
    print("=" * 60)
    print("Git hands-on — approval stamps (commits) and parallel universes (branches)")
    print("=" * 60)
    if shutil.which("git"):
        print("\n(git found — we'll create a real repository in a temporary folder.")
        print(" None of your other folders or repositories are touched.)")
        real_git_demo()
    else:
        fallback_simulation()

    print("\nRecap: commit small and often, and write the 'why' in the message.")
    print("       Anything committed can be recovered, no matter what mistake follows.")


if __name__ == "__main__":
    main()
