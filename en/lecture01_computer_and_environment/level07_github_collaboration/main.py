"""
Lecture 01 / Level 07 — GitHub — Remote Repositories and Collaboration

Practice remote collaboration without the internet. We create a bare
repository (the HQ archive) in a temporary folder to serve as a 'fake GitHub',
then automatically run a collaboration scenario in which developers A and B
exchange documents via clone -> push -> pull.
If git is missing, we fall back to installation guidance plus a concept
simulation.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path


def run(args, cwd, actor="", note=""):
    """Run a git command and print 'who (the name before $) did what'."""
    label = f"[{actor}] " if actor else ""
    print(f"\n  {label}$ git {' '.join(args)}")
    if note:
        print(f"      note: {note}")
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    for line in output.splitlines()[:8]:
        print(f"      | {line}")
    return output


def setup_identity(repo, name):
    """Register a practice commit identity (valid inside this repository only)."""
    run(["config", "user.name", name], repo)
    run(["config", "user.email", f"{name}@example.com"], repo)


def list_files(repo, actor):
    """Show the file listing of a branch office's cabinet (the working folder)."""
    files = sorted(p.name for p in Path(repo).iterdir() if p.is_file())
    print(f"      [{actor}] office folder contents: {files if files else '(empty)'}")


def real_remote_demo():
    """Practice the push/pull collaboration flow with a bare remote + two offices (A and B)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        hq = base / "headquarters.git"   # HQ central archive (bare)
        dev_a = base / "dev_a"           # developer A's branch office
        dev_b = base / "dev_b"           # developer B's branch office

        # [1] Found HQ: a bare repository = history only, no work desks (the actual shape of GitHub's servers)
        print("\n[1] Founding HQ — creating the bare remote repository")
        run(["init", "--bare", "-b", "main", str(hq)], base,
            note="--bare: a central archive with no working folder. This is our 'fake GitHub'")

        # [2] Open the branches: A and B each clone
        print("\n[2] Opening the branch offices — developers A and B clone")
        run(["clone", str(hq), str(dev_a)], base, "A", "duplicate the whole HQ archive (still empty)")
        run(["clone", str(hq), str(dev_b)], base, "B", "B opens an office too")
        setup_identity(dev_a, "dev-a")
        setup_identity(dev_b, "dev-b")

        # [3] A pushes: create file -> commit -> send up to HQ
        print("\n[3] A's work and push")
        (dev_a / "menu.txt").write_text("Americano KRW 4000\n", encoding="utf-8")
        run(["add", "menu.txt"], dev_a, "A")
        run(["commit", "-m", "Draft the menu"], dev_a, "A", "the office approval (commit) exists only on A's computer so far")
        run(["push", "origin", "main"], dev_a, "A", "push = dispatch the approved documents to the HQ archive")

        # [4] B pulls: fetch the latest documents from HQ — the core moment of collaboration
        print("\n[4] B pulls — A's work arrives at B's office")
        list_files(dev_b, "B (before pull)")
        run(["pull", "origin", "main"], dev_b, "B", "pull = fetch and merge HQ's new documents")
        list_files(dev_b, "B (after pull)")
        print("      -> the menu.txt that A created has appeared in B's office!")

        # [5] The other direction: B adds -> push, A updates via pull
        print("\n[5] The reverse direction — B adds and A receives")
        menu_b = dev_b / "menu.txt"
        menu_b.write_text(menu_b.read_text(encoding="utf-8") + "Latte KRW 4500\n", encoding="utf-8")
        run(["add", "."], dev_b, "B")
        run(["commit", "-m", "Add latte"], dev_b, "B")
        run(["push", "origin", "main"], dev_b, "B")
        run(["pull", "origin", "main"], dev_a, "A", "pull first, push later — collaboration etiquette")
        print(f"      [A] final menu: {(dev_a / 'menu.txt').read_text(encoding='utf-8').splitlines()}")
        print("      -> both offices are now on the same latest state.")

        # [6] On real GitHub: the PR flow explained
        print("\n[6] In real GitHub collaboration, 'review' slots into this flow:")
        for step in [
            "1) create a branch, work, and push",
            "2) open a Pull Request (the approval request) on the GitHub website",
            "3) colleagues review — comments and change requests go back and forth",
            "4) once approved, merge into main (joining the main storyline)",
            "5) the whole team updates with git pull",
        ]:
            print(f"      {step}")


def fallback_simulation():
    """When git is missing: simulate the concepts in text only."""
    print("\n[Notice] The git command was not found.")
    print("  Mac: 'xcode-select --install' / Windows: install from git-scm.com, then re-run.")
    print("\n[Concept simulation] A day of remote collaboration:")
    events = [
        ("HQ", "central archive founded (a bare repository)"),
        ("A", "clone — open a branch office by duplicating the whole archive"),
        ("A", "commit then push — dispatch approved documents to HQ"),
        ("B", "pull — fetch HQ's new documents and update the office"),
        ("B", "more work then push; A pulls again"),
    ]
    for actor, action in events:
        print(f"  [{actor:^4}] {action}")
    print("\n  Whether the address is an internet URL (GitHub) or a local folder, the mechanics are identical.")


def main():
    print("=" * 60)
    print("Remote repository collaboration — HQ (remote) and two offices (A and B)")
    print("=" * 60)
    if shutil.which("git"):
        real_remote_demo()
    else:
        fallback_simulation()

    print("\nRecap: start with clone, send up with push, receive with pull.")
    print("       Pull first and push small-and-often — the etiquette that reduces conflicts.")


if __name__ == "__main__":
    main()
