"""
Lecture 01 / Level 02 — First Steps in the Terminal

Experience how the pwd/ls/cd/mkdir/cp/mv/rm commands work using a
'mini shell simulator' written in Python, safely inside a temporary folder.
The code structure demonstrates that a shell is, in the end,
'an interpreter that remembers the current location (cwd) and does file
operations on your behalf.'
"""

import shutil
import tempfile
from pathlib import Path


class MiniShell:
    """A class imitating the core behavior of a real shell (managing the current location + file commands)."""

    def __init__(self, root: Path):
        # resolve(): the absolute path with symlinks unwrapped (guards against macOS /var -> /private/var)
        self.root = root.resolve()   # top of the practice area (we never step outside it)
        self.cwd = self.root      # the current working directory — the shell's core state!

    def run(self, line: str) -> str:
        """Parse a line like 'cp a.txt backup' and call the matching method."""
        parts = line.split()
        command, args = parts[0], parts[1:]
        handler = getattr(self, f"cmd_{command}", None)  # cmd_ls, cmd_cd ...
        if handler is None:
            return f"(error) '{command}': command not found"
        return handler(*args)

    # ---- Each command's implementation: ultimately just pathlib calls ----
    def cmd_pwd(self):
        rel = self.cwd.relative_to(self.root)
        return "/" + str(rel) if str(rel) != "." else "/ (top of the practice area)"

    def cmd_ls(self):
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in self.cwd.iterdir())
        return "  ".join(names) if names else "(empty)"

    def cmd_cd(self, name):
        target = (self.cwd / name).resolve()
        if not target.is_dir():
            return f"(error) '{name}': no such folder"
        self.cwd = target  # 'moving' is nothing more than changing the current-location variable!
        return f"moved -> {self.cmd_pwd()}"

    def cmd_mkdir(self, name):
        (self.cwd / name).mkdir()
        return f"folder '{name}' created"

    def cmd_cp(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src  # copying into a folder keeps the same name
        shutil.copy(self.cwd / src, dst_path)
        return f"copied '{src}' -> '{dst}' (the original stays put)"

    def cmd_mv(self, src, dst):
        dst_path = self.cwd / dst
        if dst_path.is_dir():
            dst_path = dst_path / src
        (self.cwd / src).rename(dst_path)
        return f"moved '{src}' -> '{dst}' (within the same folder, that's a rename)"

    def cmd_rm(self, name):
        (self.cwd / name).unlink()
        return f"'{name}' deleted — gone immediately, no recycle bin!"


# Practice scenario: (command, explanation of what it does)
SCENARIO = [
    ("pwd", "Check where I am standing — the number-one habit of terminal work"),
    ("ls", "Check what is in this room (folder)"),
    ("mkdir backup", "Install a new cabinet (folder) for backups"),
    ("cp weekly_report.txt backup", "File a copy of the report in the backup folder"),
    ("cd backup", "Step into the backup folder"),
    ("ls", "Confirm the copy arrived safely"),
    ("mv weekly_report.txt weekly_report_backup.txt", "mv within the same folder = a rename"),
    ("cd ..", "'..' = one level up (the parent folder)"),
    ("rm scratch_note.txt", "Delete a note we no longer need (immediate and permanent!)"),
    ("ls", "Check the final state of the tidied room"),
]


def main():
    print("=" * 60)
    print("First steps in the terminal — mini shell simulator")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        office = Path(tmp) / "practice_office"
        office.mkdir()

        # [1] Set up the practice area: lay out sample files (a temp folder, so mistakes are safe)
        for name, content in [
            ("weekly_report.txt", "This week's sales summary"),
            ("meeting_notes.txt", "September regular meeting"),
            ("scratch_note.txt", "Lunch menu candidates"),
        ]:
            (office / name).write_text(content, encoding="utf-8")
        print(f"\n[1] Practice area ready: 3 sample files created in a temporary folder")
        print("    (none of the real files on your computer are touched)")

        # [2] Run the scenario: one command at a time, 'input -> execution -> explanation'
        shell = MiniShell(office)
        print("\n[2] Running the command scenario (lines marked $ are 'what you typed')")
        for step, (line, note) in enumerate(SCENARIO, start=1):
            print(f"\n  ({step}) $ {line}")
            print(f"      -> {shell.run(line)}")
            print(f"      note: {note}")

        # [3] Final state and recap
        n_ops = len(SCENARIO)
        print("\n[3] Final folder structure:")
        for p in sorted(office.rglob("*")):
            depth = len(p.relative_to(office).parts) - 1
            tag = "/" if p.is_dir() else ""
            print("      " + "  " * depth + f"- {p.name}{tag}")
        print(f"\n    Done in {n_ops} command lines. With a mouse: opening windows, dragging, clicking names…")
        print("    And these {n} lines, once saved, can be replayed tomorrow — or by your successor — exactly as-is.".format(n=n_ops))

    print("\nRecap: a shell = an interpreter that remembers 'where you are' and handles files for you.")
    print("       Now open a real terminal and try pwd, ls, and cd yourself!")


if __name__ == "__main__":
    main()
