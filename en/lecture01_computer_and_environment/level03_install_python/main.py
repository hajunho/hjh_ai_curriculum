"""
Lecture 01 / Level 03 — Installing Python and Running It for the First Time

A 'health check' script that diagnoses the Python environment on your machine.
It verifies the version, executable location, operating system, and PATH,
and contrasts REPL (interactive) execution with script execution in its output.
"""

import os
import platform
import sys

# The minimum Python version this curriculum requires (change it in exercise 2 of "Try it yourself")
REQUIRED = (3, 10)


def check_python_identity():
    """[1] Verify the interpreter's identity: version and executable location."""
    version = sys.version_info  # (major, minor, micro, ...) — comparable as numbers
    ok = version >= REQUIRED
    print("[1] Python identity check")
    print(f"    version             : {platform.python_version()}")
    print(f"    executable location : {sys.executable}")
    print(f"    required version    : {REQUIRED[0]}.{REQUIRED[1]} or later")
    print(f"    verdict             : {'pass — ready for the curriculum' if ok else 'below requirement — upgrade needed'}")
    return ok


def check_workplace():
    """[2] Check the workplace: operating system and current working directory."""
    os_name = platform.system()  # 'Darwin' (Mac) / 'Windows' / 'Linux'
    friendly = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(os_name, os_name)
    print("\n[2] Work environment check")
    print(f"    operating system    : {friendly} ({platform.release()})")
    print(f"    processor           : {platform.machine()}")
    print(f"    current work folder : {os.getcwd()}")
    print("    -> every relative path is interpreted from this folder (level01 recap)")
    return True


def check_path():
    """[3] A look at PATH: the folder list the shell searches for commands."""
    raw = os.environ.get("PATH", "")
    entries = [e for e in raw.split(os.pathsep) if e]
    print("\n[3] PATH — the folders the shell combs through for commands like 'python3'")
    print(f"    registered folders  : {len(entries)} (searched front to back)")
    for i, entry in enumerate(entries[:5], start=1):
        print(f"    #{i}: {entry}")
    if len(entries) > 5:
        print(f"    ... and {len(entries) - 5} more")
    print("    -> 'command not found' = that name appears nowhere in this list")
    return len(entries) > 0


def compare_repl_vs_script():
    """[4] REPL (on-the-spot conversation) vs. script (written instructions)."""
    print("\n[4] Comparing the two ways to run Python")
    print("    (a) REPL — the '>>>' conversation you get by typing just python3:")
    print("        >>> 120 * 12")
    print(f"        {120 * 12}")
    print("        >>> 'report' + '_final.xlsx'")
    print(f"        '{'report' + '_final.xlsx'}'")
    print("    (b) Script — right now! Everything you see here is the result of handing")
    print("        the document main.py to python3 and running it top to bottom.")
    print("    -> experiments in the REPL; official work as scripts (storable, shareable, re-runnable)")
    return True


def main():
    print("=" * 60)
    print("Python environment self-diagnosis — is my interpreter ready?")
    print("=" * 60 + "\n")

    results = {
        "Python version": check_python_identity(),
        "Operating system check": check_workplace(),
        "PATH setup": check_path(),
        "Execution modes understood": compare_repl_vs_script(),
    }

    # [5] Overall verdict table
    print("\n[5] Diagnosis summary")
    for item, ok in results.items():
        mark = "[OK]  " if ok else "[WARN]"
        print(f"    {mark} {item}")
    if all(results.values()):
        print("\n    Congratulations! This computer is ready for the curriculum.")
    else:
        print("\n    Fix the [WARN] items and run the script again.")

    print("\nRecap: whenever the environment misbehaves, make it a habit to check")
    print("       sys.executable (which Python?) and python3 --version (which version?) first.")


if __name__ == "__main__":
    main()
