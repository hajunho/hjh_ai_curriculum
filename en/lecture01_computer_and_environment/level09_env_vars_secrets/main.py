"""
Lecture 01 / Level 09 — Handling Environment Variables and Secrets Safely

Hands-on with the tools for keeping secrets outside the code:
1) reading environment variables via os.environ, 2) masked output of secrets,
3) a .env file parser (the dotenv mechanics), 4) a mini scanner that catches
hard-coded secrets.
Every key value is a practice fake, and nothing connects to the internet.
"""

import os
import re

# Hard-coded secret detection rules: (description, regex) — a scale model of real security tools
SECRET_PATTERNS = [
    ("API-key shape (sk-...)", re.compile(r"sk-[A-Za-z0-9]{8,}")),
    ("hard-coded password=", re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)),
    ("hard-coded secret/token", re.compile(r"(secret|token)\s*=\s*['\"][^'\"]{6,}['\"]", re.IGNORECASE)),
]

# Example code inspected in [4] — the bad way (secret inside the code) and the good way (name only)
BAD_CODE = '''
# bad_app.py — never write it like this!
api_key = "sk-demo1234567890abcdef"
password = "corp!2024"
db = connect("10.0.0.7", password=password)
'''
GOOD_CODE = '''
# good_app.py — secrets are referenced by name only
import os
api_key = os.environ["LLM_API_KEY"]      # the value lives in an environment variable (the locker)
password = os.environ.get("DB_PASSWORD")  # the code carries only the name tag
'''


def mask(value, show=4):
    """Hide a secret as 'sk-d****5678'. Verify without exposing."""
    if value is None:
        return "(not set)"
    if len(value) <= show * 2:
        return "*" * len(value)
    return value[:show] + "*" * 4 + value[-show:]


def parse_dotenv(text):
    """[3] .env parser: 'name=value' text into a dictionary. The mechanics of the dotenv tool."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):   # skip blank lines and comments
            continue
        name, _, value = line.partition("=")
        result[name.strip()] = value.strip().strip('"').strip("'")
    return result


def scan_for_secrets(code, filename):
    """[4] Mini scanner: find strings in code that have 'the shape of a secret'."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((filename, lineno, label, line.strip()))
    return findings


def main():
    print("=" * 60)
    print("Environment variables and secrets — secrets outside the code, names only inside")
    print("=" * 60)

    # [1] Reading environment variables: set a practice variable in this process and read it back
    os.environ["DEMO_LLM_API_KEY"] = "sk-demo1122334455667788"  # a practice fake value
    print("\n[1] Reading environment variables — the 'name=value' notes the OS hands over")
    print(f"    os.environ['DEMO_LLM_API_KEY'] -> (read OK, value masked below)")
    print(f"    the safe .get() read: a missing variable -> {os.environ.get('NO_SUCH_VAR')}")
    print(f"    length of PATH, a variable you were already using: {len(os.environ.get('PATH', ''))} characters")
    practice = os.environ.get("PRACTICE_KEY")
    print(f"    PRACTICE_KEY: {mask(practice)}  (try exporting it in exercise 1 of 'Try it yourself')")

    # [2] Masked output: reveal only 'whether it is set' and hide the value
    print("\n[2] Masked output — never print a secret whole to logs or screen")
    secret = os.environ["DEMO_LLM_API_KEY"]
    print(f"    printing the raw value   : (absolutely forbidden!)")
    print(f"    printing it masked       : {mask(secret)}")

    # [3] .env parser: how file contents get promoted into environment variables
    print("\n[3] The .env pattern — the project folder's secrets file (NEVER into Git)")
    dotenv_text = '# practice .env contents\nDB_PASSWORD="s3cret!pw"\nSLACK_TOKEN=xoxb-demo-9988\n'
    loaded = parse_dotenv(dotenv_text)
    for name, value in loaded.items():
        os.environ[name] = value           # promote into environment variables (what dotenv does)
        print(f"    {name:<14} = {mask(value)}  -> loaded into os.environ")
    print("    -> the code can now just call the name: os.environ['DB_PASSWORD']")

    # [4] Hard-coding scanner: the alarm should ring only for the bad code
    print("\n[4] Mini hard-coding scanner — the mechanics of the automatic pre-commit check")
    for filename, code in [("bad_app.py", BAD_CODE), ("good_app.py", GOOD_CODE)]:
        findings = scan_for_secrets(code, filename)
        if findings:
            print(f"    {filename}: {len(findings)} alarms!")
            for fname, lineno, label, line in findings:
                print(f"      - line {lineno} [{label}] {line}")
        else:
            print(f"    {filename}: pass — no hard-coded secrets")

    # [5] Rules checklist
    print("\n[5] Three-lines-of-defense checklist")
    print("    [prevent] secrets only in .env/environment variables; .env registered in .gitignore")
    print("    [detect]  scan for 'the shape of a secret' before every commit")
    print("    [respond] if leaked, revoke and reissue the key — deleting the commit is not enough")

    print("\nRecap: don't write the safe combination (the secret) on the front-door note (the code).")
    print("       The manual keeps only the name tag: 'the key is in the locker.'")


if __name__ == "__main__":
    main()
