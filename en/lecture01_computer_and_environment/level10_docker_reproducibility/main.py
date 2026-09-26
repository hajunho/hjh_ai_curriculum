"""
Lecture 01 / Level 10 — Docker — Reproducing an Entire Environment

An exercise that teaches Docker without needing Docker.
1) diagnoses whether Docker is installed, 2) auto-generates Dockerfile text
from project specifications (dictionaries) and prints it with line-by-line
commentary, 3) saves the generated Dockerfiles into outputs/, and 4) walks
through how the layers stack during a build.
"""

import os
import shutil
import subprocess
from pathlib import Path

# Two project specifications to generate Dockerfiles from (change them in "Try it yourself")
SPECS = {
    "data-analysis batch": {
        "python_version": "3.12",
        "packages": ["pandas", "matplotlib"],
        "entry": "analyze.py",
    },
    "web API server": {
        "python_version": "3.12",
        "packages": ["fastapi", "uvicorn"],
        "entry": "server.py",
    },
}

# One-line commentary per command (the standardized-cargo analogy)
LINE_NOTES = {
    "FROM": "choose the base cargo — start from a mini Linux with Python installed",
    "WORKDIR": "set the workbench location inside the box",
    "COPY": "load files from my computer into the box",
    "RUN": "assembly work performed during packing (the build)",
    "CMD": "the switch flipped automatically on unpacking (startup)",
}


def diagnose_docker():
    """[1] Docker installation diagnosis: version check if present, guidance if not."""
    print("[1] Docker installation diagnosis")
    docker_path = shutil.which("docker")
    if docker_path:
        print(f"    docker command found: {docker_path}")
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=False)
        version = (result.stdout or result.stderr).strip()
        print(f"    version: {version if version else '(check failed — Docker Desktop may be off)'}")
        print("    -> you can try 'docker build' with the Dockerfiles saved in [3]")
        return True
    print("    docker command not found — that's fine, today's exercise proceeds without Docker.")
    print("    To install: Docker Desktop from docker.com (Mac and Windows alike)")
    print("    (on a company computer, check the installation policy first)")
    return False


def generate_dockerfile(spec):
    """[2] Generate the packing instructions: a spec dictionary -> Dockerfile text.

    A Dockerfile is, in the end, 'text with rules', so string assembly can build it."""
    lines = [
        f"FROM python:{spec['python_version']}-slim",
        "WORKDIR /app",
        # Why copy requirements first: to exploit the layer cache!
        # If only the code changes, the package-installation layer is reused and builds get fast.
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "COPY . .",
        f'CMD ["python", "{spec["entry"]}"]',
    ]
    return "\n".join(lines) + "\n"


def explain_dockerfile(name, spec):
    """Print the generated Dockerfile with line-by-line commentary."""
    text = generate_dockerfile(spec)
    print(f"\n    ── Dockerfile for the {name} ──")
    print(f"    (requirements.txt: {', '.join(spec['packages'])})")
    for line in text.splitlines():
        keyword = line.split()[0] if line.strip() else ""
        note = LINE_NOTES.get(keyword, "")
        print(f"    {line:<52} # {note}" if note else f"    {line}")
    return text


def main():
    print("=" * 60)
    print("Docker concepts hands-on — reading and writing the packing instructions (Dockerfile)")
    print("=" * 60 + "\n")

    diagnose_docker()

    # [2] Generate a Dockerfile per spec, with commentary
    print("\n[2] Dockerfile generator — the instructions change with the project spec")
    generated = {name: explain_dockerfile(name, spec) for name, spec in SPECS.items()}
    print("\n    Comparison point: the two sets of instructions differ only in CMD (the entry file)")
    print("    and the package list — the structure (FROM->WORKDIR->COPY->RUN->COPY->CMD) is conventionally identical.")

    # [3] Save into outputs/ — so you can actually build once Docker is installed
    out_dir = Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    print("\n[3] Saving the generated instructions to files")
    for name, text in generated.items():
        slug = "analysis" if "analysis" in name else "webapi"
        dockerfile = out_dir / f"Dockerfile.{slug}"
        dockerfile.write_text(text, encoding="utf-8")
        print(f"    saved: {dockerfile}")
    print("    -> after installing Docker, try packing with 'docker build -f Dockerfile.analysis -t myapp .'")

    # [4] If a build ran: how the layers stack
    print("\n[4] Build-flow walkthrough — layers stack up into an image")
    steps = [
        ("1/6 FROM", "download the base image (first time only, cached afterwards)"),
        ("2/6 WORKDIR", "create the working-folder layer"),
        ("3/6 COPY req...", "the requirements.txt layer — cache reused if the file is unchanged"),
        ("4/6 RUN pip...", "the package-installation layer — slowest, but 0 seconds when cached"),
        ("5/6 COPY . .", "the code layer — code changes often, which is why it goes last"),
        ("6/6 CMD", "record the startup switch -> image complete (packing done)"),
    ]
    for step, note in steps:
        print(f"    [{step:<14}] {note}")
    print("    Once the image is complete: 'docker run' = unpack and operate the cargo (start a container)")

    print("\nRecap: Dockerfile (instructions) -> build -> image (packed cargo) -> run -> container (running).")
    print("       Turn the whole environment into cargo and it runs identically at any port (computer).")


if __name__ == "__main__":
    main()
