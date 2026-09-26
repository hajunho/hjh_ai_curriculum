# Lecture 01 · Level 10 — Docker — Reproducing an Entire Environment

> Understand containers — the final answer to "but it works on my machine" — through the standardized shipping-container analogy, and learn to read and write basic Dockerfile syntax.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level04 / **Estimated time** 45 min

## 1. Why learn this — the business view

The virtual environment from level04 unifies your Python packages, but it cannot reach below them. When the operating system differs, the system-level programs differ, and the settings differ, "but it works on my machine" strikes anyway. The AI service finished on a developer's Mac refuses to run on the Linux server — that kind of thing.

**Docker** ends this problem. It packs not just the program but **the entire environment the program runs in — OS libraries, Python, packages, settings, code — into a single box** that runs identically on any computer. It is the de-facto standard for deploying web services and AI models today, so understanding dev-team conversations ("I'll build the image and push it," "I restarted the container") and cloud pricing (billed per container) requires knowing the concept. As a non-engineer you will rarely operate Docker yourself, but **being able to read it** is worth a lot — Docker shows up in job postings, outsourcing quotes, and incident reports alike.

## 2. Understanding through an analogy

**A container is the shipping industry's standardized cargo container.** This analogy is not decoration — it is the actual etymology.

Picture a port before the container was invented. Rice sacks, furniture, machine parts were loaded onto ships each in their own shape. Every ship loaded differently, every port unloaded differently; it was slow and breakage was constant. Then the **standardized container** arrived. Whatever the contents, once packed into the same-size steel box, any crane at any port in the world could handle it identically. Logistics costs collapsed.

Software's Docker container is exactly this. Regardless of the contents (any language, any program), packed into the standard box, it runs identically on any computer (port) with Docker installed.

Mapping the three terms onto logistics:

- **Dockerfile** = **the packing instructions**. A document stating "what goes into the box, in what order."
- **Image** = **the packed, standardized cargo** produced by following the instructions. The original, ready to duplicate and ship.
- **Container** = that cargo **unpacked and actually running**. One image can launch many containers at once (like opening several stores from the same blueprint).

The difference from a venv also becomes sharp in this analogy: a venv "unifies only the toolbox"; Docker "packs up the whole workshop, building and all."

## 3. Core concepts

### 3-1. Dockerfile syntax — six commands and you can read one

```dockerfile
FROM python:3.12-slim            # base cargo: start from a mini Linux with Python installed
WORKDIR /app                     # set the work location inside the box
COPY requirements.txt .          # copy my file into the box
RUN pip install -r requirements.txt   # a command run during packing (tool installation)
COPY . .                         # copy all the remaining code
CMD ["python", "main.py"]        # the command run automatically on unpacking (startup)
```

| Command | Meaning | Analogy |
|---|---|---|
| FROM | Which ready-made image to start from | Choosing the pre-assembled base cargo |
| WORKDIR | The working folder inside the container | The workbench location inside the box |
| COPY | Copy files: my computer → inside the box | Loading goods |
| RUN | A command run during packing (the build) | Assembly work at the packing stage |
| ENV | Set environment variables (level09!) | The locker note inside the box |
| CMD | The command run when the container starts | The switch flipped the moment it is unpacked |

There is a reason `requirements.txt` is COPY'd first and the code later. Docker **caches the result of each line as a layer** and skips repacking layers that have not changed. Putting the frequently changing code last means package installation is not redone every time — builds get fast.

### 3-2. Remember just two commands

```bash
docker build -t myapp .     # build the image per the packing instructions (Dockerfile)
docker run myapp            # run the image as a container
```

Built images are shared by uploading them to a **registry** — a cargo terminal (Docker Hub and the like). The server downloads the image and just `run`s it — the server does not even need Python installed.

### 3-3. The difference from a virtual machine (a favorite interview fact)

A virtual machine (VM) builds **another whole building inside the building**, so it is heavy (a full operating system inside, gigabytes, tens of seconds to boot). A container shares the building's frame (the host OS) and **standardizes only the room**, so it is light (megabytes, sub-second startup). Being light, one server can host dozens of containers — the foundation of cloud economics.

### 3-4. Docker in AI practice

The standard AI deployment flow is "pack model + code + dependencies into an image → run as containers in the cloud." GPU training environments are also touchy about CUDA driver combinations, so starting from an official pre-combined image is the convention. Docker is likewise the prescription for AI research's chronic ailment: "the environment can't be reproduced, so the experiment results differ."

## 4. Hands-on — main.py

```bash
python3 main.py
```

This exercise works without Docker installed. It has two parts.

- **[1] Docker installation diagnosis**: checks for the docker command with `shutil.which`; if present, runs `docker --version`; if absent, prints installation guidance (Docker Desktop). Either way, we proceed.
- **[2] Dockerfile generator**: takes project specifications (Python version, needed packages, entry file) as dictionaries and **auto-generates Dockerfile text**, printing it with line-by-line commentary. Two specs — a data-analysis batch and a web API — are generated for comparison.
- **[3] Save to files**: stores the generated Dockerfiles in an `outputs/` folder so you can actually pack them with `docker build` once Docker is installed.
- **[4] Build-flow walkthrough**: shows, step by step, how the layers would stack if a real build ran.

The key code is the `generate_dockerfile()` function. A Dockerfile is, in the end, just text with rules, so it can be assembled from Python strings — demonstrating in code that this level's goal is "being able to read and write the packing instructions."

## 5. Try it yourself

1. **(Easy)** In the project specs of [2], change the Python version to 3.11 and check how the generated Dockerfile's FROM line changes. *(Hint: `python_version` in the `SPECS` dictionary.)*
2. **(Medium)** Add an `env` entry to a spec (e.g. `{"APP_MODE": "production"}`) and modify `generate_dockerfile()` to emit ENV lines. *(Hint: slot the ENV lines between COPY and CMD. And remember — secrets are never baked in via ENV!)*
3. **(Challenge)** Install Docker Desktop and, with a Dockerfile from outputs/ and one simple `main.py`, actually run `docker build -t hello .` → `docker run hello`. *(Hint: on a company computer, check the installation policy first.)*

## 6. Common mistakes

- **Confusing image with container**: the image is the packed original (static); the container is the running instance (dynamic). "Restarting an image" is not a sentence that parses.
- **Keeping data inside a container**: containers are consumables, discarded and recreated at any time. Data that must survive belongs in outside storage (volumes, databases).
- **Baking secrets into the image**: put .env files or keys in via COPY or ENV, and the secret is delivered to everyone who pulls the image. Secrets are injected at run time (an extension of the level09 principle).
- **Choosing a giant base image**: a full-size image in FROM makes multi-GB cargo. Starting from the slim family is the convention.

## Coming up next

Packing skills acquired — last comes **the ship to carry the cargo**: remote servers and the cloud. In level11 we learn what connecting to a server over ssh means, why GPUs are essential to AI, and how to develop a feel for cloud costs — and we verify the power of parallel computation in code.
