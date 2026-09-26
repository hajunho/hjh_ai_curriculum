# Lecture 01 · Level 11 — Remote Servers, the Cloud, and GPU Environments

> Learn what "borrowing someone else's computer over ssh" means, why GPUs are essential to AI (parallelism), and how to develop a feel for cloud costs — then verify the effect of parallel computation in code.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level02, level10 / **Estimated time** 45 min

## 1. Why learn this — the business view

Training an AI model on your laptop takes days; on a cloud GPU server it finishes in hours. That is why almost all real AI work happens on **remote servers** — and why developers say things like "I'll ssh into the server," "spin up a GPU instance," "we ran it on eight A100s for two days."

For non-engineers, this level matters because of **cost**. Cloud GPUs bill by the hour, so "how many days will it run?" *is* the budget. A high-end GPU server costs tens of thousands of Korean won (a few dozen dollars) per hour, so one badly planned experiment can burn millions of won — while a manager who understands the concepts protects the budget with calls like "let's prototype briefly on a smaller GPU before committing." Finish this level and you can read the "GPU cost" line in an AI project quote, and you will have felt the essence of the GPU — parallelism — in code.

## 2. Understanding through an analogy

### A remote server = a well-equipped co-working office

When your own office (laptop) lacks the equipment, the company **rents a desk by the hour at a well-equipped co-working office**. That is the cloud. Companies like Amazon (AWS), Google (GCP), and Microsoft (Azure) fill enormous buildings (data centers) with hundreds of thousands of computers and rent them by the hour. One rented computer is called an **instance**.

**ssh (Secure Shell) is the secure dedicated phone line into that co-working office.** One line — `ssh account@server-address` — puts the remote computer's terminal on your screen, and the commands from level02 work **unchanged** (remote servers mostly run Linux — here is where learning the terminal pays off). To send files you use the courier service, `scp` (secure copy): `scp file account@server:/path`. The standard is to enter with a **key file (an ssh key)** instead of a password — and per the level09 principle, that key never goes into code or a repository either.

### CPU vs. GPU = four PhDs vs. four thousand part-timers

- A **CPU** is **a handful of PhD-level experts** (4–16 cores). They can solve any complex problem and each is individually fast, but there are few of them.
- A **GPU (graphics processing unit)** is **thousands of part-timers doing simple arithmetic**. Each one only does basic sums, but thousands of them calculate **simultaneously**.

Who wins at stuffing a million flyers into envelopes? The part-timer army, overwhelmingly. And AI training is exactly that kind of job — **endless repetition of simple arithmetic** (matrix multiplication), multiplying and adding millions of numbers. The GPU was originally built to paint millions of game-screen pixels at once, and AI training, being the same shape of computation, fit it perfectly. The key word is **parallelism** — splitting the work so many workers do it at the same time.

## 3. Core concepts

### 3-1. The basic remote workflow

```bash
ssh myname@203.0.113.10       # 1) connect to the server over the secure phone line
git clone <repo-address>       # 2) code arrives via GitHub (level07!)
scp data.csv myname@server:~/  # 3) data files transfer directly via scp
python3 train.py               # 4) run in the server's terminal (level02 commands, unchanged)
```

One caution: if the ssh connection drops, the running program dies with it. That is why practitioners pair it with tools that keep work running through disconnects (tmux, nohup, etc.) — knowing "why such tools become necessary" is enough for now.

### 3-2. Why AI means GPUs — in numbers

Training an AI model is, at heart, repeated matrix multiplication (vast numbers of multiplies and adds). These calculations have **no dependencies on each other**, so the work can be split into ten thousand pieces and given to ten thousand workers at once. The parallelism gap — 8 CPU cores vs. thousands of GPU cores — is what turns "days" into "hours" in practice. Conversely, work that cannot be split (each step needing the previous step's result) is actually *slower* on a GPU. **"Can the work be split?" is the criterion for using a GPU.**

### 3-3. A feel for cloud costs

Remember just the three rules of cloud billing:

1. **Billed by the hour**: an instance keeps billing as long as it is on. **Forgetting to turn it off** is the number-one beginner spending accident.
2. **GPU tiers differ enormously**: a small practice GPU (hundreds to a few thousand KRW per hour) and a top-tier training GPU (tens of thousands of KRW per hour) differ by dozens of times. Experiment on small hardware; reserve the big hardware for the confirmed training run.
3. **Ancillary costs**: storage and data transfer (especially traffic leaving the data center) are billed too.

A quick estimating drill: a GPU instance at KRW 30,000/hour training for 48 hours costs KRW 1,440,000. Run two preliminary experiments on a KRW 1,000/hour machine (3 hours each) to catch configuration mistakes early, and KRW 6,000 spent prevents a KRW 1,440,000 re-run.

### 3-4. The limits of parallelization — a feel for Amdahl's law

Some fraction of any job is inescapably sequential (reading data, gathering results). If 20% of the whole is sequential work, then no matter how many workers you add, you can never go more than 5× faster. This is Amdahl's law. "Double the workers, double the speed" does not always hold — exactly like the relationship between headcount and deadlines.

## 4. Hands-on — main.py

```bash
python3 main.py
```

Without any GPU, using just a few cores of your CPU, you measure **the effect of parallelism** directly.

- **[1] Prepare the work**: creates a nicely splittable computation job — "counting primes." Seed-fixed, so it is the same job every time.
- **[2] Working alone**: times one worker (a single process) handling the entire range.
- **[3] Dividing the work**: splits the range into pieces, hands them to several workers (multiple processes) simultaneously, and times it.
- **[4] The report card**: compares the two times and the speedup, and explains why it does not scale exactly with the core count (worker-recruitment overhead, sequential portions).
- **[5] Extending to the GPU**: draws the connection — "make the workers number in the thousands and this principle becomes the GPU" — and prints a sample cloud estimate.

The key code is the use of `concurrent.futures.ProcessPoolExecutor`. One line — `executor.map(work_fn, pieces)` — accomplishes "divide the work and assign it simultaneously." The crucial insight that carries into GPUs and AI: **parallelization was possible because the work could be split.**

## 5. Try it yourself

1. **(Easy)** Change `N_WORKERS` to 2, 4, and 8 and record how the speedup changes. What happens once you exceed your CPU's core count (see the `os.cpu_count()` output)? *(Hint: more workers than seats does not get faster.)*
2. **(Medium)** Halve the job size (`RANGE_END`). Confirm the parallelization payoff shrinks, and think about why. *(Hint: with a small job, the worker-recruitment overhead looms relatively larger.)*
3. **(Challenge)** Estimate a project that trains for 60 hours on a KRW 40,000/hour GPU instance, then build a plan-A/plan-B cost comparison that includes "two preliminary experiments on cheap hardware at 10% the performance." *(Hint: use the estimate format in output [5] as a model.)*

## 6. Common mistakes

- **Forgetting to stop the instance**: the #1 cloud spending accident. Make "work done = instance stopped" a single habit, and set up budget alerts.
- **Starting on the top-tier GPU**: configuration mistakes and code bugs surface just as well on small hardware. Experiment cheap; only the real training run goes expensive.
- **Committing ssh keys to a repository**: a leaked key file (`id_rsa` etc.) opens the whole server. The level09 principle applies to key files too.
- **Trying to parallelize everything**: sequential work that cannot be split gets no faster from parallelization — only more complex. Ask "can it be split?" first.

## Coming up next

Congratulations — you have completed lecture01! From how a computer works, through the terminal, the Python environment, Git collaboration, secret management, Docker, and the cloud, you now hold the full map of the development environment. In the next lecture (lecture02) we finally begin Python programming itself — variables, conditionals, loops, step by step.
