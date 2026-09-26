# Lecture 01 · Level 00 — What Kind of Machine Is a Computer?

> A computer is, in the end, a machine that "takes input → computes → produces output," and its CPU, memory, and storage divide the work exactly like the departments of a company.

**Difficulty** ⭐ / **Prerequisites** none / **Estimated time** 30 min

## 1. Why learn this — the business view

Throughout this curriculum we are going to give the computer an enormous amount of work: cleaning up data, drawing charts, and eventually training AI models. But if the person handing out the work knows nothing about how the worker operates, trouble follows. You end up unable to answer questions like "why does this task finish in one second while that one takes an hour?", "why did I lose my work when I closed the program?", or "what does an out-of-memory error even mean?"

It is the same in business. When you outsource work to a vendor, even a rough idea of how that company is organized and how it operates helps you understand why the quote and the timeline look the way they do. The computer is a "vendor" we hand work to every single day. Knowing its internal structure just at the big-picture level means the terms you will meet in every later level (CPU, memory, disk, process) will never feel foreign.

This level has exactly one goal: to internalize the big picture that **a computer is a machine that repeats "input → compute → output."**

## 2. Understanding through an analogy

Picture the computer as a small company.

- **The CPU (central processing unit) is the staff member who does the actual work.** It is the only one that truly calculates and decides. It is astonishingly fast, but it can only do one tiny thing at a time (an addition, a comparison). In exchange, it does those tiny things billions of times per second — a hyper-speed clerk who signs off on a document in 0.0000001 seconds.
- **Memory (RAM) is that clerk's desk.** It is where the documents currently being worked on are spread out. Papers on the desk are within arm's reach, so access is very fast — but when the clerk goes home (power off), the desk is swept completely clean. In other words, **memory is fast but volatile**. "I closed it without saving and my document vanished" is exactly this.
- **Storage (SSD/HDD) is the document archive.** It is where papers are kept permanently. Walking to the archive and retrieving a file takes time (hundreds to thousands of times slower than memory), but the contents survive a power-off. "Saving" a file means formally filing the papers from the desk into the archive.
- **Input devices (keyboard, mouse) are the customer service counter**, and **output devices (monitor, printer) are the shipping department.**

The flow of work is identical to a company's workflow. An order arrives at the front counter (input), the needed materials are fetched from the archive and spread on the desk (storage → memory), the clerk does the calculations (CPU), the result is shipped to the customer (output), and the records are filed back in the archive (memory → storage). Opening Excel, entering a formula, and saving the file follows exactly this flow.

## 3. Core concepts

### 3-1. Input → Compute → Output (the IPO model)

Every program, without exception, consists of these three stages. This is called the IPO (Input-Process-Output) model.

| Stage | Example in Excel | Example in this level's exercise |
|---|---|---|
| Input | Typing sales figures into cells | The list of order data |
| Process | SUM and AVERAGE formulas | Computing totals and averages in Python |
| Output | Results on screen, saved file | Printing to screen + saving a file |

No matter how complex an AI program looks later on, split it into three pieces — "what is the input, what is being computed, what is the output?" — and you are already halfway to understanding it.

### 3-2. Why computers speak in 0s and 1s

The CPU works with electrical signals. The only two states it can distinguish reliably are current flowing / not flowing, so it represents all information in 0s and 1s (binary). A single 0 or 1 is a bit; a group of eight is a byte. Text, photos, videos, Excel files — all of them are ultimately strings of 0s and 1s. Capacity units like "1 MB, 1 GB" simply count those bytes by the million or the billion.

### 3-3. The speed hierarchy — why some tasks are slow

The key point of the desk (memory) vs. archive (storage) analogy is the **difference in speed**. A rough feel is all you need:

- Calculations inside the CPU: billions per second
- Reading from memory: tens to hundreds of times slower than a CPU calculation
- Reading from an SSD: hundreds of times slower than memory
- Downloading from the internet: tens of times slower again than the SSD

So when a program is slow, the cause is usually not "too much math" but "too many trips to the archive (disk) or to the outside world (network)." This intuition returns later in the lecture on large datasets.

### 3-4. Programs and processes

A work manual sitting in the archive (a program saved as a file) does nothing by itself. Only when a clerk spreads the manual on the desk and actually starts working does any work happen. A **program that is currently running is called a process**. Just as several clerks can work from the same manual at once, launching one program (say, Chrome) several times creates several processes.

## 4. Hands-on — main.py

Move into this folder in the terminal and run the script. (If the terminal still feels unfamiliar, level02 covers it in detail — for now, just follow along.)

```bash
python3 main.py
```

The script walks through a small "cafe headquarters sales report" task, mapped onto the departments of our computer company.

- **[1] Input**: order data from each store arrives (the front counter).
- **[2] Storage → memory**: the data is saved to a temporary file (the archive) and read back, demonstrating the difference between "saved data persists, memory evaporates."
- **[3] Compute**: playing the role of the CPU, it calculates totals, averages, and the top-revenue store. A timing check shows that repeating the same calculation a million times still takes under a second.
- **[4] Output**: the results are printed as a human-friendly report.
- **[5] A taste of binary**: it shows exactly which 0s and 1s the text "AI" is stored as.

The part of the code worth studying is the `process_orders()` function. It takes input (the orders list), computes (totals and averages), and produces output (a returned dictionary) — the IPO model captured in a single function. Don't worry if you don't know Python syntax yet; the comments after each `#` are written so that reading them alone reveals the flow.

## 5. Try it yourself

1. **(Easy)** Find the repeat-calculation count in step [3] of the output. Predict how much longer it would take if you multiplied `LOOP_COUNT` by 10, then change it and run the script to check. *(Hint: computation and time are nearly proportional.)*
2. **(Medium)** Add a store with a neighborhood name you know to the order data in step [1]. Check how the total and the top-revenue store change. *(Hint: add one tuple of the same shape to the `orders` list.)*
3. **(Challenge)** Using step [5] as a guide, change the script so it prints how your own initials are represented in binary. *(Hint: find the string `"AI"` and replace it.)*

## 6. Common mistakes

- **Confusing "saved" with "open"**: while you edit a document, the contents live only on the desk (memory). You must press Save for them to be filed in the archive (disk). The same goes for code — after editing a file, save it before running, or your changes won't take effect.
- **Confusing capacity with speed**: "16 GB of memory" and "a 512 GB SSD" are both storage figures, but their roles differ. Memory is the size of the desk (how much you can work on at once); the SSD is the size of the archive (how much you can keep in total).
- **Assuming the computer is smart**: a computer does only what it is told, exactly as told. When you hit an error in later lectures, build the habit of first asking "where were my instructions ambiguous?" rather than "the computer is acting up."

## Coming up next

You have seen the org chart of the computer company; next it is time to learn **how its document archive is organized**. In level01 we explore how files, folders, and paths are structured — and use Python to navigate that address system called the "path" ourselves.
