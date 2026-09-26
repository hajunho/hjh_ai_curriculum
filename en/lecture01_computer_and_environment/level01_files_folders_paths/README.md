# Lecture 01 · Level 01 — Files, Folders, and Paths

> All data on a computer is a file, and a file's address is its path. We build and explore a folder tree ourselves with Python's pathlib.

**Difficulty** ⭐ / **Prerequisites** level00 / **Estimated time** 35 min

## 1. Why learn this — the business view

A large share of the errors you meet while learning to code are not grand at all — they are "I can't find that file (FileNotFoundError)." To analyze data you must read a CSV file; when you produce a report it has to be saved somewhere; even AI models end up saved as files. And if you don't know how to tell the computer exactly *where* a file lives — that is, the **path** — you cannot take a single step forward.

From a business standpoint, path literacy converts directly into money. The time spent trading "where is that file again?" messages on the shared drive, the version chaos caused by saving into the wrong folder, the folder that got left out of the backup — all of it stems from a weak feel for paths and folder structure. Finish this level and you will be able to express the location of any file as one precise line of address, and to create or search folder structures automatically with Python.

## 2. Understanding through an analogy

**A folder structure is the address system of a company building.**

Picture a corporate headquarters: "HQ building > 3rd floor > Sales office > Ms. Kim's cabinet > 2024 contracts folder > AcmeCo_contract.pdf". Stepping down from the largest unit to the smallest pins down any document in the world to exactly one place. A computer path works exactly this way.

```
/Users/kim/Documents/contracts/2024/AcmeCo_contract.pdf
```

The leading `/` is "the building's front entrance (the root)", and each step separated by `/` corresponds to a floor, an office, a cabinet. Writing the **entire address starting from the front entrance is an absolute path**.

Inside the office, though, people abbreviate: "it's in the 2024 folder in our team's cabinet." That works because both parties know where they currently are. That is a **relative path** — an address measured from your current location, like `contracts/2024/AcmeCo_contract.pdf`. Relative paths have two special notations: `.` means "this very room" and `..` means "one level up (the parent folder)."

The **extension** at the end of a file name — `.pdf`, `.xlsx`, `.py` — is **the document-type stamp on the envelope**. It is a convention that announces what the contents are (a document, a table, code), and the operating system looks at this stamp to decide which program should open it.

## 3. Core concepts

### 3-1. Everything is a file

Inside a computer, documents, photos, programs, even configuration values are all files. A file consists of "a name + contents (a string of 0s and 1s) + a location," and a folder (directory) is a container for files that can also contain other folders. The whole structure therefore becomes a **tree** — branches spreading out from a root.

### 3-2. Absolute vs. relative paths

| | Absolute path | Relative path |
|---|---|---|
| Starting point | The root (`/`) or a drive (`C:\`) | The current working folder |
| Example | `/Users/kim/data/sales.csv` | `data/sales.csv` |
| Strength | Points to the same file no matter where you run from | Short, and stays valid if the whole project moves |
| Weakness | Breaks easily on another computer | Points somewhere unexpected if your "current location" differs |

That "current location" is called the **current working directory (CWD)**. Whatever folder you launched the program from is the CWD, and every relative path is interpreted from there. 90% of relative-path errors come from a mismatch between "where I think I am" and "the actual CWD."

### 3-3. Differences between operating systems

macOS/Linux use `/` as the separator and have a single root, `/`. Windows uses `\` and has a root per drive, like `C:\` and `D:\`. Because of this difference, gluing paths together as raw strings breaks on the other operating system. Python's **pathlib** library handles the difference automatically, so this curriculum uses pathlib as the standard from day one.

### 3-4. The essential five of pathlib

```python
from pathlib import Path
p = Path("data") / "sales.csv"   # joining paths: safe with the / operator
p.exists()                        # does it exist?
p.mkdir(parents=True, exist_ok=True)  # create a folder (with intermediates, no error if present)
p.resolve()                       # convert to an absolute path
p.suffix, p.stem, p.parent        # extension, bare name, parent folder
```

Know these five and you can read most of the file handling in the rest of the curriculum.

## 4. Hands-on — main.py

```bash
python3 main.py
```

The script builds and explores the document tree of a small fictional company, "Hanbit Trading," inside a temporary folder.

- **[1] Build the tree**: creates department folders like `general_affairs/`, `sales/contracts/`, `sales/reports/` and sample files, using pathlib.
- **[2] Draw the tree**: a recursive function prints the folder structure as an indented tree diagram — a Python imitation of the terminal's `tree` command.
- **[3] Absolute vs. relative**: expresses the same file both ways and shows how `..` behaves.
- **[4] Extension census**: sweeps the whole tree (`rglob`) and counts files per extension — like counting how many documents of each type sit in the company archive.
- **[5] Path anatomy**: dissects one path into parent / name / stem / suffix.

The key piece of code is the `draw_tree()` function. It opens a folder, lists the contents, and calls itself again whenever an item is a folder — a **recursion**. It is the most natural way to handle tree structures, so you will meet it again and again. When the exercise finishes, the temporary folder is deleted automatically, leaving no trace on your computer.

## 5. Try it yourself

1. **(Easy)** Inside `build_company_tree()`, add an "hr" (human resources) folder with a `job_posting.txt` file beneath it, and confirm it appears in the tree output. *(Hint: copy a line that creates an existing department folder and change the name.)*
2. **(Medium)** Add two pdf files to any folder so that the extension census in step [4] picks up `.pdf`. Verify the counts. *(Hint: `write_text` with any placeholder text is fine — the extension is only a naming convention.)*
3. **(Challenge)** Instead of counting files per extension, count "files per folder" and print the department with the most documents. *(Hint: from the `rglob("*")` results, keep only entries where `p.is_file()` and count them by `p.parent.name`.)*

## 6. Common mistakes

- **Joining paths with string addition**: `"data" + "/" + "sales.csv"` can break on Windows. Always write `Path("data") / "sales.csv"`.
- **Misjudging the CWD**: the folder containing the script and the folder you ran it from can differ. When you need a path anchored to the script, use `Path(__file__).parent`.
- **Deleting or mis-changing an extension**: the extension does not change the contents. Renaming `report.xlsx` to `report.csv` does not make it a CSV — it just stops opening properly.
- **Paths with spaces or special characters**: make it a habit to name folders and files with lowercase letters and underscores (`sales_2024.csv`). Spaces and non-ASCII characters mean constant fiddling with quotes in the terminal and in code.

## Coming up next

Now that you know the archive's address system, it is time to learn to **roam the warehouse with the keyboard alone**. In level02 we open the terminal and practice the developer's fundamentals — moving between folders and handling files with commands like `ls`, `cd`, and `pwd`.
