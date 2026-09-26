"""
Lecture 01 / Level 05 — Code Editors and Jupyter

A simulator that lets you experience the notebook's 'cell execution' mechanics
without installing Jupyter. It imitates the kernel's memory with a single
dictionary, showing how cells share that memory — and reproduces the
NameError and skewed results (out-of-order accidents) that happen when the
execution order goes astray.
"""

import io
from contextlib import redirect_stdout


class NotebookKernel:
    """A scale model of a Jupyter kernel: holds the memory (variables) and runs cell code."""

    def __init__(self):
        self.memory = {}        # the kernel's memory — every cell shares this one!
        self.run_counter = 0    # the [n] execution ordinal shown left of a cell

    def restart(self):
        """Kernel restart: the memory is wiped completely."""
        self.memory = {}
        self.run_counter = 0

    def run_cell(self, code: str):
        """Run one cell and return (ordinal, output, error)."""
        self.run_counter += 1
        buffer = io.StringIO()
        error = None
        try:
            with redirect_stdout(buffer):        # capture the print output
                exec(code, self.memory)          # execute on top of the shared memory
        except Exception as exc:                 # errors are merely displayed, notebook-style
            error = f"{type(exc).__name__}: {exc}"
        return self.run_counter, buffer.getvalue().rstrip(), error

    def visible_vars(self):
        """User variables in the kernel memory (internal entries excluded)."""
        return sorted(k for k in self.memory if not k.startswith("__"))


# The notebook document: (cell title, cell code) — in top-to-bottom screen order
CELLS = [
    ("Cell 1 — prepare data", "revenues = [1512, 1098, 1745, 702]\nprint('Store revenue (KRW 1,000s):', revenues)"),
    ("Cell 2 — total", "total = sum(revenues)\nprint('Total revenue:', total, 'thousand KRW')"),
    ("Cell 3 — average", "average = total / len(revenues)\nprint('Average revenue:', round(average, 1), 'thousand KRW')"),
    ("Cell 4 — report", "print(f'Report: total {total}k KRW, store average {average:.1f}k KRW')"),
]


def show_run(kernel, index, note=""):
    """Run one cell and print it the way a notebook screen would."""
    title, code = CELLS[index]
    n, out, err = kernel.run_cell(code)
    print(f"\n  [{n}] {title}{note}")
    for line in code.splitlines():
        print(f"      | {line}")
    if err:
        print(f"      !! error: {err}")
    elif out:
        print(f"      => {out}")
    print(f"      (kernel memory: {kernel.visible_vars() or 'empty'})")


def main():
    print("=" * 60)
    print("Mini notebook simulator — how cells and the kernel work")
    print("=" * 60)

    kernel = NotebookKernel()

    # [1] Introduce the notebook layout
    print(f"\n[1] Notebook layout: {len(CELLS)} cells (a sales-analysis scenario)")
    print("    every cell shares one Python memory called the 'kernel'")

    # [2] Run All: top to bottom — the memory accumulates cell by cell
    print("\n[2] Run All — executing top to bottom")
    for i in range(len(CELLS)):
        show_run(kernel, i)

    # [3] Reproduce the out-of-order accident: restart, then start from a middle cell
    print("\n[3] Out-of-order accident — what if we restart the kernel and run from cell 3?")
    kernel.restart()
    print("    (kernel restarted: the memory was wiped completely)")
    show_run(kernel, 2, "  <- the cells above were skipped!")
    print("      note: total is missing from memory, hence NameError — the #1 notebook accident")

    print("\n    Now let's run in the order 1 -> 3 -> 2 -> 4, against the screen order:")
    kernel.restart()
    for i, note in [(0, ""), (2, "  <- before the total (cell 2)!"), (1, ""), (3, "")]:
        show_run(kernel, i, note)
    print("      note: errors appear, or gaps get filled in belatedly.")
    print("            once the on-screen code order and the kernel memory drift apart,")
    print("            a colleague reading the document top to bottom cannot reproduce it.")

    # [4] Lessons recap
    print("\n[4] Lessons recap")
    print("    - the cell number [n] is the 'order of execution', not the screen position")
    print("    - before sharing, always verify reproducibility with Restart & Run All")
    print("    - experiments in notebooks; settled logic moves to .py scripts — that's the standard")


if __name__ == "__main__":
    main()
