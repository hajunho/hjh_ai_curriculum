"""
Lecture 01 / Level 00 — What Kind of Machine Is a Computer?

Experience how a computer works through the 'input -> compute -> output' flow.
We map the division of labor between the CPU (the clerk), memory (the desk),
and storage (the document archive) onto a cafe headquarters' sales report task.
Uses only the standard library; no internet connection required.
"""

import json
import tempfile
import time
from pathlib import Path

# Number of repeated calculations (change this value in exercise 1 of "Try it yourself")
LOOP_COUNT = 1_000_000


def receive_orders():
    """[Input] The front counter: order data arrives from each store."""
    # (store name, order count, total revenue [KRW]) — synthetic data created right here in the code.
    orders = [
        ("Downtown", 182, 1_512_000),
        ("Riverside", 141, 1_098_000),
        ("Airport", 210, 1_745_000),
        ("Lakeside", 95, 702_000),
    ]
    return orders


def save_to_storage(orders, path):
    """[Storage] Filing in the archive: data saved to a file survives a power-off."""
    rows = [{"store": s, "count": c, "revenue": r} for s, c, r in orders]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.stat().st_size  # size of the filed document in bytes


def load_from_storage(path):
    """[Memory] From the archive back onto the desk: read the file into memory."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(row["store"], row["count"], row["revenue"]) for row in rows]


def process_orders(orders):
    """[Compute] The CPU clerk: takes input, calculates, returns the result. (The IPO model)"""
    total_revenue = sum(revenue for _, _, revenue in orders)
    total_count = sum(count for _, count, _ in orders)
    avg_per_order = total_revenue / total_count
    best_store = max(orders, key=lambda row: row[2])
    return {
        "total_revenue": total_revenue,
        "total_count": total_count,
        "avg_per_order": avg_per_order,
        "best_store": best_store[0],
        "best_revenue": best_store[2],
    }


def measure_cpu_speed():
    """Time how fast the CPU can repeat a trivial calculation."""
    started = time.perf_counter()
    acc = 0
    for i in range(LOOP_COUNT):
        acc += i  # repeat a tiny addition over and over
    elapsed = time.perf_counter() - started
    return elapsed, acc


def show_binary(text):
    """Show how characters are represented as 0s and 1s inside the computer."""
    for ch in text:
        code = ord(ch)  # every character has an agreed-upon number (Unicode)
        print(f"    character '{ch}' -> number {code} -> binary {code:08b}")


def main():
    print("=" * 60)
    print("Computer Company Inc. workflow tour — input -> compute -> output")
    print("=" * 60)

    # [1] Input: order data arrives at the front counter.
    orders = receive_orders()
    print(f"\n[1] Input: orders from {len(orders)} stores arrived at the front counter")
    for store, count, revenue in orders:
        print(f"    - {store}: {count} orders, KRW {revenue:,}")

    # [2] Storage <-> memory: file it in the archive, then bring it back to the desk.
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "orders_archive.json"
        size = save_to_storage(orders, archive)
        print(f"\n[2] Storage (the archive): filed as '{archive.name}' ({size} bytes)")
        print("    - contents saved to a file survive a power-off (non-volatile)")
        orders_on_desk = load_from_storage(archive)
        print(f"    - fetched from the archive back onto the desk (memory): {len(orders_on_desk)} rows restored")
        print("    - data in memory disappears when the program ends (volatile)")

    # [3] Compute: the CPU clerk produces totals and averages.
    report = process_orders(orders_on_desk)
    elapsed, _ = measure_cpu_speed()
    print(f"\n[3] Process: a feel for the CPU clerk's sign-off speed")
    print(f"    - time for {LOOP_COUNT:,} repeated simple additions: {elapsed:.3f}s")
    print(f"    - roughly {LOOP_COUNT / elapsed:,.0f} per second — no human comes close")

    # [4] Output: ship a human-friendly report.
    print(f"\n[4] Output: sales summary report")
    print(f"    - total revenue        : KRW {report['total_revenue']:,}")
    print(f"    - total order count    : {report['total_count']:,}")
    print(f"    - avg revenue per order: KRW {report['avg_per_order']:,.0f}")
    print(f"    - top-revenue store    : {report['best_store']} (KRW {report['best_revenue']:,})")

    # [5] A taste of binary: all information is ultimately 0s and 1s.
    print(f"\n[5] A taste of binary: how the text 'AI' is actually stored")
    show_binary("AI")

    print("\nRecap: split any program into three pieces — 'input -> compute -> output'.")
    print("       CPU=the clerk, memory=the desk (fast, volatile), storage=the archive (slow, permanent).")


if __name__ == "__main__":
    main()
