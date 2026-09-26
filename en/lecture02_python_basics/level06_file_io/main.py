"""Reading and writing files — create a transactions CSV -> aggregate -> save a report.

We practice with open / the csv module / encoding="utf-8".
[1] Create a transactions CSV, [2] read and aggregate it with DictReader,
[3] save a human-readable summary report as text.
[4] Finish with an append-mode ("a") log and a safe reproduction of mojibake.
Artifacts are saved under outputs/ next to this file.
"""

import csv
import os
import random

# Build the outputs/ path relative to this file's folder (works from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")


def create_transactions_csv(path):
    """[1] Generate a 20-row transactions CSV (playing the role of a system export)."""
    random.seed(7)                     # reproducibility: always the same data
    categories = ["Office supplies", "Meals", "Transport", "Software"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "category", "amount"])          # header
        for day in range(1, 21):                                 # September 1-20
            category = random.choice(categories)
            amount = random.randint(5, 300) * 1000               # 5k-300k KRW
            writer.writerow([f"2026-09-{day:02d}", category, amount])


def summarize_transactions(path):
    """[2] Read the CSV and return (total, count, per-category totals, largest transaction)."""
    total = 0
    count = 0
    by_category = {}                   # category -> total drawer cabinet
    biggest = ("", "", 0)              # (date, category, amount)

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):              # each row arrives as a dictionary
            amount = int(row["amount"])            # CSV values are all strings -> must convert!
            total += amount
            count += 1
            by_category[row["category"]] = by_category.get(row["category"], 0) + amount
            if amount > biggest[2]:
                biggest = (row["date"], row["category"], amount)

    return total, count, by_category, biggest


def write_report(path, total, count, by_category, biggest):
    """[3] Save the aggregation results as a human-readable report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== September transactions summary ===\n")
        f.write(f"Total spend  : {total:,} KRW ({count} transactions)\n")
        f.write(f"Average each : {total // count:,} KRW\n")
        f.write("Spend by category:\n")
        for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
            share = amount / total
            f.write(f"  - {category:15s}: {amount:>9,} KRW ({share:.1%})\n")
        f.write(f"Largest transaction: {biggest[0]} {biggest[1]} {biggest[2]:,} KRW\n")


def main():
    print("=" * 56)
    print(" File I/O — transactions CSV and a summary report")
    print("=" * 56)

    os.makedirs(OUT_DIR, exist_ok=True)            # create outputs/ if missing

    # ---------------------------------------------------------
    # [1] Create the transactions CSV ("w" write mode)
    # ---------------------------------------------------------
    print("\n[1] Creating the transactions CSV")
    csv_path = os.path.join(OUT_DIR, "transactions.csv")
    create_transactions_csv(csv_path)
    print(f"  Saved -> {csv_path}")

    with open(csv_path, encoding="utf-8") as f:    # preview only the first 3 lines
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  Preview {i}: {line.strip()}")   # strip removes the trailing \n

    # ---------------------------------------------------------
    # [2] Read the CSV and aggregate (DictReader)
    # ---------------------------------------------------------
    print("\n[2] Reading and aggregating the CSV")
    total, count, by_category, biggest = summarize_transactions(csv_path)
    print(f"  Total spend {total:,} KRW / {count} transactions")
    for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {category:15s}: {amount:>9,} KRW")
    print(f"  Largest transaction: {biggest[0]} {biggest[1]} {biggest[2]:,} KRW")

    # ---------------------------------------------------------
    # [3] Save the summary report, then read it back to verify
    # ---------------------------------------------------------
    print("\n[3] Saving the summary report")
    report_path = os.path.join(OUT_DIR, "daily_report.txt")
    write_report(report_path, total, count, by_category, biggest)
    print(f"  Saved -> {report_path}")

    with open(report_path, encoding="utf-8") as f:
        for line in f:                              # read line by line (memory-friendly)
            print(f"  | {line.rstrip()}")

    # ---------------------------------------------------------
    # [4] "a" append-mode log + reproducing mojibake
    # ---------------------------------------------------------
    print("\n[4] Append mode and encoding")

    log_path = os.path.join(OUT_DIR, "run_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:      # "a": continue after existing content
        f.write(f"Report generated: total {total:,} KRW / {count} transactions\n")
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  run_log.txt now {len(lines)} line(s) (grows with each rerun — 'a' mode)")

    # What if utf-8 text is read as latin-1? (plugging into the wrong socket)
    sample_path = os.path.join(OUT_DIR, "encoding_sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("Monthly revenue — café report ₩")
    with open(sample_path, encoding="utf-8") as f:
        ok_text = f.read()
    with open(sample_path, encoding="latin-1") as f:      # deliberately wrong encoding
        broken_text = f.read()
    print(f"  Read as utf-8   : {ok_text!r}")
    print(f"  Read as latin-1 : {broken_text!r}  <- mojibake!")
    print("  -> Spelling out encoding='utf-8' on read AND write is the survival rule for non-ASCII text.")

    print("\n[End] The files in outputs/ remain after the program ends. Open them yourself.")


if __name__ == "__main__":
    main()
