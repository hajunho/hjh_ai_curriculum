"""Exception handling and debugging — before/after with deliberately broken sales data.

We take contaminated data (text amounts, blank values, hidden whitespace,
a missing column) and compare [2] unprotected processing that dies against
[3] processing that survives with try/except. We also observe the five
representative exceptions, practice print debugging (!r), and sound early
alarms with raise.
"""


def validate_amount(amount):
    """Validation function for [5]: a negative amount breaks the contract -> raise immediately."""
    if amount < 0:
        raise ValueError(f"Amount cannot be negative: {amount}")
    return amount


def main():
    print("=" * 56)
    print(" Exceptions and debugging — surviving broken data")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] Trigger the five representative exceptions and read the 'last line'
    # ---------------------------------------------------------
    print("\n[1] Observing representative exceptions (type: cause message)")

    demos = [
        ("int('TBD')", lambda: int("TBD")),
        ("'100' + 5", lambda: "100" + 5),
        ("{'a':1}['amount']", lambda: {"a": 1}["amount"]),
        ("[][0]", lambda: [][0]),
        ("10 / 0", lambda: 10 / 0),
    ]
    for code_text, run in demos:
        try:
            run()
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as e:
            # This is the information on the traceback's last line: type name + cause
            print(f"  {code_text:18s} -> {type(e).__name__}: {e}")
    print("  -> Read error messages bottom to top. The last line is 'type: cause'.")

    # Contaminated monthly sales data (a collection of common real-file accidents)
    dirty_rows = [
        {"store": "Downtown", "amount": "1200000"},
        {"store": "Riverside", "amount": "980000"},
        {"store": "Airport", "amount": "TBD"},         # text amount -> ValueError
        {"store": "University", "amount": "1450000"},
        {"store": "Lakeside", "amount": ""},           # blank value -> ValueError
        {"store": "Midtown", "amount": " 730000 "},    # hidden whitespace (int works, but see [4])
        {"store": "Harbor"},                            # the amount key itself is missing -> KeyError
        {"store": "Hillside", "amount": "2100000"},
        {"store": "Station", "amount": "880000"},
        {"store": "Parkside", "amount": "1010000"},
    ]

    # ---------------------------------------------------------
    # [2] Unprotected processing: drops dead on the 3rd row
    # ---------------------------------------------------------
    print("\n[2] What if we aggregate with no exception handling?")
    try:
        total = 0
        for i, row in enumerate(dirty_rows):
            total += int(row["amount"])           # no protection!
        print(f"  Total: {total}")                 # never reached
    except (ValueError, KeyError) as e:
        print(f"  Died on row {i} ({dirty_rows[i]['store']}) -> {type(e).__name__}: {e}")
        print(f"  -> The partial total so far, {total:,} KRW, is thrown away, and the remaining rows never run.")

    # ---------------------------------------------------------
    # [3] Protected processing: log the bad rows, skip, and finish
    # ---------------------------------------------------------
    print("\n[3] Aggregation protected by try/except")

    total = 0
    ok_count = 0
    failures = []                                  # (store, reason) records
    for row in dirty_rows:
        try:
            amount = int(row["amount"].strip())    # only the smallest risky stretch inside try
        except KeyError:
            failures.append((row["store"], "no amount column"))
            continue
        except ValueError as e:
            failures.append((row["store"], f"bad amount format ({row['amount']!r})"))
            continue
        total += amount
        ok_count += 1

    print(f"  Succeeded {ok_count} / failed {len(failures)} / total {total:,} KRW")
    print("  Failure details (never silently pass — always record):")
    for store, reason in failures:
        print(f"    - {store}: {reason}")
    print("  -> Same data, but [2] dies while [3] even produces an operations report.")

    # ---------------------------------------------------------
    # [4] print debugging: when there is no error but the value is off
    # ---------------------------------------------------------
    print("\n[4] print debugging — catching hidden whitespace")

    raw = " 730000 "
    print(f"  print(raw)      -> {raw}    (looks fine at a glance)")
    print(f"  print(f'{{raw!r}}') -> {raw!r}  <- printed with !r, the spaces show!")
    print(f"  int(raw.strip()) = {int(raw.strip()):,}  (strip then convert is the safe way)")
    print("  Technique: print with the variable name, use !r, only in the suspect stretch, delete after fixing.")

    # ---------------------------------------------------------
    # [5] raise: reject bad values early and loudly
    # ---------------------------------------------------------
    print("\n[5] raise — early alarm on negative amounts")
    try:
        validate_amount(150000)
        print("  validate_amount(150000)  -> passed")
        validate_amount(-50000)
        print("  this line never runs")
    except ValueError as e:
        print(f"  validate_amount(-50000) -> ValueError: {e}")
        print("  -> Rejecting an abnormal value early beats letting it pass silently.")

    # ---------------------------------------------------------
    # finally demo: cleanup regardless of accidents
    # ---------------------------------------------------------
    print("\n[6] finally — the cleanup that happens no matter what")
    try:
        risky = int("bad")
    except ValueError:
        print("  except: bad value handled")
    finally:
        print("  finally: (even after an error) lock-up chores like closing files/connections still run")

    print("\n[End] Real-world data is always broken.")
    print("      Exception handling means: log the one bad row and skip it, but never stop the whole job.")


if __name__ == "__main__":
    main()
