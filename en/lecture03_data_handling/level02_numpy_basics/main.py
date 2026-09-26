"""
NumPy array basics — measure Python loop vs vectorization speed yourself,
then turn the sales data into an ndarray to practice shape/dtype, aggregation,
broadcasting, and boolean masks.
The key message: make 'a bundle of numbers', not 'one number', the unit of
computation, and your code gets shorter while running tens of times faster.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

N_SPEED = 1_000_000     # how many numbers to use in the speed experiment


def python_sum_of_squares(values: list[float]) -> float:
    """The Python-loop way — pick up each value, square it, accumulate."""
    total = 0.0
    for v in values:
        total += v * v
    return total


def main() -> None:
    rng = np.random.default_rng(42)          # fixed seed -> same random numbers every time

    print(f"[1] Speed duel — sum of squares of {N_SPEED:,} numbers")
    data_arr = rng.random(N_SPEED)           # one million floats between 0 and 1
    data_list = data_arr.tolist()            # the same values as a Python list

    t0 = time.perf_counter()
    loop_result = python_sum_of_squares(data_list)
    loop_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_result = float((data_arr * data_arr).sum())   # vectorized: no loop
    vec_sec = time.perf_counter() - t0

    print(f"    Python loop      : {loop_sec * 1000:8.1f} ms")
    print(f"    NumPy vectorized : {vec_sec * 1000:8.1f} ms")
    print(f"    -> About {loop_sec / vec_sec:.0f}x faster. "
          f"(difference between the answers {abs(loop_result - vec_result):.6f} "
          f"-> same computation)")
    print()

    print("[2] Sales data as an array — checking shape and dtype")
    rows = hjh_data.sales_table(n_days=90, seed=42)
    # Extract revenue only, excluding missing values (None), into an array
    rev = np.array([r["revenue"] for r in rows if r["revenue"] is not None])
    weekend_mask_src = [r["weekday"] in ("Sat", "Sun")
                        for r in rows if r["revenue"] is not None]
    is_weekend = np.array(weekend_mask_src)
    n_missing = len(rows) - len(rev)
    print(f"    {len(rows)} source rows minus {n_missing} missing -> array of {len(rev)}")
    print(f"    rev.shape = {rev.shape}, rev.dtype = {rev.dtype}")
    print()

    print("[3] One-line aggregation — statistics without a loop")
    print(f"    Sum : KRW {rev.sum():>16,}")
    print(f"    Mean: KRW {rev.mean():>16,.0f}")
    print(f"    Std : KRW {rev.std():>16,.0f}")
    print(f"    Max : KRW {rev.max():>16,} / Min: KRW {rev.min():,} "
          f"(negative contamination included)")
    print()

    print("[4] Broadcasting — 10% VAT applied with 'one stamp'")
    with_vat = rev * 1.1                       # one number expands over the whole array
    rounded = np.round(with_vat, -3)           # rounding to thousands, also vectorized
    print(f"    Pre-tax mean : KRW {rev.mean():>14,.0f}")
    print(f"    Post-tax mean: KRW {with_vat.mean():>14,.0f} (= pre-tax x 1.1)")
    print(f"    Round-to-thousands example: {rev[:3]} -> {rounded[:3].astype(int)}")
    print()

    print("[5] Boolean masks — selecting by condition")
    weekend_avg = rev[is_weekend].mean()       # keep only values where the mask is True
    weekday_avg = rev[~is_weekend].mean()      # ~ flips True/False
    print(f"    Weekend mean: KRW {weekend_avg:>14,.0f} ({int(is_weekend.sum())} rows)")
    print(f"    Weekday mean: KRW {weekday_avg:>14,.0f} ({int((~is_weekend).sum())} rows)")
    print(f"    -> Weekends run at {weekend_avg / weekday_avg:.2f}x the weekday level")
    negative = rev < 0
    print(f"    Negative contamination: {int(negative.sum())} rows "
          f"(counting Trues = summing the mask)")
    print(f"    Mean excluding negatives: KRW {rev[~negative].mean():,.0f}")
    print()
    print("    In the next level, we dress this array in 'column names': the DataFrame.")


if __name__ == "__main__":
    main()
