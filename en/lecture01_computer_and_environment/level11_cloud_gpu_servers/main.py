"""
Lecture 01 / Level 11 — Remote Servers, the Cloud, and GPU Environments

Experience 'parallelism' — the reason GPUs are essential to AI — with just a
few CPU cores. We take a nicely splittable computation job (counting primes
per range) and process it (a) alone (a single process) vs. (b) divided
(multiple processes), compare the times, and see how this principle leads to
the GPU with its thousands of workers, and to cloud costs.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

# Job settings (change them in "Try it yourself")
RANGE_END = 400_000   # count primes from 2 up to this number
N_CHUNKS = 8          # how many pieces to split the job into
N_WORKERS = 4         # number of workers (processes) working simultaneously


def count_primes(bounds):
    """Count primes in the range [start, end) — a simple, nicely splittable job."""
    start, end = bounds
    count = 0
    for n in range(max(start, 2), end):
        is_prime = True
        divisor = 2
        while divisor * divisor <= n:     # dividing up to the square root suffices
            if n % divisor == 0:
                is_prime = False
                break
            divisor += 1
        if is_prime:
            count += 1
    return count


def make_chunks(end, n_chunks):
    """[1] Prepare the work: split the full range into n pieces (always the same job — reproducibility)."""
    size = end // n_chunks
    return [(i * size, end if i == n_chunks - 1 else (i + 1) * size) for i in range(n_chunks)]


def solo_run(chunks):
    """[2] Working alone: one process handles the pieces one after another."""
    started = time.perf_counter()
    total = sum(count_primes(chunk) for chunk in chunks)
    return total, time.perf_counter() - started


def team_run(chunks, n_workers):
    """[3] Dividing the work: several processes handle the pieces simultaneously."""
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(count_primes, chunks))  # the key line!
    return sum(results), time.perf_counter() - started


def main():
    print("=" * 60)
    print("Experiencing parallelism — alone vs. divided, and the road to the GPU")
    print("=" * 60)

    cores = os.cpu_count()
    print(f"\n[1] Preparing the work: counting primes in 2~{RANGE_END:,}, split into {N_CHUNKS} pieces")
    print(f"    CPU cores on this machine (expert seats): {cores} / workers today: {N_WORKERS}")
    chunks = make_chunks(RANGE_END, N_CHUNKS)

    print("\n[2] Working alone — a single process handles the pieces in turn")
    solo_total, solo_time = solo_run(chunks)
    print(f"    primes found: {solo_total:,} / time taken: {solo_time:.2f}s")

    print(f"\n[3] Dividing the work — {N_WORKERS} workers (processes) handle it simultaneously")
    team_total, team_time = team_run(chunks, N_WORKERS)
    print(f"    primes found: {team_total:,} / time taken: {team_time:.2f}s")
    print(f"    results match: {'yes — splitting the work does not change the answer' if solo_total == team_total else 'different?!'}")

    # [4] The report card: the speedup, and why it falls short of theory
    speedup = solo_time / team_time if team_time > 0 else float("inf")
    print("\n[4] The report card")
    print(f"    speedup: {speedup:.2f}x (with {N_WORKERS} workers deployed)")
    print(f"    why it falls short of the theoretical {N_WORKERS}x:")
    print("      - worker-recruitment overhead: creating processes and handing out work takes time too")
    print("      - sequential portions: unsplittable work remains, like gathering results (Amdahl's law)")
    print("      - seat limits: more workers than cores leaves some with nowhere to sit")

    # [5] Extending to the GPU and the cloud
    print("\n[5] At the end of this principle stands the GPU")
    print("    - today: 4 workers (CPU cores) — AI training: thousands of workers (GPU cores)")
    print("    - AI training = an ocean of dependency-free multiplies and adds -> the ultimate splittable job")
    print("    - connect via ssh (the secure phone line), transfer via scp (the courier); servers are mostly Linux")
    print("\n    A feel for cloud estimates (a made-up example, in Korean won):")
    hourly_big, hourly_small = 30_000, 1_000
    plan_a = hourly_big * 48
    plan_b = hourly_small * 3 * 2 + hourly_big * 48
    print(f"      Plan A: big GPU (KRW {hourly_big:,}/hour) straight for 48 hours = KRW {plan_a:,}")
    print(f"              (a config mistake means re-running -> up to KRW {plan_a * 2:,})")
    print(f"      Plan B: small machine (KRW {hourly_small:,}/hour) for two 3-hour trial runs, then the real training")
    print(f"              = KRW {plan_b:,} — mistakes get caught in a KRW 6,000 experiment first")
    print("      Lesson: experiment cheap, train expensive — and turn the instance off!")

    print("\nRecap: the criterion for parallelization is 'can the work be split?'")
    print("       Splittable work + thousands of workers (a GPU) = the computing power of the AI era.")


if __name__ == "__main__":
    main()
