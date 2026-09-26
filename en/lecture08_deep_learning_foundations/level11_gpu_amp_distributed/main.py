"""
A concept exercise that grounds the 'why' of GPUs, mixed precision, and distributed training in numbers.
(1) Time CPU matrix multiplications by size -> confirm computation grows as a cube,
(2) an fp32/fp16/bf16 memory calculator -> the precision-for-memory trade,
(3) a training-memory estimate for an 8B-parameter model -> numbers for the GPU budget meeting.
Every calculation finishes in seconds on a CPU.
"""

import time

import numpy as np
import torch

GB = 1024 ** 3


def bench_matmul(sizes, repeats=3):
    """Time (n,n)@(n,n) matrix multiplications. The cost is about 2*n^3 FLOPs."""
    results = []
    for n in sizes:
        rng = np.random.default_rng(0)                    # reproducibility
        A = rng.standard_normal((n, n), dtype=np.float32)
        B = rng.standard_normal((n, n), dtype=np.float32)
        A @ B                                             # warm-up (cache/thread preparation)
        t0 = time.perf_counter()
        for _ in range(repeats):
            A @ B
        dt = (time.perf_counter() - t0) / repeats
        gflops = 2 * n ** 3 / dt / 1e9
        results.append((n, dt, gflops))
    return results


def bytes_per_element(dtype_name):
    return {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[dtype_name]


def model_memory_gb(n_params, dtype_name):
    """Parameter storage memory (GB) = parameter count x bytes per element."""
    return n_params * bytes_per_element(dtype_name) / GB


def training_memory_gb(n_params, mixed=False, act_gb=20.0):
    """Training-memory estimate (GB). The conventional approximation:
    - pure fp32            : weights 4 + gradients 4 + Adam state 8 (m,v in fp32)   = 16 bytes/param
    - mixed precision (AMP): weights 2 (fp16) + gradients 2 + fp32 master 4 + Adam 8 = 16 bytes/param
      (compute gets faster, but the point is that Adam training 'storage' memory barely shrinks)
    We then add a rough figure for activations (intermediate results)."""
    per_param = 16
    detail = ("weights fp16 2 + grad 2 + fp32 master 4 + Adam 8" if mixed
              else "weights 4 + grad 4 + Adam(m,v) 8")
    states = n_params * per_param / GB
    return states, states + act_gb, detail


def main():
    torch.manual_seed(0)

    print("[1] Why GPUs — over 90% of deep-learning computation is matrix multiplication")
    print("    CPU = a few PhDs (great at complex work), GPU = thousands of arithmetic clerks (simple multiplications, simultaneously).")
    print("    Matrix multiplication is 'simple multiplications processed in bulk at once' — the GPU's winning event.\n")

    print("[2] CPU matmul benchmark — (n,n)@(n,n), cost = 2n^3")
    sizes = [128, 256, 512, 1024]
    rows = bench_matmul(sizes)
    print(f"    {'n':>6s} {'time(ms)':>10s} {'work(GFLOP)':>14s} {'speed(GFLOP/s)':>15s}")
    for n, dt, gflops in rows:
        print(f"    {n:6d} {dt*1e3:10.2f} {2*n**3/1e9:14.3f} {gflops:15.1f}")
    n0, t0_, _ = rows[0]
    n3, t3_, _ = rows[-1]
    print(f"    n grows {n3//n0}x -> the work grows {(n3/n0)**3:.0f}x. A modest size increase explodes the math.")
    print("    A modern GPU does hundreds of TFLOP/s in this event — hundreds to thousands of times the CPU.\n")

    print("[3] Memory calculator by precision (dtype) — the 'fewer-digit calculator' trade")
    print(f"    {'parameters':>14s} {'fp32':>10s} {'fp16/bf16':>10s} {'int8':>10s}")
    for label, n_params in [("100M (0.1B)", int(1e8)), ("1B", int(1e9)),
                            ("8B", int(8e9)), ("70B", int(7e10))]:
        f32 = model_memory_gb(n_params, "fp32")
        f16 = model_memory_gb(n_params, "fp16")
        i8 = model_memory_gb(n_params, "int8")
        print(f"    {label:>13s} {f32:9.1f}G {f16:9.1f}G {i8:9.1f}G")
    print("    fp16: give up half the decimal precision, gain 2x in memory/speed.")
    print("    bf16: the same 2 bytes as fp16, but the exponent matches fp32 — robust to large values -> the LLM training standard.")
    print("    Mixed precision (AMP): multiplications in fp16/bf16, accumulations and the master weights in fp32 -> speed plus stability.\n")

    print("[4] 8B model 'training' memory estimate — a different dimension from inference")
    n_params = int(8e9)
    act = 20.0                                            # rough activations figure (varies with batch and context length)
    for mixed in (False, True):
        act_now = act / 2 if mixed else act               # AMP: activations at half precision, so about 1/2
        states, total, detail = training_memory_gb(n_params, mixed, act_now)
        mode = "mixed precision (AMP)" if mixed else "pure fp32"
        print(f"    [{mode}] 16 bytes per parameter ({detail})")
        print(f"      state storage {states:.0f} GB + activations ~{act_now:.0f} GB = about {total:.0f} GB")
    print("    The point: even with AMP, 'storage' memory stays at 16 bytes/parameter because of Adam state.")
    print("    The gains come from (1) ~2x compute speed (2) halved activations (3) halved communication.")
    total_gb = training_memory_gb(n_params, True, act / 2)[1]
    print(f"    => Does it fit on one 80GB GPU? {'yes' if total_gb <= 80 else 'no'} "
          f"(needs {total_gb:.0f} GB) — it has to be carried in shares.\n")

    print("[5] Distributed training = carrying a big load together")
    print("    DDP  (data parallel): copy the same model to every GPU, split only the data.")
    print("          -> The way to gain speed when the load (model) fits on 1 card. Only gradients get averaged.")
    print("    FSDP (ZeRO, model sharding): the weights, gradients, and Adam state themselves are sliced and shared out.")
    print("          -> The way when the load doesn't fit on 1 card. Shards gather only when needed, then scatter.")
    n_gpus_options = [1, 2, 4, 8, 16]
    states = n_params * 16 / GB
    print(f"\n    Per-GPU burden for the 8B model under FSDP (state {states:.0f} GB divided + activations {act:.0f} GB each)")
    print(f"    {'GPUs':>8s} {'state/card(GB)':>15s} {'total/card(GB)':>15s} {'80GB card OK?':>14s}")
    for g in n_gpus_options:
        share = states / g + act
        print(f"    {g:8d} {states/g:15.1f} {share:15.1f} {'OK' if share <= 80 else 'X':>10s}")

    print("\n[6] What the real code looks like (concept-checking comments — not executed here on CPU)")
    print("    # AMP : with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x)")
    print("    #       with fp16, pair it with GradScaler's loss scaling (inflate the loss, deflate it back)")
    print("    # DDP : model = torch.nn.parallel.DistributedDataParallel(model)")
    print("    # FSDP: model = torch.distributed.fsdp.FullyShardedDataParallel(model)")
    print("\n[7] Recap: the answer to a GPU budget question is the multiplication 'parameter count x bytes + activations'.")
    print("    With this one estimate you can explain in a meeting why training an 8B model needs multiple GPUs.")


if __name__ == "__main__":
    main()
