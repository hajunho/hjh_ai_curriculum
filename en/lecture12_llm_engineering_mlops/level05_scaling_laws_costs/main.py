"""
Hands-on scaling laws and training-cost estimation.
- Estimate GPU hours with the formula: training FLOPs ≈ 6 x parameters x tokens,
  then convert all the way to money — electricity and cloud rental.
- Verify the Chinchilla intuition (about 20 tokens per parameter), and
  draw 'which model size is optimal for a given budget' as a scaling-curve PNG.
"""
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MFU = 0.35          # share of GPU performance actually harvested (Model FLOPs Utilization)
PUE = 1.3           # data-center power efficiency factor (includes cooling and overhead)
ELEC_USD_KWH = 0.12 # industrial electricity rate (dollars/kWh, ballpark)

GPUS = {  # name: (bf16 peak TFLOPS, power draw W, cloud dollars per hour)
    "H200":     (989.0, 700, 3.50),
    "A100":     (312.0, 400, 1.80),
    "4090-class consumer": (165.0, 450, 0.45),
}

# A loss-prediction function built from approximate coefficients reported in Chinchilla-family papers
def predicted_loss(n_params, n_tokens):
    """L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 — size and data determine the loss."""
    return 1.69 + 406.4 / n_params ** 0.34 + 410.7 / n_tokens ** 0.28


def fmt(x):
    """Large numbers in easy-to-read units."""
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(x) >= div:
            return f"{x / div:,.1f}{unit}"
    return f"{x:,.0f}"


def estimate(name, n_params, n_tokens, gpu="H200", n_gpus=1):
    """Print a FLOPs -> GPU hours -> electricity/cloud cost estimate."""
    tflops, watt, usd_hr = GPUS[gpu]
    flops = 6.0 * n_params * n_tokens                      # training compute approximation
    gpu_hours = flops / (tflops * 1e12 * MFU) / 3600.0     # total GPU hours
    wall_days = gpu_hours / n_gpus / 24.0                  # wall-clock days
    kwh = watt * gpu_hours * PUE / 1000.0                  # energy consumption
    elec = kwh * ELEC_USD_KWH
    cloud = gpu_hours * usd_hr
    ratio = n_tokens / n_params
    verdict = "about right (near Chinchilla)" if 10 <= ratio <= 40 else \
              ("data-starved — the model learns less" if ratio < 10 else "data-heavy — possibly wasteful for a small model")
    print(f"  > {name}")
    print(f"     {fmt(n_params)} parameters x {fmt(n_tokens)} tokens (tokens/parameter = {ratio:,.1f} -> {verdict})")
    dur = f"{gpu_hours:,.1f} GPU-hours, about {wall_days:,.1f} days" if gpu_hours >= 0.1 \
        else f"{gpu_hours * 3600:.2f} GPU-seconds"
    print(f"     FLOPs 6ND = {flops:.2e} | on {gpu} x {n_gpus:,}: {dur}")
    print(f"     electricity {kwh:,.0f} kWh ≈ ${elec:,.0f} | cloud rental ≈ ${cloud:,.0f}")
    if n_params >= 1e7:  # the loss formula is a large-model approximation, meaningless for micro models
        print(f"     predicted final loss ≈ {predicted_loss(n_params, n_tokens):.3f}")
    else:
        print("     (the loss-prediction formula is a large-model approximation; not applied at this size)")


def draw_scaling_chart(out_path):
    """Draw loss curves per model size at equal compute budgets + the optimal frontier."""
    # default dataviz palette: blue/orange/aqua trio (fixed order), light surface
    colors = {"100M": "#2a78d6", "1B": "#eb6834", "10B": "#1baf7a"}
    sizes = {"100M": 1e8, "1B": 1e9, "10B": 1e10}
    C = np.logspace(19.5, 24, 200)                         # compute-budget axis
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for label, n in sizes.items():
        D = C / (6.0 * n)                                  # spending the whole budget fixes the token count
        L = predicted_loss(n, D)
        valid = D >= 1e8                                   # exclude the range with too few tokens
        ax.plot(C[valid], L[valid], lw=2, color=colors[label],
                label=f"{label} parameters")
        ax.annotate(f"{label}", (C[valid][-1], L[valid][-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=colors[label], fontsize=10, va="center")
    n_opt = np.sqrt(C / 120.0)                             # C=6ND, D=20N -> N=sqrt(C/120)
    ax.plot(C, predicted_loss(n_opt, 20 * n_opt), ls="--", lw=1.6,
            color="#52514e", label="compute-optimal frontier (Chinchilla)")
    ax.set_xscale("log")
    ax.set_xlabel("Compute budget (training FLOPs)", color="#0b0b0b")
    ax.set_ylabel("Predicted loss (lower is better)", color="#0b0b0b")
    ax.set_title("At the same budget, reachable loss depends on model size", color="#0b0b0b")
    ax.grid(True, alpha=0.25, lw=0.6)                      # keep the grid subtle
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def main():
    np.random.seed(0)  # seed fixed by convention (this level uses no randomness)

    # [1] the basic training-cost formula ------------------------------------
    print("[1] Basic formula: training FLOPs ≈ 6 x N (parameters) x D (tokens)")
    print("    Forward pass 2ND + backward pass 4ND. Divide by effective GPU performance (MFU) to get time.")
    print(f"    Assumptions: MFU {MFU:.0%}, electricity ${ELEC_USD_KWH}/kWh, PUE {PUE}\n")

    # [2] estimates by scale — from miniature to large ------------------------
    print("[2] Training-cost estimates by scale")
    estimate("Our TinyGPT (the model we trained in Level 04)",
             n_params=1.11e5, n_tokens=2.15e6, gpu="4090-class consumer", n_gpus=1)
    estimate("Small model (120M parameters)", 1.2e8, 2.4e9, gpu="A100", n_gpus=8)
    estimate("Mid-size model (7B parameters)", 7e9, 1.4e11, gpu="H200", n_gpus=256)
    estimate("Large model (70B parameters)", 7e10, 1.4e12, gpu="H200", n_gpus=2048)

    # [3] the Chinchilla intuition — a fixed budget fixes the optimal size ----
    print("\n[3] Chinchilla intuition: once the compute budget C is fixed, the optimum is D ≈ 20N")
    print("    Substituting D = 20N into C = 6ND gives N_opt = sqrt(C/120)")
    for C in (1e21, 1e22, 1e23, 1e24):
        n_opt = (C / 120.0) ** 0.5
        d_opt = 20.0 * n_opt
        print(f"    budget {C:.0e} FLOPs -> optimal {fmt(n_opt)} parameters, {fmt(d_opt)} tokens, "
              f"loss ≈ {predicted_loss(n_opt, d_opt):.3f}")
    print("    -> Too big for the budget: 'a large model that under-learned'. Too small: 'a small model stuck at its ceiling'.")

    # [4] draw the scaling curves ----------------------------------------------
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "scaling_curves.png"
    draw_scaling_chart(out_path)
    print(f"\n[4] Scaling curves saved: {out_path}")
    print("    How to read them: at small budgets the small model wins, but as the budget grows")
    print("    the curves cross and larger models take over. The dashed line (Chinchilla frontier) is the limit at each budget.")

    # [5] practical takeaways ---------------------------------------------------
    print("\n[5] Practical takeaways")
    print("    - Pre-training costs run millions to tens of millions of dollars -> very few companies do it themselves")
    print("    - For most businesses, fine-tuning (next level) or APIs are plenty")
    print("    - Remember just two things from the estimating meeting: 'FLOPs = 6ND' and")
    print("      'tokens ≈ 20x parameters' — enough to sanity-check any proposal's scale.")


if __name__ == "__main__":
    main()
