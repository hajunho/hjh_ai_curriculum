"""
Experiments with learning-rate schedules and training-stabilization techniques.
On a waveform regression problem (y=sin 2x) we compare four conditions with SGD+momentum:
(A) a small fixed learning rate, (B) a large fixed learning rate (reproducing a divergence crash),
(C) a large learning rate + gradient clipping, (D) a warmup+cosine schedule + clipping.
Saves the loss curves and the learning-rate schedule curves as a PNG.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
EPOCHS, BATCH = 60, 32


def make_data():
    """Synthetic regression data: y = sin(2x) + noise. The task is tracing the curve."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, (512, 1)).astype(np.float32)
    y = (np.sin(2 * X) + 0.1 * rng.normal(size=(512, 1))).astype(np.float32)
    X_va = np.linspace(-3, 3, 200, dtype=np.float32).reshape(-1, 1)
    y_va = np.sin(2 * X_va)
    to = torch.from_numpy
    return to(X), to(y), to(X_va), to(y_va)


def build_model():
    return nn.Sequential(nn.Linear(1, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, 1))


def lr_at(step, total, base_lr, schedule):
    """Per-step learning rate. warmcos = ramp up gently for the first 10% (warmup),
    then descend smoothly along a cosine curve (decay)."""
    if schedule == "fixed":
        return base_lr
    warm = int(0.1 * total)
    if step < warm:
        return base_lr * step / warm                      # warming up right after arriving at work
    progress = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))  # fine-tuning before the deadline


def train(name, base_lr, schedule="fixed", clip=None):
    """Mini-batch SGD+momentum training. Returns per-epoch valid MSE and per-step learning-rate history."""
    torch.manual_seed(2)                                  # same initial weights for every condition
    X, y, X_va, y_va = make_data()
    model = build_model()
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    loss_fn = nn.MSELoss()
    n, spe = len(X), len(X) // BATCH                      # spe = steps per epoch
    total = EPOCHS * spe
    g = torch.Generator().manual_seed(0)

    va_hist, lr_hist, step = [], [], 0
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(spe):
            b = perm[i * BATCH:(i + 1) * BATCH]
            lr = lr_at(step, total, base_lr, schedule)
            for group in opt.param_groups:
                group["lr"] = lr
            lr_hist.append(lr)
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            if clip is not None:                          # limit the gradients' overall size (norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            step += 1
        with torch.no_grad():
            va_hist.append(loss_fn(model(X_va), y_va).item())

    final = va_hist[-1]
    status = "diverged (NaN)" if math.isnan(final) else f"{final:.4f}"
    print(f"    {name:40s}: final valid MSE = {status}")
    return va_hist, lr_hist


def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] Problem: y = sin(2x) curve regression, 1-64-64-1 MLP, SGD + momentum 0.9")
    print(f"    {EPOCHS} epochs x batch {BATCH} (16 steps per epoch). 4 conditions, same seed.\n")

    print("[2] Training per condition")
    runs = {}
    runs["A"] = train("A. fixed lr=0.05 (safe driving)", 0.05)
    runs["B"] = train("B. fixed lr=0.2  (speeding, unprotected)", 0.2)
    runs["C"] = train("C. lr=0.2 + clipping 0.5", 0.2, clip=0.5)
    runs["D"] = train("D. warmup+cosine lr=0.2 + clipping", 0.2, schedule="warmcos", clip=0.5)

    a, b, c, d = (runs[k][0][-1] for k in "ABCD")
    assert math.isnan(b), "B is the divergence demo, but it did not diverge"
    assert not any(math.isnan(v) for v in (a, c, d)), "A/C/D diverged"
    assert d < a, "the schedule combo (D) should beat the baseline (A)"

    print("\n[3] Interpretation")
    print("    A: press gently and it converges safely (around 0.007).")
    print("    B: 4x the lr -> the loss explodes within a few steps, NaN. A 'speeding crash' at the scene.")
    print("    C: at the same speed, clipping (the steering-angle limiter) keeps it alive.")
    print("    D: add warmup+cosine on top -> the lowest MSE of all conditions.")
    print("       Slow at first (warmup), bold in the middle, fine at the end — the power of a schedule.\n")

    print("[4] Saving the figure (loss curves + learning-rate schedule)")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    colors = {"A": "#1f77b4", "B": "#d62728", "C": "#ff7f0e", "D": "#2ca02c"}
    labels = {"A": "A fixed 0.05", "B": "B fixed 0.2 (diverged)",
              "C": "C 0.2 + clip", "D": "D warmup+cosine + clip"}
    for key, (va, _) in runs.items():
        vals = [min(v, 10.0) if not math.isnan(v) else 10.0 for v in va]  # NaN capped at the top for display
        axes[0].plot(range(1, EPOCHS + 1), vals, label=labels[key], color=colors[key])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("valid MSE (log, capped at 10)")
    axes[0].set_title("Stability: same model, four training recipes")
    axes[0].legend(fontsize=8)
    for key in ("A", "D"):
        axes[1].plot(runs[key][1], label=labels[key], color=colors[key])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("LR schedule: fixed vs warmup+cosine")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "lr_schedule_compare.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"    saved: {png_path}\n")

    print("[5] Concept memo — batch normalization (BatchNorm)")
    print("    Passing through layer after layer, the values' distribution (the gauge) drifts and training can destabilize.")
    print("    Batch normalization recalibrates each layer's input to mean 0 / variance 1 per batch,")
    print("    a 'gauge recalibrator between layers' that lets training withstand large learning rates.")
    print("    In code, just insert nn.BatchNorm1d(64) after a Linear layer (exercise 3).")


if __name__ == "__main__":
    main()
