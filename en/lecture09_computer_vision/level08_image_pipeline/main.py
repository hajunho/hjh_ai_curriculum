"""
level08 — A Real-World Image Classification Pipeline

We run the full loop: data inspection -> split -> train -> error analysis -> improve.
After deliberately corrupting 8% of the training labels (label noise),
we demonstrate that the error-analysis step — 'human review of high-loss
training samples' — hunts down the corrupted labels.
"""

import os
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_EN = ["square", "circle", "triangle"]
NOISE_RATE = 0.08      # corrupt 8% of the training labels into wrong classes
EPOCHS = 18


def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(256, 3))


def train_model(xtr, ytr, seed: int = 8) -> nn.Module:
    """Training with fixed settings (for controlled comparison)."""
    torch.manual_seed(seed)
    model = make_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


@torch.no_grad()
def per_sample_loss(model, x, y) -> np.ndarray:
    """Per-sample cross-entropy — 'how much the model cannot make sense of it'."""
    model.eval()
    return nn.CrossEntropyLoss(reduction="none")(model(x), y).numpy()


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(8)
    np.random.seed(8)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Data inspection — look at the data before the model")
    X, y_true = hjh_data.shape_images(n=900, size=16, seed=13)
    print(f"    count: {len(X)} / size: {X.shape[1]}x{X.shape[2]} / value range: {X.min():.2f}~{X.max():.2f}")
    counts = np.bincount(y_true)
    print(f"    class balance: " + ", ".join(f"{CLASS_EN[c]} {counts[c]}" for c in range(3)) + "  (balanced, OK)")

    # Recreating reality: suppose some training labels were misattached during outsourced labeling
    y_noisy = y_true.copy()
    n_corrupt = int(600 * NOISE_RATE)
    corrupt_idx = rng.choice(600, size=n_corrupt, replace=False)
    for i in corrupt_idx:
        y_noisy[i] = (y_true[i] + int(rng.integers(1, 3))) % 3   # always a different class
    print(f"    (scenario) {n_corrupt} of the 600 training labels ({NOISE_RATE:.0%}) are wrong — and we pretend not to know!")

    # ------------------------------------------------------------------
    print("\n[2] Split — train 600 / val 150 / test 150 (val and test labels are clean)")
    to_t = lambda a: torch.from_numpy(a)
    xt = to_t(X).unsqueeze(1)
    xtr, ytr = xt[:600], to_t(y_noisy[:600])
    xva, yva = xt[600:750], to_t(y_true[600:750])
    xte, yte = xt[750:], to_t(y_true[750:])

    # ------------------------------------------------------------------
    print("\n[3] First training — what happens when we train on corrupted labels?")
    model_v1 = train_model(xtr, ytr)
    acc_v1 = accuracy(model_v1, xte, yte)
    print(f"    v1 test accuracy: {acc_v1:.3f}  (low for shape classification… why?)")

    # ------------------------------------------------------------------
    print("\n[4] Error analysis — re-reviewing 'the training samples the model never made sense of'")
    losses = per_sample_loss(model_v1, xtr, ytr)
    order = np.argsort(-losses)                        # descending by loss
    top_k = n_corrupt + 12                             # review a bit more than the corruption count
    flagged = order[:top_k]
    hit = np.isin(flagged, corrupt_idx).sum()
    print(f"    Suppose a human re-reviews the top {top_k} by training loss.")
    print(f"    -> truly corrupted labels among them: {hit} / detected {hit / n_corrupt:.0%} of all {n_corrupt} corruptions!")
    print(f"    Picking {top_k} at random would be expected to catch {top_k * n_corrupt / 600:.1f}.")
    print("    The simple rule 'look at high-loss samples first' works as a label-error detector.")

    # ------------------------------------------------------------------
    print("\n[5] Improvement loop — fix the reviewed labels and retrain")
    y_fixed = y_noisy.copy()
    y_fixed[flagged] = y_true[flagged]                 # reviewed samples get their correct labels back
    model_v2 = train_model(xtr, to_t(y_fixed[:600]))
    acc_v2 = accuracy(model_v2, xte, yte)
    model_oracle = train_model(xtr, to_t(y_true[:600]))   # reference: if it had been clean all along
    acc_oracle = accuracy(model_oracle, xte, yte)
    print(f"      {'version':<30} {'test acc':>10}")
    print(f"      {'v1: corrupted labels as-is':<30} {acc_v1:>10.3f}")
    print(f"      {'v2: top-loss review & fix':<30} {acc_v2:>10.3f}")
    print(f"      {'ref: fully clean (ideal)':<30} {acc_oracle:>10.3f}")
    print(f"    -> Same model architecture; fixing only the data bought +{(acc_v2 - acc_v1) * 100:.1f} points.")
    print("       In practice, the cheapest performance button is often 'data cleaning'.")

    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 6))
    for k in range(8):                                  # top-8 loss = the review list
        i = int(order[k])
        ax = fig.add_subplot(2, 8, k + 1 + (8 if k >= 4 else 0) - (0 if k < 4 else 4))
        ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
        bad = i in corrupt_idx
        ax.set_title(f"label {CLASS_EN[int(y_noisy[i])]}\n{'WRONG!' if bad else 'hard'}",
                     fontsize=7, color="red" if bad else "black")
        ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    names = ["v1 noisy labels", "v2 after review", "oracle clean"]
    vals = [acc_v1, acc_v2, acc_oracle]
    bars = ax.bar(names, vals, width=0.5, color=["#c66", "#69c", "#6a6"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Error analysis loop: fix data, not model")
    fig.suptitle("Left: highest-loss training samples for human review (red = truly mislabeled)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pipeline_error_analysis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print(f"\n[Recap] Inspect -> split -> train -> analyze errors -> improve. Skill is the number of loops you've run (took {time.time() - t0:.1f}s).")
    print("        Next level: beyond 'what is there' to 'where it is' — how object detection works.")


if __name__ == "__main__":
    main()
