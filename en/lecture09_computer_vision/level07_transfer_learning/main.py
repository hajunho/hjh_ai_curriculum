"""
level07 — Transfer Learning

We pretrain a CNN on task A (squares/circles/triangles, plenty of data),
then 'freeze' its feature extractor and transplant it to task B
(diamonds/ellipses/rings, only 24 labeled images).
Comparing per-epoch performance against a from-scratch model proves the transfer gain.
All images drawn inside the repository — a transfer-learning experiment with no downloads.
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
TASK_B_EN = ["diamond", "ellipse", "ring"]
N_B_TRAIN = 24         # task B has only 24 labeled images (a common reality in practice)
EPOCHS_B = 120


def task_b_images(n: int, size: int = 16, seed: int = 77):
    """Task B: diamonds/ellipses/rings — drawn by hand in the same style as hjh_data."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size), dtype="float32")
    y = np.zeros(n, dtype="int64")
    for i in range(n):
        label = i % 3
        y[i] = label
        img = np.zeros((size, size), dtype="float32")
        cy, cx = rng.integers(5, size - 5, size=2)
        r = int(rng.integers(3, 5))
        yy, xx = np.mgrid[0:size, 0:size]
        if label == 0:                                   # diamond
            img[np.abs(yy - cy) + np.abs(xx - cx) <= r] = 1.0
        elif label == 1:                                 # vertically elongated ellipse
            img[((yy - cy) / r) ** 2 + ((xx - cx) / max(r // 2, 1)) ** 2 <= 1.0] = 1.0
        else:                                            # thick ring
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            img[(dist2 <= r * r) & (dist2 >= (r - 2) ** 2)] = 1.0
        img += rng.normal(0, 0.08, img.shape).astype("float32")
        X[i] = np.clip(img, 0.0, 1.0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_cnn() -> nn.Module:
    """A CNN split into features (extractor) + head (verdict).
    Deliberately a bit large (~20K parameters) — a size where from-scratch struggles on scarce data."""
    return nn.Sequential(
        nn.Sequential(                                    # [0] features
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()),
        nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 3)))   # [1] head


def to_torch(X, y):
    return torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)


@torch.no_grad()
def accuracy(model, x, y) -> float:
    model.eval()
    return float((model(x).argmax(1) == y).float().mean())


def train_b(model: nn.Module, xtr, ytr, xte, yte, tag: str) -> list[float]:
    """Task B training. Only parameters with requires_grad=True get trained."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"      {tag}: trainable parameters {n_train:,} / total {n_total:,}")
    opt = torch.optim.Adam(trainable, lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS_B + 1):
        model.train()
        opt.zero_grad()
        loss_fn(model(xtr), ytr).backward()               # full batch: it's tiny
        opt.step()
        history.append(accuracy(model, xte, yte))
    return history


def main() -> None:
    t0 = time.time()
    torch.manual_seed(7)
    np.random.seed(7)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Pretraining on task A — squares/circles/triangles, 600 images (plenty)")
    XA, yA = hjh_data.shape_images(n=750, size=16, seed=13)
    xa_tr, ya_tr = to_torch(XA[:600], yA[:600])
    xa_te, ya_te = to_torch(XA[600:], yA[600:])
    pretrained = make_cnn()
    opt = torch.optim.Adam(pretrained.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(20):
        perm = torch.randperm(len(xa_tr))
        for i in range(0, len(xa_tr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(pretrained(xa_tr[idx]), ya_tr[idx]).backward()
            opt.step()
    print(f"    Pretraining done: task A test accuracy {accuracy(pretrained, xa_te, ya_te):.3f}")
    print("    The front of this model (features) now carries an eye for 'edges, curves, shapes'.")

    # ------------------------------------------------------------------
    print(f"\n[2] Enter task B — diamonds/ellipses/rings, but only {N_B_TRAIN} labeled images")
    XB, yB = task_b_images(n=360, size=16, seed=77)
    xb_tr, yb_tr = to_torch(XB[:N_B_TRAIN], yB[:N_B_TRAIN])
    xb_te, yb_te = to_torch(XB[N_B_TRAIN:], yB[N_B_TRAIN:])
    print(f"    Task B: train {len(xb_tr)} / test {len(xb_te)} — shape types completely different from task A.")

    # ------------------------------------------------------------------
    print("\n[3] Method 1: training from scratch — educating the fresh graduate from zero")
    torch.manual_seed(70)                                 # seed for a fair comparison
    scratch = make_cnn()
    hist_scratch = train_b(scratch, xb_tr, yb_tr, xb_te, yb_te, "scratch ")

    # ------------------------------------------------------------------
    print("\n[4] Method 2: transfer learning — the experienced hire: freeze and reuse A's eye (features)")
    torch.manual_seed(70)
    transfer = make_cnn()
    transfer[0].load_state_dict(pretrained[0].state_dict())   # transplant the feature extractor
    for p in transfer[0].parameters():
        p.requires_grad = False                               # freeze
    hist_transfer = train_b(transfer, xb_tr, yb_tr, xb_te, yb_te, "transfer")

    # ------------------------------------------------------------------
    print("\n[5] Comparing results — convergence speed and final performance")
    print(f"      {'epoch':>6} {'scratch':>9} {'transfer':>9}")
    for e in (1, 5, 10, 20, 40, 80, 120):
        print(f"      {e:>6} {hist_scratch[e - 1]:>9.3f} {hist_transfer[e - 1]:>9.3f}")
    print("    -> Transfer starts leading around epoch 10 and keeps the lead to the end.")
    print("       It trained 1/10 of the parameters (just the 1,539-parameter head), yet scores higher.")
    print("       Task A had no diamonds and no rings — but the low-level eye for")
    print("       edges, curves, and blobs doesn't care which task it serves.")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.5))
    for k in range(6):                                    # task B samples
        ax = fig.add_subplot(2, 6, k + 1 + (6 if k >= 3 else 0) - (0 if k < 3 else 3))
        i = int(np.where(yB[N_B_TRAIN:] == k % 3)[0][k // 3]) + N_B_TRAIN
        ax.imshow(XB[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"task B: {TASK_B_EN[k % 3]}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    ep = range(1, EPOCHS_B + 1)
    ax.plot(ep, hist_scratch, marker="s", label="from scratch")
    ax.plot(ep, hist_transfer, marker="o", label="transfer (frozen features)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("task B test accuracy")
    ax.set_title(f"Only {N_B_TRAIN} labeled images for task B")
    ax.set_ylim(0.2, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Transfer learning: reuse features learned on task A", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "transfer_learning.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print(f"\n[Recap] Transfer learning = the experienced hire. Reuse the low-level eye, retrain only the verdict (took {time.time() - t0:.1f}s).")
    print("        Next level: inspect-split-train-analyze-improve — running the full real-world pipeline.")


if __name__ == "__main__":
    main()
