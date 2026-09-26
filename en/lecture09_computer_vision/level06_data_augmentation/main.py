"""
level06 — Data Augmentation

We implement rotation, shift, flip, and noise augmentation directly in numpy,
then run a controlled comparison — same model, same settings — of test performance
with and without augmentation when training data is scarce (120 images).
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
N_TRAIN = 120          # deliberately create a 'data-scarce' situation
EPOCHS = 30
LR = 1e-3


# ----------------------------------------------------------------------
# numpy augmentation functions — pure array operations, no libraries
# ----------------------------------------------------------------------

def aug_rotate90(img: np.ndarray, k: int) -> np.ndarray:
    """Rotation in 90-degree steps. The shape's class is unchanged (a label-preserving transform)."""
    return np.rot90(img, k).copy()


def aug_flip(img: np.ndarray) -> np.ndarray:
    """Horizontal flip."""
    return img[:, ::-1].copy()


def aug_shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift up/down/left/right (vacated cells become 0). np.roll, then erase what wrapped around."""
    out = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    if dy > 0:
        out[:dy] = 0
    elif dy < 0:
        out[dy:] = 0
    if dx > 0:
        out[:, :dx] = 0
    elif dx < 0:
        out[:, dx:] = 0
    return out


def aug_noise(img: np.ndarray, rng: np.random.Generator, std: float = 0.10) -> np.ndarray:
    """Add Gaussian noise — imitating jitter in shooting conditions."""
    return np.clip(img + rng.normal(0, std, img.shape), 0.0, 1.0).astype("float32")


def augment_dataset(X: np.ndarray, y: np.ndarray, per_image: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create per_image random augmented copies of every image and concatenate with the originals."""
    outs, labels = [X], [y]
    for _ in range(per_image):
        batch = np.empty_like(X)
        for i, img in enumerate(X):
            a = img
            a = aug_rotate90(a, int(rng.integers(0, 4)))     # rotate 0/90/180/270
            if rng.random() < 0.5:
                a = aug_flip(a)                               # flip
            a = aug_shift(a, int(rng.integers(-2, 3)), int(rng.integers(-2, 3)))
            a = aug_noise(a, rng)                             # noise
            batch[i] = a
        outs.append(batch)
        labels.append(y)
    return np.concatenate(outs), np.concatenate(labels)


# ----------------------------------------------------------------------
# Small CNN (same architecture as level05)
# ----------------------------------------------------------------------

def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(256, 3))


def train_and_eval(xtr, ytr, xte, yte, tag: str) -> float:
    """Train with fixed settings and return test accuracy (fair comparison)."""
    torch.manual_seed(6)                                      # identical initial weights for both runs
    model = make_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(xte).argmax(1) == yte).float().mean())
    print(f"      {tag:<28} train {len(xtr):>4} imgs -> test accuracy {acc:.3f}")
    return acc


def to_torch(X: np.ndarray, y: np.ndarray):
    return torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(6)
    np.random.seed(6)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Creating a data-scarce situation — train 120 / test 300")
    X, y = hjh_data.shape_images(n=420, size=16, seed=13)
    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xte, yte = X[N_TRAIN:], y[N_TRAIN:]
    print(f"    A common situation in practice: only {N_TRAIN} labeled images.")

    # ------------------------------------------------------------------
    print("\n[2] Implementing augmentation — making transformed copies of one original")
    demo = Xtr[0]
    variants = [("original", demo),
                ("rotate 90", aug_rotate90(demo, 1)),
                ("rotate 180", aug_rotate90(demo, 2)),
                ("flip", aug_flip(demo)),
                ("shift (+2,+2)", aug_shift(demo, 2, 2)),
                ("noise", aug_noise(demo, rng))]
    print("    Rotate/flip/shift/noise — all transforms that 'do not change the label'.")
    print(f"    (a {CLASS_EN[int(ytr[0])]} rotated, shifted, and noised is still a {CLASS_EN[int(ytr[0])]})")

    # ------------------------------------------------------------------
    print("\n[3] Inflating the dataset with augmentation — 120 -> 600 images")
    Xaug, yaug = augment_dataset(Xtr, ytr, per_image=4, rng=rng)
    print(f"    {len(Xtr)} originals + {len(Xaug) - len(Xtr)} augmented copies = {len(Xaug)} images")
    print("    (the 300 test images are never augmented — the exam paper stays original)")

    # ------------------------------------------------------------------
    print("\n[4] The fair comparison — same model, same epochs, same initial weights")
    xte_t, yte_t = to_torch(Xte, yte)
    acc_plain = train_and_eval(*to_torch(Xtr, ytr), xte_t, yte_t, "no augmentation (120 imgs)")
    acc_aug = train_and_eval(*to_torch(Xaug, yaug), xte_t, yte_t, "with augmentation (600 imgs)")
    gain = (acc_aug - acc_plain) * 100
    print(f"    -> Augmentation effect: test accuracy {gain:+.1f} points. Gained without creating a single new image.")

    # ------------------------------------------------------------------
    print("\n[5] Saving the comparison-board PNG")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(11, 6))
    for k, (name, im) in enumerate(variants):
        ax = fig.add_subplot(2, 6, k + 1)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    bars = ax.bar(["no augmentation\n(120 imgs)", "with augmentation\n(600 imgs)"],
                  [acc_plain, acc_aug], width=0.4)
    for b, v in zip(bars, [acc_plain, acc_aug]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("test accuracy (300 imgs)")
    ax.set_title("Same model & epochs - augmentation only difference")
    fig.suptitle("Data augmentation: label-preserving transforms buy accuracy for free", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "augmentation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print(f"\n[Recap] Augmentation = free data via label-preserving transforms (took {time.time() - t0:.1f}s).")
    print("        Next level: an even bigger lever — borrowing someone else's training: transfer learning.")


if __name__ == "__main__":
    main()
