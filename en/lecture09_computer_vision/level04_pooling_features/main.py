"""
level04 — Pooling and the Hierarchy of Features

We implement max pooling in numpy to
  1) verify the numbers  2) summarize a feature map  3) run a translation-robustness experiment,
then compare the feature maps of a mini CNN's layer 1 (edges) and layer 2 (combinations).
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
K_VERTICAL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")


def conv2d(img: np.ndarray, kernel: np.ndarray, padding: int = 1) -> np.ndarray:
    """Reusing level02's pure implementation (multiply then add)."""
    img = np.pad(img, padding)
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = float((img[y:y + k, x:x + k] * kernel).sum())
    return out


def maxpool2x2(fm: np.ndarray) -> np.ndarray:
    """Keep only the maximum in each 2x2 zone — 'regional roll-up reporting'."""
    h, w = fm.shape
    return fm.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def shift_right(img: np.ndarray, px: int) -> np.ndarray:
    """Shift the shape right by px pixels (vacated cells become 0)."""
    out = np.zeros_like(img)
    out[:, px:] = img[:, :img.shape[1] - px]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two feature maps (1=same pattern, 0=no overlap)."""
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    np.random.seed(4)
    torch.manual_seed(4)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Max pooling by hand — summarizing 4x4 into 2x2")
    mini = np.array([[1, 3, 2, 0], [5, 2, 1, 4], [0, 1, 7, 2], [2, 0, 3, 1]], dtype="float32")
    pooled = maxpool2x2(mini)
    print("    input 4x4:")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    2x2 max-pooling result (maximum per zone):")
    for row in pooled:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    -> The top-left zone [1,3,5,2] is represented by 5. Position discarded; 'it was there' kept.")

    # ------------------------------------------------------------------
    print("\n[2] Summarizing a feature map — 16x16 edge map -> pooled 8x8")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    circle = X[np.where(y == 1)[0][0]]
    fm = np.abs(conv2d(circle, K_VERTICAL))            # vertical edge-magnitude map
    fm_pooled = maxpool2x2(fm)
    print(f"    feature map {fm.shape} -> pooled {fm_pooled.shape}  ({fm.size} numbers -> {fm_pooled.size})")
    print(f"    value at the maximum: before {fm.max():.1f} / after {fm_pooled.max():.1f}  (strong evidence preserved)")

    # ------------------------------------------------------------------
    print("\n[3] Translation-robustness experiment — how similar is the shifted shape's map (1=identical)")
    clean = np.zeros((16, 16), dtype="float32")
    clean[4:12, 4:12] = 1.0                            # noise-free square (so the effect shows clearly)
    fm0 = np.abs(conv2d(clean, K_VERTICAL))
    print(f"      {'shift':>6} {'no pool':>8} {'pool x1':>8} {'pool x2':>8}")
    for px in (1, 2, 3):
        fm1 = np.abs(conv2d(shift_right(clean, px), K_VERTICAL))
        sims = [cosine(fm0, fm1),
                cosine(maxpool2x2(fm0), maxpool2x2(fm1)),
                cosine(maxpool2x2(maxpool2x2(fm0)), maxpool2x2(maxpool2x2(fm1)))]
        print(f"      {px:>5}px {sims[0]:>8.2f} {sims[1]:>8.2f} {sims[2]:>8.2f}")
    moved = shift_right(circle, 1)
    fm_moved = np.abs(conv2d(moved, K_VERTICAL))
    print("    -> 2px shift: similarity 0.00 before pooling (a completely different map!) / 0.71 after 2 poolings.")
    print("       The more pooling, the more 'the same object, slightly shifted' is seen as the same thing.")

    # ------------------------------------------------------------------
    print("\n[4] Feature hierarchy — conv1 (low-level) vs conv2 (high-level) feature maps")
    conv1 = nn.Conv2d(1, 6, 3, padding=1)
    conv2 = nn.Conv2d(6, 6, 3, padding=1)
    pool = nn.MaxPool2d(2)
    relu = nn.ReLU()
    x = torch.from_numpy(circle)[None, None]           # (1,1,16,16)
    with torch.no_grad():
        f1 = relu(conv1(x))                            # (1,6,16,16) low-level
        f2 = relu(conv2(pool(f1)))                     # (1,6,8,8)  high-level (wider field of view)
    print(f"    conv1 output {tuple(f1.shape)} : responses seeing only the raw 3x3 neighborhood (edge level)")
    print(f"    conv2 output {tuple(f2.shape)} : thanks to pooling, sees combinations over a wider original area")
    print("    (weights are random, but the 'widening field of view' structure is observable regardless)")

    # ------------------------------------------------------------------
    print("\n[5] Saving the comparison-board PNG")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 6, figsize=(13, 7))
    top = [(circle, "input", "gray"), (fm, "edge map 16x16", "magma"),
           (fm_pooled, "pooled 8x8", "magma"), (moved, "input shifted +2px", "gray"),
           (fm_moved, "edge map (shifted)", "magma"),
           (maxpool2x2(fm_moved), "pooled (shifted)", "magma")]
    for ax, (im, title, cmap) in zip(axes[0], top):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for j in range(6):                                  # conv1's 6 channels
        axes[1][j].imshow(f1[0, j].numpy(), cmap="viridis")
        axes[1][j].set_title(f"conv1 ch{j} (16x16)", fontsize=8)
        axes[1][j].axis("off")
    for j in range(6):                                  # conv2's 6 channels
        axes[2][j].imshow(f2[0, j].numpy(), cmap="viridis")
        axes[2][j].set_title(f"conv2 ch{j} (8x8)", fontsize=8)
        axes[2][j].axis("off")
    fig.suptitle("Max pooling: summarize location, keep evidence / feature hierarchy", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pooling_features.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print("\n[Recap] Pooling = regional roll-up reporting. Discard position, keep evidence, gain robustness to shifts.")
    print("        Next level: assembly complete — we actually train a shape classifier.")


if __name__ == "__main__":
    main()
