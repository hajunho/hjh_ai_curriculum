"""
level00 — How Computers See Images

We confirm that an image is nothing but a "grid of numbers", in three representations:
  1) print the raw number table  2) text art (brightness -> character)  3) save a PNG
As a bonus, we lower the resolution step by step to feel that "tile count" = information.
"""

import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CLASS_NAMES = ["square", "circle", "triangle"]
CLASS_NAMES_EN = ["square", "circle", "triangle"]  # for figures
CHARS = " .:-=+*#@"  # character palette, ordered dark -> bright


def to_ascii(img: np.ndarray) -> str:
    """Turn brightness (0-1) into characters to make 'text art'.
    Rendering = a rule for converting numbers into readable symbols, nothing more."""
    lines = []
    for row in img:
        idx = (np.clip(row, 0.0, 1.0) * (len(CHARS) - 1)).astype(int)
        # Terminal characters are tall, so print 2 chars per pixel to fix the aspect ratio
        lines.append("".join(CHARS[i] * 2 for i in idx))
    return "\n".join(lines)


def downscale_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Lower the resolution by collapsing each factor x factor block into its mean."""
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def main() -> None:
    np.random.seed(0)  # fix the random seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Creating shape-image data — generated on the spot with numpy, no downloads")
    X, y = hjh_data.shape_images(n=300, size=16, seed=13)
    print(f"    X.shape = {X.shape}  (300 images, each a 16-row x 16-col number table)")
    print(f"    value range = {X.min():.2f} ~ {X.max():.2f}  (0=black, 1=white)")
    counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(3)}
    print(f"    class composition = {counts}")

    # Pick one circle (label=1) image as our representative
    sample = X[np.where(y == 1)[0][0]]

    # ------------------------------------------------------------------
    print("\n[2] One image as a raw 'number table' — the original the computer sees")
    print("    (only the first decimal shown; the shape is where the 1.0s cluster)")
    for row in sample:
        print("    " + " ".join(f"{v:.1f}"[1:] for v in row))  # shorten '0.7' -> '.7'
    print("    -> To the human eye it's just a pile of numbers, but this IS the whole image.")

    # ------------------------------------------------------------------
    print("\n[3] The same array as 'text art' — one number->character rule added")
    print(to_ascii(sample))
    print("    -> The array is unchanged, yet the shape (a circle) appears. A picture is an arrangement of numbers.")

    # ------------------------------------------------------------------
    print("\n[4] Resolution experiment — what happens as we shrink the mosaic tile count")
    for factor, name in [(1, "16x16 (original)"), (2, "8x8"), (4, "4x4")]:
        small = sample if factor == 1 else downscale_mean(sample, factor)
        print(f"\n    --- {name}: {small.size} numbers ---")
        for line in to_ascii(small).split("\n"):
            print("    " + line)
    print("\n    -> Around 4x4 it becomes hard to tell a circle from a square.")
    print("       Resolution is both 'information' and 'compute cost'. Use only what the task needs.")

    # ------------------------------------------------------------------
    print("\n[5] Saving as PNG — the very same array we've been looking at, as an image file")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for c in range(3):  # 3 samples per class
        idx = np.where(y == c)[0][:3]
        for j, i in enumerate(idx):
            ax = axes[c][j]
            ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{CLASS_NAMES_EN[c]} (y={c})", fontsize=9)
            ax.axis("off")
    fig.suptitle("shape_images: 16x16 grayscale, 3 classes", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "shapes_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")
    print("    Open the file — it is exactly the same data as the number table in [2].")

    print("\n[Recap] An image = a grid (spreadsheet) of brightness numbers.")
    print("        Next level: if images are numbers, editing is arithmetic — pixels, channels, image operations.")


if __name__ == "__main__":
    main()
