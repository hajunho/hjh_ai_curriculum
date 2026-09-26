"""
level02 — Understanding Convolution

We implement convolution from scratch with numpy loops.
  1) trace every multiply-add on a 6x6 mini example
  2) detect shape boundaries with vertical/horizontal edge kernels
  3) show that changing only the kernel numbers gives blur/sharpen + a PNG comparison board
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

# The inspection stamps: a kernel = a number board engraved with "what to look for"
K_VERTICAL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32")   # vertical boundaries
K_HORIZONTAL = K_VERTICAL.T                                                     # horizontal boundaries
K_BLUR = np.full((3, 3), 1.0 / 9.0, dtype="float32")                            # mean = blur
K_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype="float32")    # sharpen


def conv2d(img: np.ndarray, kernel: np.ndarray, padding: int = 0) -> np.ndarray:
    """All of convolution: slide the kernel and 'multiply then add'. (pure educational implementation)"""
    if padding > 0:
        img = np.pad(img, padding)                    # border of zeros around the edges
    h, w = img.shape
    k = kernel.shape[0]
    out = np.zeros((h - k + 1, w - k + 1), dtype="float32")
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = img[y:y + k, x:x + k]             # the area the stamp covers
            out[y, x] = float((patch * kernel).sum()) # sum of the 9 products
    return out


def main() -> None:
    np.random.seed(2)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Mini example — tracing one convolution number by number")
    mini = np.zeros((6, 6), dtype="float32")
    mini[:, 3:] = 1.0                                  # vertical boundary: dark left, bright right
    print("    input 6x6 (left=0, right=1, a vertical boundary):")
    for row in mini:
        print("      " + " ".join(f"{v:.0f}" for v in row))
    print("    kernel (vertical edge):")
    for row in K_VERTICAL:
        print("      " + " ".join(f"{v:+.0f}" for v in row))
    patch = mini[0:3, 2:5]                             # first stamp position straddling the boundary
    print("    Stamping at position (0,2) — the 9 products:")
    terms = []
    for i in range(3):
        for j in range(3):
            terms.append(f"{patch[i, j]:.0f}x{K_VERTICAL[i, j]:+.0f}")
    print("      " + "  ".join(terms))
    print(f"      sum = {(patch * K_VERTICAL).sum():+.0f}  (large because it's a boundary)")
    flat = mini[0:3, 0:3]
    print(f"    sum at flat position (0,0) = {(flat * K_VERTICAL).sum():+.0f}  (no change -> 0)")

    # ------------------------------------------------------------------
    print("\n[2] Verifying conv2d — output size and padding")
    out = conv2d(mini, K_VERTICAL)
    out_pad = conv2d(mini, K_VERTICAL, padding=1)
    print(f"    no padding: {mini.shape} -> {out.shape}   (shrinks to n-k+1)")
    print(f"    padding 1 : {mini.shape} -> {out_pad.shape}   (size preserved)")
    print("    output (no padding) — large values only in the boundary columns:")
    for row in out:
        print("      " + " ".join(f"{v:+4.0f}" for v in row))

    # ------------------------------------------------------------------
    print("\n[3] Edge kernels on shapes — each kernel asks a different question")
    X, y = hjh_data.shape_images(n=60, size=16, seed=13)
    names = ["square", "circle", "triangle"]
    samples = [X[np.where(y == c)[0][0]] for c in range(3)]
    sq_v = np.abs(conv2d(samples[0], K_VERTICAL, padding=1))
    sq_h = np.abs(conv2d(samples[0], K_HORIZONTAL, padding=1))
    left_right = sq_v[:, :].max(axis=0)
    print(f"    square + vertical kernel:   max response {sq_v.max():.1f} (at the left/right sides)")
    print(f"    square + horizontal kernel: max response {sq_h.max():.1f} (at the top/bottom sides)")
    print("    -> Same image, different kernel = a different thing becomes 'visible'.")
    _ = left_right  # (reference computation)

    # ------------------------------------------------------------------
    print("\n[4] Same operation, different kernel — blur and sharpen")
    circle = samples[1]
    blurred = conv2d(circle, K_BLUR, padding=1)
    sharpened = np.clip(conv2d(circle, K_SHARPEN, padding=1), 0, 1)
    print(f"    original std   = {circle.std():.3f}")
    print(f"    after blur std = {blurred.std():.3f}  (values squashed toward the mean)")
    print(f"    after sharpen  = {sharpened.std():.3f}  (differences pulled further apart)")

    # ------------------------------------------------------------------
    print("\n[5] Saving the comparison-board PNG — 3 shapes x (input/vertical/horizontal/magnitude)")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for r, (name, img) in enumerate(zip(names, samples)):
        gv = conv2d(img, K_VERTICAL, padding=1)
        gh = conv2d(img, K_HORIZONTAL, padding=1)
        mag = np.sqrt(gv ** 2 + gh ** 2)               # direction-agnostic edge magnitude
        panels = [(img, f"{name} (input)", "gray"),
                  (gv, "vertical edges", "coolwarm"),
                  (gh, "horizontal edges", "coolwarm"),
                  (mag, "edge magnitude", "magma")]
        for c, (im, title, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Hand-made 3x3 kernels: convolution as a pattern detector", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "convolution_edges.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print("\n[Recap] Convolution = sliding a number stamp and 'multiplying then adding'. The kernel IS the question.")
    print("        Next level: let the data decide the kernel numbers and stack them in layers — assembling a CNN.")


if __name__ == "__main__":
    main()
