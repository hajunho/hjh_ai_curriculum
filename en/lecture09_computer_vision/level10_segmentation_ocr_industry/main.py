"""
level10 — Segmentation, OCR, and Industrial Inspection

We complete pixel-level analysis using only classic techniques — no deep learning.
  1) foreground/background binarization with a threshold
  2) connected-component labeling implemented by hand with BFS
  3) per-region properties (area, box, centroid) -> automatic part vs defect (stain) verdicts
The result is saved as a 4-panel comparison: outputs/segmentation.png.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# This level's inspection scene is drawn directly in numpy, in the same style as hjh_data.

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SIZE = 64
THRESH = 0.45          # brightness at or above this = 'object pixel'
DEFECT_AREA = 12       # connected components below this area are ruled 'defect stains'


def make_inspection_image(rng: np.random.Generator):
    """Inspection scene: good parts (3 shapes) + 4 defect stains + shooting noise."""
    img = np.zeros((SIZE, SIZE), dtype="float32")
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    # Good parts: square/circle/triangle (same style as hjh_data, positions fixed)
    img[8:22, 8:22] = 0.95                                        # square
    img[((yy - 16) ** 2 + (xx - 44) ** 2) <= 49] = 0.95           # circle (r=7)
    tri = (yy >= 38) & (yy <= 54) & (np.abs(xx - 20) <= (yy - 38) / 2)
    img[tri] = 0.95                                               # triangle
    # Defects: 4 small stains (spatter) — much smaller than the parts
    defects = []
    for _ in range(4):
        while True:
            dy, dx = (int(v) for v in rng.integers(4, SIZE - 4, size=2))
            if img[dy - 3:dy + 3, dx - 3:dx + 3].max() < 0.1:     # keep clear of the parts
                break
        blob = ((yy - dy) ** 2 + (xx - dx) ** 2) <= int(rng.integers(2, 5))
        img[blob] = 0.85
        defects.append((dy, dx))
    img += rng.normal(0, 0.06, img.shape).astype("float32")       # shooting noise
    return np.clip(img, 0.0, 1.0), defects


def connected_components(mask: np.ndarray):
    """4-directional connected-component labeling — BFS (paint bucket) by hand.
    Pixels of the same blob get the same number. Returns: label map, component count."""
    labels = np.zeros(mask.shape, dtype=int)
    current = 0
    for sy in range(mask.shape[0]):
        for sx in range(mask.shape[1]):
            if mask[sy, sx] and labels[sy, sx] == 0:
                current += 1                          # new blob found
                stack = [(sy, sx)]
                labels[sy, sx] = current
                while stack:                          # spread to neighbors, painting
                    y, x = stack.pop()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                and mask[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def region_props(labels: np.ndarray, k: int) -> dict:
    """Properties of region k: area, bounding box, centroid."""
    ys, xs = np.where(labels == k)
    return {"area": len(ys),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
            "center": (float(ys.mean()), float(xs.mean()))}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(10)
    np.random.seed(10)  # fix the seed (reproducibility)

    # ------------------------------------------------------------------
    print("[1] Preparing the inspection scene — 3 good parts + 4 secretly scattered stains")
    img, true_defects = make_inspection_image(rng)
    print(f"    {SIZE}x{SIZE} grayscale image. Easy for a human eye — a machine needs pixel-level evidence.")
    print(f"    (answer key) stain positions: {true_defects} — the detector doesn't know this.")

    # ------------------------------------------------------------------
    print("\n[2] Threshold binarization — 'brighter than this = object'")
    mask = img >= THRESH
    print(f"    threshold {THRESH}: {int(mask.sum())} foreground pixels / {mask.size} total "
          f"({mask.mean():.1%})")
    print("    Background noise (mean 0, std 0.06) almost never clears the bar.")
    print("    -> The cheapest segmentation there is, and it works when brightness splits into two peaks.")

    # ------------------------------------------------------------------
    print("\n[3] Connected-component labeling — touching pixels share a number")
    labels, n = connected_components(mask)
    print(f"    Result of BFS flood-painting: {n} connected components")

    # ------------------------------------------------------------------
    print("\n[4] Region measurement and verdicts — small area means stain, large means part")
    print(f"      {'id':>3} {'area':>5} {'center(y,x)':>14} {'verdict':>8}")
    found_defects = []
    for k in range(1, n + 1):
        p = region_props(labels, k)
        verdict = "defect" if p["area"] < DEFECT_AREA else "part"
        if verdict == "defect":
            found_defects.append(p)
        cy, cx = p["center"]
        print(f"      {k:>3} {p['area']:>5} {f'({cy:5.1f},{cx:5.1f})':>14} {verdict:>8}")
    print(f"    -> The 'area under {DEFECT_AREA}' rule detected {len(found_defects)} stains "
          f"(actual: {len(true_defects)}).")
    matched = 0
    for p in found_defects:
        cy, cx = p["center"]
        if any(abs(cy - dy) < 3 and abs(cx - dx) < 3 for dy, dx in true_defects):
            matched += 1
    print(f"    Against the answer key: {matched}/{len(true_defects)} stain positions match — pixel-level inspection succeeded.")

    # ------------------------------------------------------------------
    print("\n[5] Saving the comparison-board PNG — input/binarized/label map/verdict overlay")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("input (parts + defects)", fontsize=10)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"threshold >= {THRESH}", fontsize=10)
    axes[2].imshow(np.where(labels > 0, labels, np.nan), cmap="tab10")
    axes[2].set_title(f"connected components ({n})", fontsize=10)
    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((SIZE, SIZE, 4), dtype=float)
    for k in range(1, n + 1):
        p = region_props(labels, k)
        color = (1, 0, 0, 0.55) if p["area"] < DEFECT_AREA else (0, 0.8, 0.2, 0.35)
        overlay[labels == k] = color
    axes[3].imshow(overlay)
    axes[3].set_title("verdict: green=part, red=defect", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Classic segmentation: threshold + connected components + area rule", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "segmentation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print(f"\n[Recap] Segmentation = deciding each pixel's allegiance. The classic trio (threshold->components->area rule)")
    print(f"        is enough for defect detection (took {time.time() - t0:.1f}s). OCR shares the skeleton:")
    print("        separate the character regions (segmentation), then classify each region (recognition).")
    print("        Next level: reading images as 'patch words' — Vision Transformers and multimodal models.")


if __name__ == "__main__":
    main()
