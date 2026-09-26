"""
level01 — Pixels, Channels, and Image Operations

Draw an RGB image with numpy alone, then implement everything:
brightness (addition), contrast (multiplication), inversion, grayscale, crop, resize.
The results are summarized in a single comparison board: outputs/pixel_ops.png.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
BRIGHT_DELTA = 0.25   # brightness adjustment (addition)
CONTRAST_GAIN = 1.8   # contrast gain (multiplication)


def make_scene(size: int = 48) -> np.ndarray:
    """Draw a 'sky + red rectangle + yellow circle' scene with numpy — no downloads."""
    img = np.zeros((size, size, 3), dtype="float32")
    # Sky: vertical gradient, bright blue (top) -> dark blue (bottom)
    grad = np.linspace(0.9, 0.3, size)[:, None]           # (size,1), vertical
    img[:, :, 2] = grad                                    # blue channel
    img[:, :, 1] = grad * 0.6                              # a touch of green
    # Red rectangle (a building): strong in the R channel only
    img[26:44, 6:22] = [0.85, 0.15, 0.10]
    # Yellow circle (the sun): strong R+G, no B
    yy, xx = np.mgrid[0:size, 0:size]
    sun = ((yy - 10) ** 2 + (xx - 36) ** 2) <= 6 ** 2
    img[sun] = [1.0, 0.9, 0.1]
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale conversion: weighted average reflecting human eye sensitivity (conventional weights)."""
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def resize_nearest(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize: each new pixel references the closest original pixel."""
    h, w = img.shape[:2]
    yy = np.clip(np.round(np.linspace(0, h - 1, new_h)), 0, h - 1).astype(int)
    xx = np.clip(np.round(np.linspace(0, w - 1, new_w)), 0, w - 1).astype(int)
    return img[yy][:, xx]      # two rounds of fancy indexing and the resize is done


def stats(name: str, a: np.ndarray) -> None:
    print(f"    {name:<26} min={a.min():>6.2f}  mean={a.mean():>5.2f}  max={a.max():>6.2f}")


def main() -> None:
    np.random.seed(1)  # fix the seed (no randomness in this level, but we keep the habit)

    # ------------------------------------------------------------------
    print("[1] Drawing an RGB image directly with numpy — array manipulation = painting")
    img = make_scene(48)
    print(f"    shape = {img.shape}  (height 48, width 48, 3 channels: R/G/B)")
    print("    Sky drawn with a gradient (linspace), rectangle with slicing, circle with a distance mask.")

    # ------------------------------------------------------------------
    print("\n[2] Splitting channels — 'being red' is just per-channel number differences")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    box = (slice(30, 40), slice(10, 20))  # inside the red rectangle
    print(f"    Channel means inside the red rectangle: R={r[box].mean():.2f}  G={g[box].mean():.2f}  B={b[box].mean():.2f}")
    print("    -> Bright only in the R channel = our eyes call it 'red'")

    # ------------------------------------------------------------------
    print("\n[3] The four pixel operations — brightness=add, contrast=multiply, invert=subtract")
    brighter_raw = img + BRIGHT_DELTA                      # before clip (on purpose)
    brighter = np.clip(brighter_raw, 0.0, 1.0)
    contrast = np.clip((img - 0.5) * CONTRAST_GAIN + 0.5, 0.0, 1.0)
    inverted = 1.0 - img
    gray = to_gray(img)
    stats("original", img)
    stats(f"brightness +{BRIGHT_DELTA} (pre-clip)", brighter_raw)
    stats(f"brightness +{BRIGHT_DELTA} (clipped)", brighter)
    stats(f"contrast x{CONTRAST_GAIN}", contrast)
    stats("invert 1-x", inverted)
    stats("grayscale", gray)
    print("    -> Note the pre-clip max exceeding 1.0. Clipping after arithmetic is a habit.")

    # ------------------------------------------------------------------
    print("\n[4] Geometric operations — crop is slicing, resize is indexing")
    crop = img[24:46, 4:24]                                # just around the red rectangle
    up = resize_nearest(img, 96, 96)                       # 2x upscale
    down = resize_nearest(img, 16, 16)                     # 1/3 downscale
    print(f"    crop:     {img.shape} -> {crop.shape}   (img[24:46, 4:24])")
    print(f"    upscale:  {img.shape} -> {up.shape}  (no new information, just staircase artifacts)")
    print(f"    downscale:{img.shape} -> {down.shape}  ({img[:, :, 0].size} numbers -> {down[:, :, 0].size})")

    # ------------------------------------------------------------------
    print("\n[5] Saving the comparison-board PNG")
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = [
        ("original (RGB)", img, None), ("R channel", r, "gray"),
        ("G channel", g, "gray"), ("B channel", b, "gray"),
        (f"brightness +{BRIGHT_DELTA}", brighter, None), (f"contrast x{CONTRAST_GAIN}", contrast, None),
        ("inverted", inverted, None), ("grayscale", gray, "gray"),
        ("crop", crop, None), ("resize up 96x96", up, None),
        ("resize down 16x16", down, None), ("normalized (z-score)", (gray - gray.mean()) / gray.std(), "gray"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for ax, (title, im, cmap) in zip(axes.ravel(), panels):
        if cmap == "gray" and title.startswith("normalized"):
            ax.imshow(im, cmap="gray")                     # z-score picks its own range
        elif cmap == "gray":
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Pixel & channel operations (numpy only)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pixel_ops.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved: {path}")

    print("\n[Recap] Image editing = array arithmetic. Brightness is addition, contrast multiplication, crop slicing.")
    print("        Next level: looking at pixels 'together with their neighbors', not alone — convolution.")


if __name__ == "__main__":
    main()
