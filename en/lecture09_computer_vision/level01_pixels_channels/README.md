# Lecture 09 · Level 01 — Pixels, Channels, and Image Operations

> Brightness is addition, contrast is multiplication, inversion is subtraction — we rebuild the buttons of a photo-editing app as numpy arithmetic.
**Difficulty** ⭐⭐ / **Prerequisites** level00 / **Estimated time** 30 min

## 1. Why This Matters — the Business View

On a vision-AI project, the work you meet before any modeling is **preprocessing**. Store CCTV footage changes brightness between day and night, scanned documents have weak contrast, and every camera ships a different resolution. Feed such images to a model as-is and performance swings wildly. The first stage of any production pipeline is always image operations like "normalize brightness, unify sizes, crop the region of interest".

These operations are not Photoshop's private property. As level00 showed, an image is a table of numbers, so **adjusting brightness = adding to every cell, adjusting contrast = multiplying every cell**. It is the same act as filling `+10` down an entire column in Excel. Once you have this intuition, you can ask "what preprocessing did you apply?" — and understand the answer.

## 2. An Analogy

Picture a **stage lighting console**.

- **Push the master fader up** → the whole stage brightens uniformly = **add** the same number to every pixel
- **Increase the dimmer gain** → bright spots get much brighter, dark spots only a little; the gap widens = **multiply** every pixel by the same number (contrast)
- **Negative film** → bright areas turn dark and dark areas turn bright = `1 - pixel value` (inversion)

And color. Stage lighting layers three lamps — red, green, blue — to produce every color. A color image works the same way: **a red brightness table, a green brightness table, a blue brightness table** — three grayscale images stacked together make an RGB color image. Each of those single sheets is called a **channel**.

## 3. Core Concepts

### 3.1 Channels — color is three grayscales

A color image is a `(height, width, 3)` array. `img[:, :, 0]` is the red channel, `[..., 1]` green, `[..., 2]` blue. One pixel is a bundle of three numbers (R, G, B): (1, 0, 0) is pure red, (1, 1, 0) is yellow, (1, 1, 1) is white. Grayscale conversion is a weighted average of the three channels (the human eye is most sensitive to green, so the conventional weights are `0.299R + 0.587G + 0.114B`).

### 3.2 The four basic pixel operations

| Operation | Formula | Lighting-console analogy | Watch out |
|---|---|---|---|
| Brightness | `img + b` | Master fader | Values above 1.0 must be clipped |
| Contrast | `(img - 0.5) * a + 0.5` | Dimmer gain | Spreads around the 0.5 midpoint |
| Invert | `1 - img` | Negative film | — |
| Clip | `np.clip(img, 0, 1)` | Safety limiter | Always check after an operation |

Why contrast subtracts `0.5` first: plain `img * 2` only makes everything brighter. You have to spread values up and down around middle gray (0.5) to get "dark goes darker, bright goes brighter".

### 3.3 Geometric operations — crop and resize

- **Crop**: cutting out just the region of interest. It is numpy slicing itself — `img[y0:y1, x0:x1]`. The same as copying a sub-range of a table in Excel.
- **Resize**: changing the size. The simplest method is **nearest neighbor** — each new coordinate takes the value of the closest original pixel. Enlarge an image this way and pixels look like jagged stairs; that is the method's honest limitation (production libraries use interpolation, blending neighboring pixels).

### 3.4 Normalization — the last gate before the model

Deep learning models are sensitive to the range of their input numbers. So before training, images are brought to "around mean 0, similar scale". The two most common recipes are dividing 0–255 down to 0–1, and `(img - mean) / std`. "If you can't standardize the shooting conditions, standardize in code" — that is what normalization means in practice.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

This program builds an RGB image with numpy alone and summarizes every image operation in a single comparison-board PNG.

- **[1]** **Draws** a 48×48 RGB image directly with numpy (a sky gradient + a red rectangle + a yellow circle). First proof that a color picture emerges from nothing but array manipulation — no downloads, no image files.
- **[2]** Splits the R/G/B channels into single sheets and prints which objects look bright in each channel. The red rectangle is bright in the R channel and dark in B — the identity of "being red" revealed as numbers.
- **[3]** Applies brightness (+0.25), contrast (×1.8), inversion, and grayscale conversion, printing before/after pixel statistics (min/mean/max) as a table. It deliberately shows values exceeding 1.0 when you skip the clip.
- **[4]** Implements crop (just the rectangle area) and nearest-neighbor resize (48→96 upscale, 48→16 downscale) by hand. Check the staircase artifacts in the upscaled PNG.
- **[5]** Saves the original, the 3 channels, and every operation result as a 3×4 comparison grid to `outputs/pixel_ops.png`.

The heart of the code is `resize_nearest()`. It uses `np.linspace` to build index arrays saying "which original coordinate each new pixel should reference", and finishes the resize with one line of fancy indexing, `img[yy][:, xx]`. Geometric operations done purely through array indexing, no loops — that is the numpy way of thinking.

## 5. Try It Yourself

1. **(Easy)** Change the brightness adjustment in `[3]` from `+0.25` to `-0.25` to make a darker version. Check the PNG — the scene looks like it slid into shadow. (Hint: the `BRIGHT_DELTA` constant)
2. **(Medium)** Build a "sepia tone" filter: compute the grayscale value `g`, then reassemble channels as R=g×1.0, G=g×0.8, B=g×0.55. (Hint: `np.stack([...], axis=-1)` then clip)
3. **(Challenge)** Replace the 0.5 center in the contrast formula with **the image's actual mean**. Compare, with statistics, how the two versions differ on a dark image. (Hint: use `img.mean()` as the pivot)

## 6. Common Mistakes

- **Forgetting to clip**: after `img + 0.5`, a value of 1.4 raises no numpy error. You only discover it when colors glitch at save/display time. Make `np.clip` a reflex after pixel arithmetic.
- **Overflow in integer images**: in a 0–255 uint8 array, 200+100 gives 44, not 300 (wraparound). Convert to float, compute, then convert back — that is the safe route.
- **Confusing channel order**: this lecture uses RGB order, but some libraries (OpenCV among them) use BGR. Ninety percent of blue-tinted face photos are a channel-order bug.
- **Believing resize restores information**: scaling 16×16 up to 96×96 adds no information. It only looks bigger. Detail that isn't there cannot be created.

## Next Level Preview

Every operation so far changed each pixel **independently**. But the question "is this the edge of an object?" can only be answered by looking at the **relationship** with neighboring pixels. The operation that looks at neighbors together — convolution — is the star of level02.
