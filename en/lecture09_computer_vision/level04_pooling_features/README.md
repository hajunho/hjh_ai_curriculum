# Lecture 09 · Level 04 — Pooling and the Hierarchy of Features

> A summarizing operation that keeps "roughly where it was" and throws away the exact position — we use feature maps to see why pooling makes CNNs robust.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03 / **Estimated time** 35 min

## 1. Why This Matters — the Business View

In front of a defect-inspection camera, each product lands 1–2 mm off from the last. Scanned documents tilt slightly; products in shelf photos sit in slightly different spots. A system that is sensitive to exact pixel coordinates would misfire at every one of these "slight misalignments". The essential virtue of a real-world vision system is **making the same judgment even when positions shift a little**, and the CNN component in charge of that virtue is pooling.

One more thing. When you understand what "deeper is better" actually means — **early layers see low-level features like dots and lines; later layers see high-level features like parts and shapes** — you develop an instinct for "how many layers does our task need?". Detecting fine scratches leans on early-layer features; classifying product types leans on late-layer ones.

## 2. An Analogy

Think of **regional roll-up reporting**. Headquarters cannot review the daily metrics of all 200 stores, so stores are grouped by region and each region reports **one representative value**: "top store in the metro region: 52M". The detailed ranking of individual stores disappears, but the essential fact — "somewhere in the metro region there's a blockbuster store" — survives.

**Max pooling** is exactly this. Divide the feature map into 2×2 zones and keep only **the maximum** in each zone. The information "somewhere in this zone, a strong vertical edge was detected" is preserved; "which exact pixel" is discarded. So even if an object shifts by a pixel or two, the pooled result barely changes — the roll-up report doesn't change when a store moves one block over.

The hierarchy of features works like a **corporate reporting chain**. Front-line staff (layer 1) look at raw detail (pixel-level light/dark changes = edges); team leads (layer 2) look at combinations of staff reports (arrangements of edges = corners and curves); executives (layer 3) look at combinations of team-lead reports (layouts of parts = "an object with a rounded outline"). The higher you go, the more **abstract, and the wider the territory each report represents**.

## 3. Core Concepts

### 3.1 The max-pooling operation

`MaxPool2d(kernel_size=2, stride=2)`: slide a non-overlapping 2×2 window and take the maximum inside each window. Output: height and width halved, channel count unchanged. The important part: **zero parameters** — it is a fixed summarization rule that learns nothing.

Average pooling takes the mean instead of the max. The difference is "strongest evidence" (max) vs "overall tendency" (mean); classification CNNs typically use max in intermediate layers and mean for the final summary (global average pooling).

### 3.2 The three effects of pooling

1. **Translation robustness**: shift the input 1–2 pixels and the zone maxima barely change. We verify this experimentally in main.py.
2. **Compute and size savings**: half the height and width = 1/4 the compute at the next layer. The Flatten size shrinks too, slashing the trailing Linear parameters.
3. **Wider receptive field**: one pixel after pooling represents a wider area of the original, so the same 3×3 kernel in a later layer sees larger patterns.

### 3.3 Strided convolution vs pooling

TinyCNN in level03 shrank sizes with stride=2 convolutions. Pooling and strided convolution are both "spatial downsizing" tools: the former is a fixed rule with 0 parameters; the latter is a learned reduction. Classic architectures (e.g. VGG-style design conventions) favor `Conv-Conv-MaxPool` blocks; more recent ones favor strided convolutions. You just need to be able to read both.

### 3.4 The feature hierarchy — why does it arise?

No designer ever instructed "layer 1, look at edges". The structure makes it so. A layer-1 kernel can only see a 3×3 patch of raw pixels, so it cannot see anything beyond light/dark changes (edges); layer 2 sees a 3×3 of layer-1 outputs — i.e., combinations of edges over a wider original area — and pooling widens that field of view geometrically. **The low-to-high hierarchy is the inevitable consequence of locality + stacking + pooling.**

### 3.5 Something is lost, too

Pooling discards position information. That is an advantage for classification ("what is it?"), but in tasks where exact position matters — object detection (level09), segmentation (level10) — this loss becomes a problem, and separate machinery is needed to recover position. Remember the limitation: "the roll-up report doesn't tell you the store's address".

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Verifies 2×2 max pooling number by number on a 4×4 mini example.
- **[2]** Applies the edge kernel (reused from level02) to a shape image, builds the feature map, and compares before pooling (16×16) and after (8×8). It's blurrier, but "roughly where the edges are" survives.
- **[3]** **Translation-robustness experiment**: how similar (cosine similarity) is the feature map of a shape shifted right by 1–3 pixels to the original's — compared before pooling, after 1 pooling, after 2. At a 2-pixel shift the pre-pooling similarity is 0.00 — to pixel eyes a completely different map — while after 2 poolings it is 0.71, close to "the same object".
- **[4]** Visualizes the feature maps of a mini CNN's conv1 (edge level) and conv2 (combination level) side by side, showing responses becoming chunkier and more abstract with depth.
- **[5]** Saves everything above to `outputs/pooling_features.png`.

The heart of the code is `maxpool2x2()` — pooling implemented in one reshape trick (`reshape(h//2, 2, w//2, 2).max(axis=(1,3))`). Notice it has the same skeleton as level00's resolution downscaling (mean), with only max swapped in.

## 5. Try It Yourself

1. **(Easy)** Change the max in `maxpool2x2()` to mean to build average pooling. In the PNG, confirm the difference: max is "preserve strong evidence", mean is "soft summary". (Hint: `.max(axis=(1,3))` → `.mean(axis=(1,3))`)
2. **(Medium)** Add 4 and 5 pixels to the shift list in `[3]`. At what shift does the protection of double pooling collapse? Explain it by connecting to how many original pixels one cell represents after two 2×2 poolings. (Hint: 2×2 twice = a 4×4 zone)
3. **(Challenge)** Apply pooling twice in a row (16→8→4) and check in the PNG whether a human eye can still tell the shape classes apart. You'll develop a feel for "the minimum resolution classification needs". (Hint: compare with level00's 4×4 experiment)

## 6. Common Mistakes

- **Thinking pooling has parameters**: MaxPool learns nothing; it is a fixed rule. Count it as 0 in model-size estimates.
- **Assuming pooling also applies across channels**: pooling shrinks only space (H, W). Channel count is untouched.
- **Cramming in pooling everywhere**: pool at every layer and position information vanishes too fast — small objects and fine defects become invisible. Downsize only as much as the task allows.
- **The misconception that "later layers are smarter"**: later layers are merely more abstract. Plenty of tasks — like fine-scratch detection — are answered by low-level features. Matching the task to the layer level is what design means.

## Next Level Preview

All the parts are ready — convolution (the question), ReLU (nonlinearity), pooling (summary), Linear (the verdict). In level05 we finally assemble them and **train** a shape classifier. Training → evaluation → misclassification analysis: the full image-classification cycle, completed.
