# Lecture 09 · Level 02 — Understanding Convolution

> We implement convolution — the heart of the CNN — from scratch in numpy, and watch the moment a 3×3 kernel finds the edges of a shape.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level01 / **Estimated time** 40 min

## 1. Why This Matters — the Business View

Every technical document, proposal, or paper summary about image AI uses the words "convolution", "kernel", and "filter". Leave these words as a black box and the claim "the CNN extracts features" sounds like an incantation — a story you cannot verify.

The reality of convolution is surprisingly small: **slide a small board of numbers (the kernel) across the image, multiplying overlapping cells and adding them up.** This one operation powers classic image processing — blur, sharpen, edge detection — and the feature extraction of modern CNNs alike. Write it yourself with loops today, and from now on any CNN architecture you meet reads as "those multiply-adds, stacked in layers".

## 2. An Analogy

Imagine **reviewing a document with a rubber stamp**. You have a transparent 3×3 inspection stamp, with a weight number written in each cell. You press this stamp across a large document (the image), one cell at a time, from top-left to bottom-right. Each press does exactly one thing — **multiply each of the 9 stamp numbers by the document number underneath, sum them all**, and write the total at that position on a new sheet.

The key is **which numbers are engraved on the stamp**.

- A stamp where every cell is 1/9 → the average of the surrounding 9 cells = **blur**
- A stamp with -1 in the left column and +1 in the right → a test asking "is the right side brighter than the left?" = **vertical edge detection**

So the kernel is **an inspection stamp engraved with "what to look for in this region"**, and the output (the feature map) is **the inspection-results map** — a table of "where, and how strongly, the pattern appeared".

## 3. Core Concepts

### 3.1 The definition of the operation

For an input image `I` and a `k×k` kernel `K`, the output value at (y, x) is the kernel overlaid at that position, with the element-wise products all summed.

```
out[y, x] = Σ_i Σ_j  I[y+i, x+j] * K[i, j]
```

In code: two loops and one line, `(patch * K).sum()`. The `conv2d()` in main.py is exactly this shape.

### 3.2 How to read an edge-detection kernel

Let's read the vertical-edge kernel (Sobel family).

```
-1  0  +1
-2  0  +2
-1  0  +1
```

This kernel's output = (brightness sum of the 3 right cells) − (brightness sum of the 3 left cells). Equal brightness left and right gives 0; a **vertical boundary that is dark on the left and bright on the right** gives a large positive value; the opposite direction gives a large negative one. Rotate it 90 degrees and you get the horizontal-edge kernel. This is where the sentence "a kernel = a pattern detector with 'what to respond to' written on it" comes from.

### 3.3 Output size, padding, stride

- The stamp cannot go off the edge of the document, so pressing a `k×k` kernel on an `n×n` image **shrinks** the output to `(n-k+1)×(n-k+1)`. 16×16 with a 3×3 gives 14×14.
- To keep the size, wrap the edges in a border of zeros before stamping — **padding**.
- Press every two cells instead of every one and the output halves in size — **stride**. It saves computation and acts as summarization. (Formulas and layer design come in level03.)

### 3.4 Why convolution, for images?

It locks precisely into two properties of images.

1. **Locality**: "is this a boundary?" can be judged from just a few surrounding pixels. Many questions need no more than a 3×3 kernel.
2. **Translation invariance**: a vertical boundary at the top-left and one at the bottom-right are the same pattern. Convolution reuses one kernel across the whole image — "running the same inspection at every position" — so nothing needs to be learned separately per location. This is why CNNs need dramatically fewer parameters than MLPs (we confirm it with real performance in level05).

### 3.5 Kernels once carved by hand, now learned

In classic image processing, experts **designed kernels by hand** — Sobel, Laplacian, and friends. The CNN's single innovation: **let the data decide the numbers inside the kernel**. It is lecture06 level00's "humans fix only the shape of the rule; the data fixes the numbers", replayed in the world of images. Today we verify the principle with hand-carved kernels; from level03 we move to learned ones.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Traces one convolution **number by number** on a 6×6 mini image with a 3×3 kernel. At the first position it prints all 9 products and their sum, so you can verify the entire operation by eye.
- **[2]** Implements `conv2d()` with loops, and confirms the output shrinking to (n-k+1) and the size being preserved with zero padding.
- **[3]** Applies vertical/horizontal edge kernels to the squares, circles, and triangles from `shape_images()`. On the square, only the left/right sides stay bright under the vertical kernel, and only the top/bottom sides under the horizontal one — confirm that "each kernel asks a different question".
- **[4]** Also applies a blur (mean) kernel and a sharpen kernel to the same image, showing that changing only the kernel's numbers produces completely different effects.
- **[5]** Saves a grid of 3 shapes × (original, vertical edges, horizontal edges, edge magnitude) to `outputs/convolution_edges.png`. Edge magnitude is `sqrt(vertical² + horizontal²)` — a direction-agnostic "here is a boundary" map.

The heart of the code is `conv2d()`, just 8 lines. In every later level those 8 lines will be abbreviated into one line, `nn.Conv2d` — but the work is identical.

## 5. Try It Yourself

1. **(Easy)** Flip all the signs of the vertical-edge kernel in `[3]` and run it. The bright and dark areas of the feature map swap. Explain why the absolute value (edge magnitude) is unaffected. (Hint: the symmetry of (+) and (−))
2. **(Medium)** Design a diagonal-edge kernel of your own. You have succeeded when it responds most strongly on the triangle's hypotenuse. (Hint: start from a shape like `[[0,1,2],[-1,0,1],[-2,-1,0]]`)
3. **(Challenge)** Add a `stride` argument to `conv2d()` and process the shapes with stride=2. Confirm in the PNG that even at half the output size, the rough positions of the edges survive. (Hint: `range(0, H-k+1, stride)`)

## 6. Common Mistakes

- **Mistaking multiply-then-add for an "average"**: convolution is a sum. The mean kernel is implemented by making the kernel values themselves 1/9 — the operation itself is not an average.
- **Panicking at negative outputs**: negative values from an edge kernel are normal (they carry direction). imshow them raw and the image may look pitch black — view the absolute value or use a symmetric colormap.
- **Forgetting the output shrinks**: stack layers and a 3×3 kernel nibbles 2 pixels per layer until the image vanishes. This is exactly why padding exists.
- **Assuming bigger kernels are better**: two 3×3 layers cover the same field of view as one 7×7 with fewer parameters. That is why modern CNNs standardized on 3×3.

## Next Level Preview

We found edges with hand-carved kernels. Next, we let training decide the kernel numbers, and stack such convolution layers in multiple sheets and multiple tiers. Conv2d's channel/stride/padding design and parameter counting — we assemble a CNN in level03.
