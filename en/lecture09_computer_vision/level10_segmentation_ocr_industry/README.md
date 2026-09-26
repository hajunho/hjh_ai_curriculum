# Lecture 09 · Level 10 — Segmentation, OCR, and Industrial Inspection

> We implement segmentation — assigning every pixel its allegiance — using nothing but a threshold and connected components, and run it all the way to defect-stain detection. And we confirm that OCR and industrial inspection share the same skeleton.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level09 / **Estimated time** 45 min

## 1. Why This Matters — the Business View

A bounding box (level09) only answers "roughly where". But shop-floor questions are often more precise — "what is the **area** of the scratch in mm²?" (defect grades split on area), "where exactly is the tumor's **boundary**?", "**separate** the text regions from the stamp regions in this document". The moment a pixel-level answer is required, the technology you need is **segmentation**.

One more piece of practical sense: not every problem needs deep learning. In environments with controlled conditions — like an inspection line with fixed lighting — today's **classic trio (threshold → connected components → area rule)** has been processing thousands of frames per second without deep learning, on active duty for decades. Today you gain the criterion for answering "can this task be done with rule-based vision, or does it need deep learning?"

## 2. An Analogy

**Coloring books and the paint bucket.**

- **Thresholding**: in a grayscale sketch, "keep only the parts darker than some level". The simplest pixel classification, splitting foreground (object) from background with a single criterion. With consistent lighting, a brightness cutoff works astonishingly well.
- **Connected components**: the **paint-bucket tool** in a drawing app. Click one pixel and the color floods every touching pixel of the same color; likewise, starting from a foreground pixel we spread up/down/left/right, giving one "blob" the same number. Count the blobs and you have the object count; count each blob's pixels and you have its area.
- **The area rule**: classify blobs by size. "A part is 100+ pixels, a stain is under 12" — the inspector's instinct "that size is spatter" translated into numbers.

OCR (optical character recognition) shares this skeleton: binarize the document → separate the character blobs (segmentation) → classify each blob as "which character" (recognition). The front stages are today's techniques; the back stage is level05's classifier.

## 3. Core Concepts

### 3.1 The three tiers of segmentation

| Tier | Question | Example |
|---|---|---|
| Binary (foreground/background) | Is this an object pixel? | Today's threshold method |
| Semantic | What class is this pixel? | Road/car/person pixels |
| Instance | Same class, but which individual? | Person 1 vs person 2 |

Today's "binarize + connected components" is effectively mini instance segmentation — you get both pixel allegiance (binarization) and individual identity (component numbers). Deep-learning segmentation (U-Net-style encoder–decoder architectures) replaces the threshold with **a trained CNN emitting per-pixel class probabilities**, but the problem statement — "decide each pixel's allegiance" — is identical.

### 3.2 When thresholding works, and when it collapses

Thresholding works well when the brightness histogram splits into **two peaks (background/foreground)**. It collapses under the opposite: uneven lighting (one side in shadow) or overlapping foreground/background brightness. The practical response, in order: ① fix the lighting (hardware is the best fix), ② use region-adaptive thresholds, ③ if that still fails, go learning-based. This escalation order is the canon of industrial-vision consulting.

### 3.3 Connected components = graph traversal

"Touching pixels form one blob" is a graph problem. Pixels are nodes, 4-neighbor adjacency the edges, and a blob is a connected subgraph. main.py implements BFS (breadth-first search) with a stack — paint the start pixel, spread to unpainted foreground neighbors, 15 lines. Whether diagonals count as neighbors (8-connectivity) or not (4-connectivity) is a design choice set by the task.

### 3.4 Region properties — measurement is the value

Once labeling is done, measure each region's **area, bounding box, and centroid**. These numbers are the business answers: area → defect grade, centroid → robot pick-up coordinates, count → inventory tally. This transformation ending in "image → table of numbers" is the final deliverable of industrial vision.

### 3.5 A map of industrial applications

- **Surface inspection**: scratch/stain detection on metal, film, displays — today's demo, verbatim.
- **Document processing (OCR)**: binarize → separate lines/characters → classify characters → assemble text. Deep-learning OCR that merges separation and recognition dominates lately, but preprocessing binarization still decides quality.
- **Medical and bio**: cell counting (component counts), lesion area measurement.
- **Robotics and logistics**: computing the pixel region and centroid of the object to pick.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Draws an inspection scene directly with numpy: 3 good parts (square/circle/triangle) + 4 small stains scattered in secret + shooting noise. The detector does not know where the stains are.
- **[2]** Binarizes with threshold 0.45. The noise (std 0.06) cannot clear the bar and falls cleanly into background.
- **[3]** Finds 7 blobs with BFS connected-component labeling.
- **[4]** Prints each region's area and centroid as a table and rules "area under 12 = stain". Checked against the answer key: stains 4/4 detected.
- **[5]** Saves a 4-panel comparison — original → binarized → colored label map → verdict overlay (green=part, red=defect) — to `outputs/segmentation.png`.

The heart of the code is `connected_components()` — 15 lines of stack-based BFS. Confirm that an industrial inspection algorithm can be built with nothing beyond lecture02's loops and lists.

## 5. Try It Yourself

1. **(Easy)** Raise `THRESH` from 0.45 → 0.80. The stains (brightness 0.85) survive but the part boundaries waver; at 0.9 the stains vanish too. Confirm that the threshold sets the detection sensitivity. (Hint: one constant)
2. **(Medium)** Switch 4-connectivity to 8-connectivity (diagonals included). What arrangement would make the component count differ? (Hint: add the 4 diagonal directions to the neighbor list; two pixels touching only diagonally)
3. **(Challenge)** Raise the noise std from 0.06 → 0.20 and salt-grain fake foreground appears in the binarization. Add a cleanup stage — "ignore components of area ≤ 2 as noise" — to restore detection. (Hint: filter components before the verdict — this is the spirit of morphological operations)

## 6. Common Mistakes

- **Fixing one threshold for all images**: under changing lighting, a fixed threshold always collapses. Look at the histogram first; go adaptive if needed.
- **Ignoring connectivity (4 vs 8)**: a thin diagonal line breaks into fragments under 4-connectivity. Fatal in counting tasks.
- **Confusing units in the area rule**: pixel area depends on resolution. Change the camera or distance and the "under 12 pixels" rule must be recalibrated.
- **Reaching for deep learning first**: using a learning-based approach on a task classic methods handle under controlled lighting only inflates data-collection and labeling costs. Remember the escalation order (3.2).

## Next Level Preview

The CNN read images "neighbor by neighbor, step by step". But the newest vision AI reads differently — treating an image as a sentence of patch "words", deciding for itself which patch should attend to which: the **Vision Transformer (ViT)**, and the **multimodal (CLIP-style)** approach that lets images and text meet in one shared space. In the final level we implement patch embedding and self-attention ourselves and close out the lecture.
