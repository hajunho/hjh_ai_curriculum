# Lecture 09 · Level 09 — How Object Detection Works

> Beyond "what is there" to "what is where" — we build a mini detector out of nothing but a sliding window and a classifier, and learn detection's grammar through IoU and NMS.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level05, level08 / **Estimated time** 50 min

## 1. Why This Matters — the Business View

Classification answers up to "is there a defect in this photo?". But the question on the floor goes one step further — "**how many** defects, and **where**?" Finding out-of-stock gaps in a shelf photo, locating stamps and signatures on documents, tracking people and vehicles on CCTV — all of these are **object detection** problems.

Reviewing a detection solution, you will inevitably meet three terms: the **bounding box** (the rectangle enclosing an object), **IoU** (how much a predicted box overlaps the truth), and **NMS** (cleaning up duplicate boxes). Plus mAP on the performance sheet. Without these terms you cannot read a performance report that says "mAP@0.5 is 0.91". Today we build every one of these parts by hand, and confirm that detection began as **"running a classifier repeatedly at every position"**.

## 2. An Analogy

**Surveying a mural with a magnifying glass.** You must find every instance of a particular picture (shape) in a large mural (the scene) and report its location. The method is stubbornly simple — **slide the magnifying glass (a window) from top-left to bottom-right, one hand-width at a time, and at every position judge "what do I see here?"** That is the sliding window. The judge is exactly the classifier you built in level05. With one addition — it **must know how to answer "nothing here (background)"**, because at most positions on the mural there is no picture at all.

When the survey ends, the report lists the same picture many times — slide the glass 2 cm and the same picture is still in view. So you need a cleanup rule: **"among near-duplicate reports of the same location, keep only the most confident one"** — that is NMS (non-maximum suppression).

Finally, grading. How do you define "got the location right"? Demand pixel-perfect boxes and nobody ever scores. So we count a prediction as correct **when the overlap ratio (IoU) meets a threshold**. It is the same arithmetic as measuring "how much do these two land parcels overlap" in real estate.

## 3. Core Concepts

### 3.1 IoU (Intersection over Union)

```
IoU = overlap area / (area A + area B − overlap area)
```

Perfect match 1.0, no overlap 0.0. By convention, **IoU ≥ 0.5 counts as a "correct detection (TP)"**. IoU is not just for grading — it is detection's all-purpose ruler, used in training-data labeling (3.3) and in NMS (3.4) as well.

### 3.2 Detection's three error types, and mAP

- **True positive (TP)**: class matches and IoU ≥ 0.5
- **False positive (FP)**: firing where nothing is, or wrong class/position (the cause of alarm fatigue)
- **False negative (FN)**: it was there and got missed (the cause of defect escapes)

Lower the confidence threshold and misses shrink while false alarms grow — lecture06 level06's precision/recall trade-off reappears in detection. **mAP** (mean Average Precision) is the standard metric that averages this trade-off curve across classes into one number.

### 3.3 IoU builds the training data too

The raw material of detector training is "window crops + labels". Today's rule: if a crop has **IoU ≥ 0.55 with a ground-truth box, it is that shape**; otherwise it is **background**. A crucial detail — windows half-covering a shape are also taught as background. Only then does the classifier fire exclusively on "windows where the object sits properly centered", making the detection box snap to the object's center. Train once without this rule and drown in false positives (Try It Yourself 3), and you will feel why real detectors put such care into these label-assignment rules.

### 3.4 NMS — sort by confidence, remove overlaps

Walk the candidate boxes in descending confidence (score); drop any box whose IoU with an already accepted box exceeds the threshold (0.3 today). Simple, but every modern detector uses it (in some variant) as post-processing.

### 3.5 The sliding window's limits, and modern methods (concepts only)

Today's method has obvious weaknesses: (1) 289 windows each go through the CNN **individually** — slow. (2) The window size is fixed, so objects of other sizes are missed. The evolution of modern detectors is the history of dissolving these two weaknesses.

- **Two-stage family (R-CNN lineage)**: first shortlist "regions likely to contain an object", then finely classify only the shortlist.
- **One-stage family (YOLO, SSD lineage)**: divide the image into a grid and output "class + box coordinates" for every cell in **a single CNN forward pass**. It exploits the fact that convolution already sweeps every position in parallel (level02); real-time capable, it became the industrial standard.

You don't need the names. Remember one thing — **every state-of-the-art detector stands on today's grammar: classification per position + box regression + NMS + IoU grading**.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Verifies IoU by hand and by function (overlap 36 / union 164 = 0.220).
- **[2]** Builds 40 training scenes, sweeps them with the window, auto-labels the crops via the IoU rule (3 shape classes + background), and trains the classifier.
- **[3]** Sweeps a never-seen 48×48 scene (3 shapes placed) with a 16×16 window at stride 2, classifying 289 positions. Dozens of candidates above 0.9 confidence appear — many duplicates per shape.
- **[4]** NMS reduces the candidates to a final 3.
- **[5]** Grades by IoU ≥ 0.5 matching against the ground-truth boxes (typical result: TP 3 / FP 0 / FN 0). Saves ground truth (green) and detections (red) side by side to `outputs/object_detection.png`.

The heart of the code is three functions: `iou()` in 7 lines, the double for-loop of `sliding_window_detect()`, and `nms()` in 5 lines. Detection's entire grammar fits in those ~20 lines.

## 5. Try It Yourself

1. **(Easy)** Lower `SCORE_TH` from 0.90 → 0.50. How do the candidates and false positives change? An experiment confirming the precision/recall trade-off in detection. (Hint: one constant)
2. **(Medium)** Raise `STRIDE` from 2 → 6. How much does the number of swept positions (compute) fall, and do misses appear? Confirm that "stride = the scale balancing speed and accuracy". (Hint: if windows step over a shape's center, you get a miss)
3. **(Challenge)** Change the rule in `build_training_crops` to "exclude half-covering windows (0.2 < IoU < 0.55) from training" (the original first attempt). Watch false positives explode, and explain why with section 3.3. (Hint: on inputs it never learned, a model asserts arbitrary confidence)

## 6. Common Mistakes

- **Evaluating detection with classification accuracy**: detection performance must be spoken in TP/FP/FN and mAP. "The classifier is 97% accurate" guarantees nothing about detection quality.
- **Forgetting NMS**: a detector without post-processing outputs dozens of boxes per object. If the demo is wallpapered with boxes, it is almost certainly missing NMS.
- **Attempting detection without a background class**: a classifier that never learned "nothing here" insists it saw something in every window.
- **Treating FP and FN costs as equal**: in defect detection a miss is fatal; in security alarms false positives neutralize the system. Thresholds are set by business cost (connects to lecture07 level11).

## Next Level Preview

A box only says "roughly where". Move to segmentation — labeling every single pixel "you are object, you are background" — and you can measure area, shape, and boundaries. In level10 we implement segmentation with nothing but thresholds and connected components, and cut through the industrial applications: defect (stain) detection and the principle of OCR.
