# Lecture 09 · Level 06 — Data Augmentation

> Inflate the dataset with label-preserving transforms (rotation, shift, flip, noise) and run an experiment that lifts accuracy by 30 points without shooting a single new photo.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level05 / **Estimated time** 40 min

## 1. Why This Matters — the Business View

The bottleneck of a vision project is almost never the model — it is **labeled data**. Collecting defect images requires actual defects to occur; labeling eats a veteran inspector's time; privacy and security concerns can block collection altogether. "Go gather ten thousand images" being impossible is the default situation.

Data augmentation is the cheapest prescription for that bottleneck: **add slightly transformed copies of the images you already have** to the training set — no extra photography, no extra labeling. In production image-training pipelines, augmentation is standard equipment, not an option, and "what augmentation are you using?" is practically how vision engineers say hello. Today we learn the principle and its limits by experiment.

## 2. An Analogy

Think of **product photography for a new listing**. When you post an item to an online store, you don't shoot one angle. Front, slightly left, flipped over, different lighting — the same product under many conditions. The product (its identity) is unchanged; only its appearance varies.

Augmentation **imitates this multi-angle shoot in code**. Rotate, shift, flip, and add noise to the photos you already have — and they become photos that look re-shot under different conditions. In the new-inspector analogy, it is **a workbook of variant problems, the same question with the numbers changed**. An inspector who saw one circle in one position is weak on circles elsewhere; one who saw circles warped into every position and orientation learns that "roundness, not position, is the essence".

But there is one absolute rule: **the transform must not change the answer.** A circle rotated is still a circle, but a digit 6 rotated 180 degrees becomes a 9. Flip an arrow sign left-right and its direction changes. Which transforms are safe is decided by **the meaning of the task** — that is the whole of augmentation design.

## 3. Core Concepts

### 3.1 Label-preserving transforms

The formal definition of augmentation: transforms that change the input while preserving the true label. All four we implement today are safe for shape classification.

| Transform | Implementation | Reality it imitates |
|---|---|---|
| Rotation (multiples of 90°) | `np.rot90` | Product placed rotated |
| Horizontal flip | `img[:, ::-1]` | Camera mounted on the opposite side |
| Shift (±2px) | `np.roll` + zeroing the edges | Position jitter on a conveyor |
| Gaussian noise | `img + normal(0, 0.1)` | Sensor noise, lighting flicker |

### 3.2 Why it works — the relationship with overfitting

With only 120 images, the model memorizes even "the accidental quirks of these 120" (like the noise blotch in image #3's top-left) — overfitting (lecture08 level09). Mix in augmented copies and the accidental quirks differ per transform, so they can't be memorized; only what survives the transforms — the essence, the shape's form — gets learned. So augmentation is also **"regularization that keeps only the essence"**. The translation robustness that pooling provides structurally in level04, augmentation teaches once more through data.

### 3.3 Absolutely forbidden — augmenting evaluation data, and augmenting before the split

- Never put augmented copies in test/validation data. The exam paper must be reality as-is.
- **Never augment before splitting.** If an original lands in training and its rotated copy in test, the exam effectively contained the same question — performance is inflated (data leakage). The order is always "split first, augment only the training side".

### 3.4 How much, and how strong?

Augmentation too can be overdone. Raise the noise std to 0.5 and the shape itself drowns; push the shift to ±6px and the shape leaves the frame. The practical rule of thumb: transform only **within the range where a human could still get the answer right**. The strengths in main.py (rotation in 90° steps, shift ±2, noise 0.1) were chosen by that criterion.

### 3.5 How it's done these days

Production frameworks apply augmentation **randomly at every epoch, inside the training loop** (on-the-fly). That yields more diversity than today's pre-inflate approach. The principle is identical, so understanding today's implementation lets you read torchvision.transforms-style configuration.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Deliberately creates a data-scarce situation: 120 training images, 300 test.
- **[2]** Implements the 4 augmentations in numpy and builds transformed copies of one image.
- **[3]** Creates 4 random augmented copies per image, inflating training data 120→600. The test set stays original.
- **[4]** **A fair comparison experiment**: same model architecture, same epochs, same initial weights, "no augmentation vs with augmentation", compared on test accuracy. Typical result: 0.59 → 0.91 (**+32 points**). A gain bought with zero new data.
- **[5]** Saves a gallery of transformed copies plus the performance bars to `outputs/augmentation.png`.

The heart of the code is `augment_dataset()`. Fifteen lines that combine rotation→flip→shift→noise at random to create copies — a miniature of a production augmentation pipeline. Also note the identical `torch.manual_seed(6)` call right before both experiments, **unifying the initial weights**. The fairness of a comparison lives in details like this.

## 5. Try It Yourself

1. **(Easy)** Raise the noise strength `std` from 0.10 → 0.35 and rerun the experiment. Find the point where the augmentation benefit shrinks (or flips). You will feel what "a range where a human could still answer" means. (Hint: the default of `aug_noise`)
2. **(Medium)** Sweep `per_image` through 1, 2, 4, 8 and tabulate test accuracy. You'll see the marginal utility of the augmentation multiplier diminish. (Hint: `augment_dataset(..., per_image=...)`)
3. **(Challenge)** Imagine "a task where a 180° rotation changes the label", and modify `augment_dataset` to use only 90/270° rotations, excluding 180°. This is practice for reviewing an augmentation list on a real task. (Hint: give `rng.integers` a list of candidates instead)

## 6. Common Mistakes

- **Using transforms that change the label**: casually rotating/flipping characters, digits, or directional objects changes the answer. The augmentation list must be reviewed by someone who knows the task's meaning.
- **Augmenting before the split**: originals and copies split across train/test — data leakage. A classic cause of performance collapsing after deployment.
- **Judging the effect by training performance instead of test**: augmentation makes training "harder", so training loss actually looks worse. Measure the effect only on an untouched test set.
- **Treating augmentation as a cure-all**: augmentation is variation on data you have; it cannot create types you lack (a defect category never photographed). It supplements collection; it does not replace it.

## Next Level Preview

If augmentation was the lever that "multiplies my data", there is a bigger lever — **borrowing what someone else (or another task) has already learned**: transfer learning. In level07 we freeze the front of a CNN trained on task A, transplant it to task B, and prove by experiment that it is faster and better than training from scratch.
