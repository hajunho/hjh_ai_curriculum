# Lecture 09 · Level 05 — Hands-On: Classifying Shape Images

> From data splitting through training, evaluation, and misclassification analysis — the standard cycle of an image-classification project, completed with a CNN.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03–04, lecture08 level08 (training loops) / **Estimated time** 50 min

## 1. Why This Matters — the Business View

This level is the lecture's first "full run". Having learned the parts one by one — convolution, pooling, Linear — today we assemble everything in the same order a real project follows: **data split → model definition → training → validation → test → misclassification analysis**.

Someone who has run this cycle once with their own hands listens differently when a vendor reports "accuracy is 96.7%". They immediately ask: "Measured on which data? Not the training data, right? What kinds of cases make up the wrong 3.3%?" Those three questions are the core of vision-project acceptance review, and today's exercise gives you the grounds to ask them.

As a bonus, we settle an old argument by experiment: "Why a CNN for images at all? Can't we just feed all the pixels to a plain neural network (MLP)?" — we pit the two against each other on the same data and settings, and answer with numbers.

## 2. An Analogy

Read the whole cycle as **training a new quality inspector**.

- **Data split**: prepare a training workbook (600 images), a mock exam (validation, 150), and a sealed final exam (test, 150) in advance. Use final-exam questions in the training and the score becomes a lie — the very principle from lecture06 level04, unchanged for images.
- **Training loop**: work through the workbook multiple rounds (epochs). After each round, record the mock-exam score and watch whether skill is improving.
- **Evaluation**: at the very end, break the seal exactly once and sit the final exam.
- **Misclassification analysis**: don't stop at the score — **lay out the missed questions and see why**. If the inspector is confused by a particular type (very noisy images? shapes clipped at the frame edge?), that points the direction of the next round of training (data reinforcement).

## 3. Core Concepts

### 3.1 Today's model — ShapeCNN

```
input (1,16,16)
Conv(1→8, 3x3, pad1) + ReLU + MaxPool2 → (8,8,8)
Conv(8→16, 3x3, pad1) + ReLU + MaxPool2 → (16,4,4)
Flatten → Linear(256→3)
```

Following level03's design rhythm (channels up, space down), this is a micro CNN with about 2,000 total parameters. Small task, small model — a model bigger than necessary is cost, and an overfitting risk.

### 3.2 The four elements of training (lecture08 review, applied to images)

- **Loss function**: the multiclass standard, cross-entropy (CrossEntropyLoss). "The lower the probability you gave the true class, the bigger the penalty."
- **Optimizer**: Adam. Nudges the kernel numbers in the direction that shrinks the penalty.
- **Mini-batches**: 600 images processed 64 at a time. Saves memory, stabilizes training.
- **Epochs**: how many passes over the whole dataset. Today, 18 — a value chosen by watching the validation-accuracy curve.

### 3.3 The confusion matrix — not "what score" but "what gets mistaken for what"

A single accuracy number hides the direction of the mistakes. The confusion matrix tallies rows=actual, columns=predicted, showing even "how many circles were seen as squares". The typical pattern in this data is circles and squares blurring into each other when noise smears the outlines. In defect inspection, the "defect seen as normal" cell is the fatal one — build the habit of reading the matrix while marking which cell is the expensive mistake (this ties to the recall discussion in lecture06 level06).

### 3.4 CNN vs MLP — architecture is prior knowledge

The MLP receives the 256 pixels flattened into one row. At that instant, the spatial information — "which pixel is next to which" — is gone. Move the shape two cells and the MLP sees an entirely different input. The CNN has locality (kernels) and translation invariance (weight sharing, pooling) **built into its structure**, so on the same data it generalizes better with fewer parameters. In the run output, watch the CNN (~2K parameters) beat the MLP (~19K parameters) on accuracy — a good architecture is like free data.

### 3.5 Misclassification analysis — the story behind the report card

Look at the missed images directly and most of them make sense: unusually heavy noise, shapes clipped at the frame edge, or shapes so small that even a human eye struggles to tell circle from square. **The model's list of weaknesses is the work order for the next improvement** — a perspective we expand fully in level08 (the real-world pipeline).

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Splits `shape_images(n=900)` into train 600 / validation 150 / test 150. The test set stays sealed until the end.
- **[2]** Trains ShapeCNN for 18 epochs. Every 3 epochs it prints training loss and validation accuracy — watch the rhythm of loss falling and accuracy rising.
- **[3]** Trains an MLP on the same data and settings.
- **[4]** Breaks the seal: final report card for both models on the 150 test images, plus the CNN's confusion matrix. (Typical result: CNN ≈ 0.97, MLP ≈ 0.85)
- **[5]** Saves up to 8 test images the CNN got wrong, with actual/predicted labels, to `outputs/shape_classification.png`. The learning curves (CNN vs MLP) share the same figure on the left.

The heart of the code is the `train()` function — the standard five-beat training loop, `zero_grad → forward → loss → backward → step`. Today's hidden lesson: the loop you learned in lecture08 applies to images without changing a single character.

## 5. Try It Yourself

1. **(Easy)** Cut EPOCHS to 5. How far does validation accuracy climb before stopping? Conversely, at 40 does test accuracy keep improving? Judge from the learning-curve PNG. (Hint: at some point the curve flattens)
2. **(Medium)** Shrink the training data from 600 to 150 images (`xtr, ytr = Xt[:150], yt[:150]`) and observe how the CNN–MLP gap changes. Explain with section 3.4 why the power of architecture grows as data shrinks. (Hint: prior knowledge compensates for missing data)
3. **(Challenge)** Add code that computes per-class recall from the confusion matrix. Which shape has the lowest recall, and does it match the cases in the misclassification PNG? (Hint: recall = diagonal value / row sum)

## 6. Common Mistakes

- **Choosing epochs or architecture using the test set**: start fixing the model based on test scores and the test set is contaminated into a validation set. Tune only on validation; the test is one final shot.
- **Stopping at accuracy**: only the confusion matrix and misclassified cases tell you "what kind of mistakes this model makes". In practice, what's expensive is not the average but a specific cell.
- **Believing that falling loss always means improvement**: if training loss keeps dropping while validation accuracy stalls or falls, that is an overfitting signal (lecture08 level09).
- **Being startled that each run differs slightly**: even with a fixed seed, decimals can wobble with library versions and the like. Read the conclusion as a "trend", not "one number".

## Next Level Preview

You want more performance but cannot draw more data? The technique of multiplying what you have through rotation, shifting, and noise — data augmentation — is implemented in level06, where we measure the with/without difference experimentally.
