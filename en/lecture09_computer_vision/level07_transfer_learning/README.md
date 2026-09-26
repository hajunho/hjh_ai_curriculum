# Lecture 09 · Level 07 — Transfer Learning

> Freeze the "trained eye" learned on task A, transplant it to task B, and prove by experiment that it beats training from scratch with only 24 new labels.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level05–06 / **Estimated time** 45 min

## 1. Why This Matters — the Business View

Nine out of ten real-world image tasks start like this: "We only have a few dozen to a few hundred labeled images of our own." Training from scratch under that condition is usually reckless, and the industry-standard remedy is **transfer learning** — take the front portion of a model already trained on massive data, and re-teach only the back portion for your task. It is why "the quote for adding vision AI came in cheaper than expected" — and, conversely, why the diagnosis "our data is too unusual for transfer to work" can multiply a project's cost. Today you will understand at the level of principle why transfer works, and when it does not.

What makes this level special: transfer-learning tutorials usually download a large pretrained model, but we prove the principle **with zero downloads**. We pretrain on task A (squares/circles/triangles) ourselves, then transplant that eye onto task B (diamonds/ellipses/rings). Only the scale is smaller — the mechanism is identical to practice.

## 2. An Analogy

**Hiring experienced staff vs hiring fresh graduates.**

The quality-inspection team needs someone to inspect a new product (task B). Two candidates.

- **The fresh graduate (from scratch)**: a blank slate. You must teach the fundamentals — "look at object outlines", "compare shapes" — from zero. With only 24 pages of training material (labeled data), even the fundamentals won't stick.
- **The experienced hire (transfer)**: someone who inspected a different product (task A) for years. The new product is unfamiliar, but **the "way of seeing" itself is already there.** You only need to teach the new product's pass/fail criteria, so 24 pages is enough.

In a CNN, the "way of seeing" lives in the front (the feature extractor) and the "judgment criteria" in the back (the head). Recall the feature hierarchy from level04 — the early layers' edge, curve, and blob detectors are **shared fundamentals needed in any image task**. That is why the front is reusable even when the task changes.

## 3. Core Concepts

### 3.1 Two flavors of transfer

| Method | What it does | When |
|---|---|---|
| Feature extraction | **Freeze** the front, train only the head | When new data is very scarce (today's method) |
| Fine-tuning | Also update the front, **at a small learning rate** | When you have some data and the task differs a bit from the original |

Freezing is one line of code: `param.requires_grad = False`. Don't hand the frozen parameters to the optimizer, and those weights stay preserved in their pretrained state.

### 3.2 Why it works — the feature hierarchy, revisited

Task A contains no diamonds and no rings. Yet the eye trained on A works on B, because what the early layers learn is not a "triangle detector" but **edge, curve, blob, and corner detectors**. Such low-level features are task-general. The final head, by contrast, makes task-specific judgments like "this combination of features means triangle", so it cannot be reused. **"Front is general, back is specific"** — the first principle of transfer-learning design.

### 3.3 When transfer fails

Even an experienced hire flounders if the industry is too different. When the **domain gap** between the pretraining data and yours is large (a model trained on everyday photos ↔ X-ray, microscope, or thermal images), even the low-level statistics differ and the transfer benefit plummets. Then you look for a similar-domain pretrained model, dial up the fine-tuning, or consider pretraining your own — this is where project quotes diverge.

### 3.4 In practice

The standard is to download a backbone pretrained on large public data. The procedure is the same as today's exercise: (1) load the backbone → (2) replace the head with your class count → (3) decide what to freeze → (4) train on your data. The two lines in today's code — `load_state_dict` + `requires_grad=False` — are that procedure in miniature.

### 3.5 Designing the fair comparison

Today's experiment is controlled as "same architecture, same initial head, same training settings; the only difference is the feature extractor's starting point". Making the model deliberately larger (~15K parameters) is part of the design — big enough relative to 24 images that from-scratch training struggles, a realistic condition.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Pretrains a CNN on task A (squares/circles/triangles, 600 images). Task A test accuracy 1.0 — this model's features now carry an eye for shapes.
- **[2]** Enter task B: **diamonds/ellipses/rings** — three shapes drawn directly in numpy inside main.py, none of which exist in task A. Only 24 labeled images (336 test).
- **[3]** Method 1 — from scratch: the same architecture trained from random initialization on 24 images (all 15,587 parameters trainable).
- **[4]** Method 2 — transfer: transplant the pretrained features via `load_state_dict`, freeze them, and train only the 1,539-parameter head.
- **[5]** Compares test accuracy per epoch. Typical result: transfer pulls ahead from epoch 10 (0.66 vs 0.49) and wins at the end too (0.94 vs 0.91). **A tenth of the trainable parameters, yet faster and better.** The curves and task-B samples are saved to `outputs/transfer_learning.png`.

One honest observation to note: given enough epochs, from-scratch training catches up considerably. Transfer learning's gain is less "making the impossible possible" than **"faster and somewhat better, with less data and less compute"**. Shrink the data or grow the model and that gap widens.

## 5. Try It Yourself

1. **(Easy)** Raise `N_B_TRAIN` from 24 to 60. Does from-scratch catch up to or overtake transfer? An experiment confirming "with enough data, transfer's edge shrinks". (Hint: one constant)
2. **(Medium)** Add a fine-tuning variant: transplant the features but do not freeze — train everything (at a low learning rate, 1e-3). Plot all three curves (scratch / frozen transfer / fine-tune) on one chart. (Hint: just drop the `requires_grad=False` line)
3. **(Challenge)** Replace task B with **thin-line shapes** like a plus (+) and an X. Watch the transfer benefit shrink, and explain it with the "domain gap" concept (section 3.3). (Hint: task A's shapes are all solid filled blobs)

## 6. Common Mistakes

- **Not replacing the head**: a pretrained model's output has the original task's class count and meaning. Swapping in a new head for your task is always step one.
- **Fine-tuning without freezing**: update everything at a large learning rate on scarce data and the carefully borrowed features are wrecked within a few epochs (catastrophic forgetting). The less data, the more you freeze.
- **Believing transfer is a cure-all**: a large domain gap slashes the benefit. "What was the pretraining data?" is the key question in model selection.
- **Failing to control the comparison**: if the transfer model and the scratch model differ in architecture, epochs, or learning rate, you cannot tell what caused the win. Leave exactly one difference, as today's code does.

## Next Level Preview

The parts and the techniques are all in hand. Now to string them in real-world order — from data inspection through the error-analysis-and-improvement loop, including catching **label noise**, the perennial trap of production data, through error analysis. Level08 demonstrates it end to end.
