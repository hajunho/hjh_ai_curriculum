# Lecture 09 · Level 08 — A Real-World Image Classification Pipeline

> We complete the inspect → split → train → error-analysis → improve loop, demonstrating how error analysis hunts down label noise, the perennial trap of production data.
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level05–07 / **Estimated time** 50 min

## 1. Why This Matters — the Business View

The post-mortems of failed vision projects contain strikingly similar sentences: "Several models were attempted, but data-quality issues were discovered late…" Model code can be changed in a few lines, but thousands of wrongly attached labels quietly lower the performance ceiling. Outsource the labeling and a few percent error rate comes bundled by default; in-house labeling wobbles too, as each annotator applies different criteria.

That is why the real skill of a production pipeline lies in **error analysis** — not stopping at the performance number, but systematically digging into the cases the model gets wrong (or struggles with) and reading "what to fix next" out of the data. Today we demonstrate this skill hunting down label noise, start to finish. None of the tools are new — we merely string the parts from level05–07 together in real-world order.

## 2. An Analogy

**Finding textbook errors through the mistake journal.**

At a cram school, mock-exam scores strangely refuse to improve. It turns out **part of the textbook's answer key was misprinted** (label noise). The student (the model) strains to match the wrong answers and even develops odd solving habits.

How do you find the answer-key errors? Re-reviewing every question costs too much. The trick: **start with the questions a diligent student has marked "no matter what, I can't make sense of this"**. If the student has built real skill on the rest of the problems, the questions they still cannot accept are likely the answer key's fault, not the student's.

In model training, the precise measure of "can't make sense of it" is the **per-sample loss**. Regrade the training data with the finished model, sort by loss descending, and the label errors cluster at the top. A human then re-reviews only that top slice — 60 targeted checks instead of 600 exhaustive ones.

## 3. Core Concepts

### 3.1 The five pipeline stages

1. **Data inspection**: counts, sizes, value ranges, class balance, eyeballing samples. Done before opening any model code.
2. **Splitting**: train/validation/test. In particular, **the validation and test labels are verified personally by your most trusted reviewer**. If the ruler is bent, every experiment is meaningless.
3. **Baseline training**: build a reference point quickly with a simple model.
4. **Error analysis**: look at cases, not numbers. Misclassified images, high-loss training samples, per-class patterns.
5. **Improvement loop**: fix the one thing the analysis points to (label correction, data reinforcement, augmentation tuning…) and re-run. Repeat.

### 3.2 The two faces of label noise

- **A lowered performance ceiling**: if 8% of the answers are lies, even the best model damages its generalization trying to fit those lies.
- **Distorted evaluation**: let noise seep into the test set and you can never know the "true performance". That is why money goes to evaluation-set label quality first.

### 3.3 Loss-based review — the principle and its limits

The principle: the model learns general rules from the majority of correct labels, so labels that clash badly with the rules (= errors) receive large losses. In the main.py experiment, re-reviewing the top 60 catches **about 98%** of the 48 corrupted labels (random review's expectation: 4.8).

Know the limits too: the high-loss slice also contains **correctly labeled but genuinely hard cases** (shapes on a boundary, heavy noise). That is why the rule is **human re-review**, not automatic deletion. Delete the hard cases and the model turns into a fair-weather performer that only solves easy problems.

### 3.4 "Fix the model, or fix the data?"

When performance falls short, the most expensive reflex is "try a bigger model". As today's experiment shows, fixing only the data with the same model yields substantial gains. The priority instinct: **evaluation-set quality → label quality → data volume and diversity (including augmentation) → model size and architecture**. The earlier items are cheaper and more reliably effective.

### 3.5 One change per loop

If you fix labels, add augmentation, and change the model all at once, and performance improves — you can't tell what worked. The iron law of experiment management is consistent since lecture06 — **one change per loop, comparison under identical conditions, verdict on an untouched evaluation set**.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

- **[1]** Data inspection: checks counts, value ranges, class balance. Then the scenario fires — 48 of the 600 training labels (8%) are secretly corrupted. The following stages proceed pretending not to know.
- **[2]** Split: train 600 (corruption included) / validation 150 / test 150 (evaluation labels are clean — the principle from 3.2).
- **[3]** v1 training: trains on the corrupted labels as-is. Test accuracy comes in below expectation (level05's ~0.97).
- **[4]** Error analysis: regrades the training data by per-sample loss and flags the top 60 as "for human re-review". Checks against the answer key how many truly corrupted labels were caught (typical result: 47 of 48, 98% detection).
- **[5]** Improvement loop: fixes the flagged labels and retrains as v2. Confirms the v1 → v2 accuracy gain and compares against "if it had been clean from the start" (oracle). Saves the high-loss gallery (red title = truly corrupted) and the performance bars to `outputs/pipeline_error_analysis.png`.

The heart of the code is `per_sample_loss()` — one line, `CrossEntropyLoss(reduction="none")`, yields "how much each sample torments the model". The instant you unfold the loss you usually crush into an average, sample by sample, the loss function transforms into a data-quality diagnostic.

## 5. Try It Yourself

1. **(Easy)** Raise `NOISE_RATE` from 0.08 → 0.20. How far does v1 collapse, and does the detection rate hold? (Hint: with more corruption, the "majority of correct labels" premise weakens)
2. **(Medium)** Cut the review budget `top_k` to half the corruption count (24). How do detection rate (caught/total corrupted) and precision (truly corrupted among reviewed) change? This is the trade-off between review labor cost and performance. (Hint: precision is highest at the very top)
3. **(Challenge)** Build a v2' that **removes suspicious samples** instead of fixing their labels, and compare with v2. Can you see the cost of removal versus correction (throwing away perfectly good hard cases)? (Hint: exclude indices with `np.setdiff1d`)

## 6. Common Mistakes

- **Leaving evaluation-set contamination alone**: cleaning evaluation labels comes before cleaning training labels. With a bent ruler, improvement doesn't look like improvement.
- **Auto-deleting high-loss samples**: the top slice contains "hard but correct cases" too. Use it as a human-review list, not an automatic filter.
- **Fixing several things at once**: you can't tell what worked, so no knowledge carries to the next project.
- **Treating error analysis as a one-off event**: after deployment, new data keeps flowing in and labeling criteria keep drifting. Error analysis is a permanent stage of the pipeline.

## Next Level Preview

So far it has been one image, one answer — "what is it". The real-world question soon evolves into "**what is where**". Bounding boxes, IoU, sliding windows through modern detectors — in level09 we build a mini detector over scenes scattered with multiple shapes.
