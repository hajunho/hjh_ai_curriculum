# Lecture 08 · Level 10 — Learning Rates and Training Stability

> Same model, same data — yet "how you drive the learning rate" decides between a divergence crash and peak performance. Learn warmup, cosine decay, gradient clipping, and batch normalization through controlled experiments.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level09 (overfitting prevention) / **Estimated time** 55 min

## 1. Why learn this — the business view

Modern training reports and papers' technical sections almost invariably contain phrases like "cosine decay after warmup, gradient clipping 1.0." It's the standard grammar of LLM training recipes. Finish this level and that phrase decodes itself. The practical value is even more direct — when the loss blows up to NaN in a training log, you can perform the first-response yourself: "lower the learning rate and add clipping." GPU time is money, so preventing a single training run lost to divergence already pays back the cost of this level. And if you started with "just use Adam" in level06, this time you learn the craft of driving the SGD family at high performance.

## 2. Understanding through an analogy

**The three rules of driving a learning rate.**

- **Warmup = don't sprint the moment you arrive at work.** Early in training the weights are random, so the early gradients are like "strong opinions from a new hire who doesn't know the situation yet." For the first few steps, ramp the stride up gradually while you survey the terrain.
- **Cosine decay = polish finer as the deadline nears.** Bold changes of direction help early in a project; right before the deadline you should only fine-tune. Lower the learning rate smoothly along a cosine curve toward 0 and you settle at the valley floor without residual jitter.
- **Gradient clipping = limiting the steering wheel's maximum turn.** Even when one weird batch sends the signal "crank the wheel 720 degrees!" (a huge gradient), this safety device physically refuses to turn past a set angle. Direction preserved, only magnitude cut.
- **Batch normalization = a gauge recalibrator between layers.** As the production line lengthens (the network deepens), the intermediate products' units balloon or shrink arbitrarily. Recalibrate to mean 0 / variance 1 at the entrance of each stage and the downstream stages always receive inputs in a familiar range, staying stable.

## 3. Key concepts

### 3-1. Anatomy of a divergence

Level03's "learning rate too big → leaping over the valley" plays out more dramatically in multi-layer networks. One big jump in the weights produces even bigger gradients, momentum accelerates in that direction, and within a few steps the loss hits infinity (NaN). Condition B in the exercise is exactly this crash. When a log shows NaN loss, respond in this order: learning rate down, add clipping, and (if severe) check the data for outliers.

### 3-2. The warmup + cosine decay schedule function

- First 10% of total steps: increase lr linearly 0 → base_lr.
- Afterwards: lr = base_lr × 0.5 × (1 + cos(π × progress)) — smoothly down to 0.

In PyTorch you'd use `torch.optim.lr_scheduler.LambdaLR` or `CosineAnnealingLR`, but the exercise sets `param_groups[0]["lr"]` directly each step so the principle stays visible. That is what the framework schedulers do internally anyway.

### 3-3. Two kinds of clipping

- **Norm clipping (the standard)**: treat all gradients as one vector; if its total length exceeds the threshold, scale it down proportionally. Direction preserved. `clip_grad_norm_(params, max_norm)`.
- **Value clipping**: clamp each component to [-c, c]. Can skew the direction, so it's auxiliary.

Common starting thresholds are 0.5–1.0; the rule of thumb is "slightly larger than your typical gradient magnitude."

### 3-4. Batch normalization, the concept (concept only, this time)

Insert `nn.BatchNorm1d(64)` after a Linear layer and it normalizes per batch, restoring expressiveness with learnable scale/shift parameters (γ, β). At evaluation it uses moving-average statistics accumulated during training, so the `model.eval()` switch matters here too. Deep CNNs (lecture09) use it in earnest.

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level10_lr_schedules_stability
python3 main.py
```

With the same initial weights (fixed seed), a 1-64-64-1 MLP learns the y=sin(2x) curve regression under four recipes.

| Condition | Recipe | Final valid MSE |
|---|---|---|
| A | fixed lr=0.05 | 0.0071 |
| B | fixed lr=0.2 | **NaN (diverged)** |
| C | lr=0.2 + clipping 0.5 | 0.0528 |
| D | warmup+cosine(0.2) + clipping | **0.0045 (best)** |

- B: just 4x the learning rate, and it explodes within a few steps. The "diverged (NaN)" in the output is what a real crash log looks like.
- C: at the same excessive speed, clipping keeps it alive. Performance is middling, though — a safety device prevents crashes; it doesn't substitute for good driving.
- D: clipping plus the schedule beats even safe driving (A). The high mid-run learning rate explores widely and the final taper settles precisely. `assert`s guarantee B's divergence and D<A reproduce.
- `outputs/lr_schedule_compare.png`: left, per-condition valid loss (log scale, NaN capped at the top); right, the step-by-step learning-rate curves of A and D — see the trapezoid-shaped warmup+cosine with your own eyes.

## 5. Try it yourself

1. **(Easy)** Change the warmup fraction 10% → 0% (off) and rerun D. How does the early loss curve change? *Hint: in `lr_at`'s warm computation, change 0.1 to 0.0. An early spike may appear.*
2. **(Medium)** Compare C with clipping thresholds 0.5 → 5.0 → 0.05. What happens when it's too big, and when too small? *Hint: too big disables the safety device (divergence returns); too small clips every step, effectively just lowering the learning rate.*
3. **(Challenge)** Add `nn.BatchNorm1d(64)` to `build_model` and rerun condition B (fixed lr=0.2, no clipping). Does the divergence disappear? *Hint: insert it between Linear(1,64) and Tanh. Batch normalization acts as a cushion for large learning rates.*

## 6. Common mistakes

- **Updating the schedule only per epoch.** Warmup is usually per "step." Done per epoch, the entire warmup can get skipped.
- **Calling clipping before backward.** Gradients only exist after `backward()`. The order is backward → clip → step.
- **Continuing training after a divergence.** Once NaN appears, the weights are already contaminated. Restart (or restore the last healthy checkpoint).
- **Over-relying on schedules with Adam.** Adam adapts on its own, so schedule gains can be smaller than with SGD. Yet in large-model training (LLMs), AdamW + warmup + cosine is the standard — common sense shifts with scale.
- **Batch normalization + batch size 1–2.** The statistics turn unstable and it does harm. For small batches use the LayerNorm family (transformers are the example).

## Next level preview

"How to train well" is now complete. The final level11 covers "what to train on" — why GPUs, what fp16/bf16 mixed precision is, and how a model that doesn't fit on one GPU is carved up (DDP/FSDP). You'll compute the training memory of an 8B-parameter model yourself, producing numbers you can use in a GPU budget meeting.
