# Lecture 08 · Level 08 — Training Loops, Batches, and Epochs

> Implement the production-standard loop that trains on data portioned onto plates (mini-batches) using Dataset/DataLoader, and learn to read the "training health report" by plotting train/valid loss curves.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level07 (MLP classifier) / **Estimated time** 55 min

## 1. Why learn this — the business view

The training-loop template you build in this level is the **skeleton of every piece of deep-learning code you will ever see**. Computer vision (lecture09), NLP (lecture10), LLM fine-tuning (lecture12) — the model and data change, the loop stays the same. And the screen a working deep-learning practitioner looks at several times a day is the learning curve. Learn to read those two lines and you can answer questions like "is it going well right now?", "will more training help?", and "when do we stop?" with data. The graph that arrives in your vendor's weekly report — after this level, you interpret it yourself.

## 2. Understanding through an analogy

**The buffet-plate analogy.** You can't eat the whole warehouse of food (all 1,600 rows) in one sitting (memory limits). So you portion it onto plates (batches of 64) and make several trips.

- **Batch** = the amount on one plate.
- **Step** = clearing one plate (one weight update from one batch).
- **Epoch** = eating your way through the entire warehouse once (1,600 ÷ 64 = 25 steps).
- **Shuffle** = randomizing the plating order every epoch. Always eating in the same order trains your palate to the taste of the last plate (the tendency of recent batches).

**Dataset = the warehouse librarian, DataLoader = the server.** The librarian (Dataset) answers only two questions — "how many items total?" (`__len__`) and "give me item i" (`__getitem__`). The server (DataLoader) calls out ticket numbers to the librarian, assembles plates, shuffles, and serves. Because the roles are separated, whether the data is a CSV or a folder of images, you swap only the librarian and the loop stays the same.

## 3. Key concepts

### 3-1. Why mini-batches

| Approach | Pros | Cons |
|---|---|---|
| full-batch (everything at once) | Exact gradients | Memory limits; 1 update per epoch, slow |
| batch of 1 (online) | Frequent updates | Wild gradients, hardware-inefficient |
| **mini-batch (32–512)** | The compromise + optimal for GPU parallelism | Adds batch size as a hyperparameter |

The mini-batch's "moderately jumpy" gradient isn't purely a downside — it also helps escape shallow dips (bad local optima).

### 3-2. The standard training-loop template

```
for epoch in range(n_epochs):
    one pass over train: per batch, zero_grad → forward → loss → backward → step
    one pass over valid: grade only, under no_grad (no backward!)
    record both losses
```

The exercise's `run_epoch()` function is the practical idiom that serves both training and evaluation depending on whether opt is passed. The key points: `model.eval()` + `torch.no_grad()` during evaluation, and weighting the loss average by batch size.

### 3-3. The learning curve = a health report

- **train and valid falling together**: healthy. Keep going.
- **train falling, valid rising (scissors opening)**: overfitting has begun. Only the practice-exam score is improving — stop, or apply level09's prescriptions.
- **both stuck high**: underfitting. Revisit model capacity, learning rate, features.
- **sawtooth thrashing**: learning rate too high or batch too small.

### 3-4. BCEWithLogitsLoss

This time the model drops the final Sigmoid and outputs raw logits (the score before conversion to a probability). That's because `BCEWithLogitsLoss` computes Sigmoid+BCE internally in one numerically safe step. It's the production standard, so we adopt it from here on. Classification works the same too: logit > 0 instead of probability > 0.5.

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level08_training_loops
python3 main.py
```

Output, in order:

- **[1]** churn_table's 2,000 rows → train 1,600 / valid 400. Uses `train_ds[0]` to see how the librarian hands over a single item.
- **[2]** DataLoader setup: 1 epoch = 25 steps. Batch, step, and epoch made concrete with real numbers.
- **[3]** 40 epochs of training. Valid loss bottoms out around epoch 5 at 0.388; afterwards train keeps falling while valid creeps up (0.39 → 0.42). **Watching overfitting begin, live** — that's the highlight of this exercise.
- **[4]** Saves `outputs/training_curve.png`. See for yourself the "scissors shape" as the blue line (train) and the red line (valid) pull apart.
- **[5]** Prints a summary table for reading curves.

## 5. Try it yourself

1. **(Easy)** Change batch_size 64 → 8 → 1600 (full batch) and rerun. How do the curve's smoothness and total training time change? *Hint: smaller batches mean more steps per epoch and rougher curves.*
2. **(Medium)** What happens if you flip `shuffle=True` to False? The difference is small on this data, but think about what would go wrong if the data were sorted by label. *Hint: plates of nothing but 0s for a while, then plates of nothing but 1s — the gradient lurches one way, then the other, on repeat.*
3. **(Challenge)** Add logic that saves the model from the epoch with the lowest valid loss. *Hint: start with `best = float('inf')` and on improvement `torch.save(model.state_dict(), "best.pt")`. This is half of the early stopping you'll learn next level.*

## 6. Common mistakes

- **Calling backward in the valid loop.** A serious accident — you'd be training on the validation data. Grade only, with no_grad and no opt.
- **Simple-averaging the epoch loss.** The last batch can be a different size, so weight it: `loss.item() * len(xb)`.
- **Recreating the Dataset every epoch.** The preprocessing repeats every time and wastes wall-clock. Build the Dataset once and reuse it.
- **Reaching for num_workers first.** Parallel loading pays off on large image datasets, but on small data the process-spawn cost dominates and it's actually slower (this exercise uses the default 0).
- **Believing "more epochs always help."** This run is the counterexample — valid performance at 40 epochs is worse than at 5.

## Next level preview

The "scissors shape" you witnessed in this run (train↓ valid↑) is overfitting itself. In level09 we'll deliberately manufacture severe overfitting, then compare how much valid performance each of three prescriptions recovers: dropout (random-absence training), weight decay (a tax on large weights), and early stopping (descend when the grades turn bad).
