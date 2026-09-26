# Lecture 08 · Level 09 — Techniques for Preventing Overfitting

> Deliberately cause an overfitting accident, then compare in a controlled experiment how much validation performance three prescriptions recover: dropout, weight decay, and early stopping.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level08 (training loops, batches, epochs) / **Estimated time** 60 min

## 1. Why learn this — the business view

Overfitting is the number-one path by which deep-learning projects fail. The report card looks brilliant during development, then performance collapses after launch — the true identity of "but it worked in the demo" is usually this. The scary part is that overfitting arrives quietly: watch only the train metrics and everything looks great. Finish this level and you'll be able to (1) spot overfitting early on a learning curve, (2) pick the prescription that fits the situation, and (3) answer "can we trust this model's performance numbers?" from a validation-methodology standpoint. If you saw tree-model overfitting in lecture06 level09, this is the neural-network version — plus the network-specific prescription, dropout.

## 2. Understanding through an analogy

**Overfitting = memorizing past exam papers wholesale.** 120 training rows, 4,700 model parameters. Studying 120 questions with 4,700 parameters is like memorizing the answers to 120 past-exam questions verbatim instead of understanding the whole syllabus. The past-exam (train) score is 100; the real-exam (valid) score craters.

All three prescriptions are forms of "sabotaging the memorization."

- **Dropout = random-absence training.** In every training session, half the team (neurons) is randomly absent. Dependence on specific neurons — "only Kim knows that problem" (co-adaptation) — becomes impossible, and everyone builds general knowledge so they can cover for each other. On exam day (evaluation), everyone shows up — that is what the `model.train()`/`model.eval()` switch really is.
- **Weight decay = a tax on large weights.** Answers that lean extremely on one input (large weights) get taxed. The model comes to prefer moderate answers that "weigh many pieces of evidence a little each."
- **Early stopping = descending at peak grades.** When the mock-exam (valid) score starts worsening, you adopt the ability at that moment as final, even with study time (epochs) left.

## 3. Key concepts

### 3-1. The condition for overfitting: capacity out of proportion to data

When parameter count ≫ data count, the model can "store" the data instead of "understanding" it. In the exercise, 100% train accuracy is not a boast but a warning — churn has inherent randomness, so an honest model could never score 100% even on train.

### 3-2. How dropout works

- During training: each neuron's output is zeroed with probability p, and the survivors are scaled by 1/(1-p) (keeping the total signal size).
- During evaluation: nothing is switched off. Forget `model.eval()` and neurons stay absent during evaluation too, eroding performance.
- Another reading of the effect: each pass trains a different sub-network, so it resembles an ensemble of many small models (lecture06 level08).

### 3-3. How weight decay works

Equivalent (under SGD) to L2 regularization — adding λ × the sum of squared weights to the loss. In PyTorch it's a single argument: `optim.Adam(..., weight_decay=3e-2)`. Too large a λ tips you into underfitting (the train accuracy dipping to 0.89 in the exercise is that early sign), so tune it by validation performance.

### 3-4. How early stopping works

Measure valid loss each epoch and save the model whenever the record low is beaten. If no improvement arrives for a set number of epochs (the patience), stop and revert to the saved copy. Nearly free (zero extra training cost) with a large effect — in practice it's effectively standard equipment.

### 3-5. Choosing a prescription

| Situation | First-line prescription |
|---|---|
| Whatever else, always | Early stopping (zero cost, few side effects) |
| Big network, scarce data | Dropout + weight decay together |
| Severe gap and budget available | Collect/augment data (the root-cause cure) |
| Even train performance is low | Not overfitting — loosen the regularizers and revisit capacity/learning rate |

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level09_regularization_dropout
python3 main.py
```

Output, in order:

- **[1]** Of churn_table's 2,000 rows, only **120** go to training and 1,880 to validation (a big validation set makes the report card stable).
- **[2]** From the same initial weights (fixed seed), trains three conditions — unprotected / dropout / weight decay — for 600 epochs each. The unprotected model: train 100% / valid 0.777 — textbook memorization.
- **[3]** The unprotected model's valid loss bottoms out at just epoch 19. Stopping there would have given valid 0.847. The remaining 581 epochs spent their time *eroding* performance.
- **[4]** Final table: unprotected 0.777 → dropout 0.820 / weight decay 0.836 / early stopping 0.847. All three prescriptions recover performance, and an `assert` guarantees the recovery reproduces.
- **[5]** `outputs/regularization_compare.png`: on the left (loss), the red dashed train line plunges toward 0 while the red solid valid line rebounds — the scissors shape; on the right (accuracy), the recovery per prescription.

## 5. Try it yourself

1. **(Easy)** Increase the training data from 120 → 1,000 rows. How does the unprotected model's valid performance change? *Hint: the root-cause cure for overfitting is data. The gaps between prescriptions shrink too.*
2. **(Medium)** Try dropout p at 0.1 / 0.5 / 0.9. What happens at p=0.9? *Hint: with a 90% absence rate the team can't work — underfitting.*
3. **(Challenge)** Implement real early stopping: patience=30 — if valid loss hasn't improved for 30 epochs, break the loop and restore the best checkpoint's `state_dict`. *Hint: save with `best_state = copy.deepcopy(model.state_dict())`, then `model.load_state_dict(best_state)`.*

## 6. Common mistakes

- **Missing `model.eval()` at evaluation.** The model is graded with dropout still on, and performance turns erratic. Conversely, don't forget to return to `model.train()` when training resumes.
- **Basing early stopping on train loss.** Train loss almost always falls, so it never stops. The criterion must be valid.
- **Early-stopping on the test set.** Choose the stopping point with valid and report final numbers on an untouched test set — otherwise the test set has effectively joined the training.
- **Turning every prescription up to maximum.** Dropout 0.5 + strong weight decay + very early stopping all at once is a straight road to underfitting. Add one at a time and check the curves.

## Next level preview

After overfitting, the next most common accident is "training that fails because it's unstable." In level10 we experiment with stabilizing training: schedules that adjust the learning rate over time (warmup, cosine decay), gradient clipping that cuts off runaway gradients, and batch normalization that recalibrates the scale at every layer.
