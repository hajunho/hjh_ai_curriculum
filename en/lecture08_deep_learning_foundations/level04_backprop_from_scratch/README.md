# Lecture 08 · Level 04 — Backpropagation from Scratch

> Implement backpropagation — the heart of neural-network training — in pure numpy to solve XOR, and even verify "my derivatives are correct" with numerical differentiation.

**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level03 (loss functions and gradient descent) / **Estimated time** 70 min

## 1. Why learn this — the business view

This is the hardest level in the entire lecture — and the most valuable. People who have implemented backpropagation by hand even once treat deep learning differently from people who haven't. To the former, deep learning is "a machine that does calculus automatically"; to the latter, "an inscrutable magic box." At work, magic boxes are dangerous — when training fails you can't reason about the cause, and you can't verify a vendor's explanation. The good news: the core is a single idea from high-school calculus, the chain rule, and from the next level on, PyTorch does all of this automatically. In other words, **you do it by hand exactly once today, then use it automatically for the rest of your life.**

## 2. Understanding through an analogy

**Tracing responsibility for a defective product.** Suppose a finished product comes off the factory line defective (= the loss is large). The quality team traces backwards like this:

1. Final inspection stage: "What percent of the defect is the assembly process's fault?"
2. Assembly process: "Of that responsibility, what percent is part A's fault, and what percent part B's?"
3. Parts process: "Of that responsibility, what percent is the raw material's fault?"

Each stage only needs to distribute responsibility **to the stage immediately before it**. Nobody needs to understand the whole line. Flowing responsibility (the error) backwards from output to input like this is backpropagation, and the mathematical tool that computes the "responsibility split ratios" is the chain rule: total effect = product of each stage's effect.

The forward pass is values flowing "raw material → parts → assembly → finished product"; the backward pass is "responsibility" flowing the same road in reverse. Because the road is the same, memoizing the intermediate values during the forward pass makes the backward pass almost free.

## 3. Key concepts

### 3-1. The network we'll build: 2-8-1

- 2 inputs (the two XOR coordinates) → 8 hidden neurons (tanh) → 1 output (sigmoid, a probability 0–1)
- The loss is binary cross-entropy (BCE) — the same grading sheet as level03.

### 3-2. Forward pass — four lines, that's all

```
z1 = X @ W1 + b1      # layer-1 weighted sum
h  = tanh(z1)         # nonlinearity (mandatory, for the reason learned in level02)
z2 = h @ W2 + b2      # layer-2 weighted sum
p  = sigmoid(z2)      # convert to a probability
```

### 3-3. Backward pass — distributing responsibility in reverse

Writing the chain rule layer by layer gives the table below. Treat the derivations as an appendix; grasping the structure — "multiply your way down from output to input" — is enough.

| Step | Formula | Analogy |
|---|---|---|
| Output-layer error | dz2 = (p − y) / n | The size of the finished-product defect. The BCE+sigmoid combo simplifies this cleanly |
| W2's responsibility | dW2 = hᵀ @ dz2 | "The more a hidden neuron contributed, the more responsibility it bears" |
| Pass to hidden layer | dh = dz2 @ W2ᵀ | Distribute responsibility to the previous stage in proportion to connection strength |
| Through tanh | dz1 = dh × (1 − h²) | Saturated tanh neurons let almost no responsibility through (the true face of vanishing gradients) |
| W1's responsibility | dW1 = Xᵀ @ dz1 | Distribute responsibility by how much each input contributed |

The last step is exactly level03: update every weight with `W -= learning_rate × dW`.

### 3-4. Numerical-gradient verification — backprop's unit test

A backprop implementation is easy to break with one wrong sign or one wrong transpose. The check is simple: move one weight by ±0.000001, measure the loss change directly (central difference), and compare with what backprop computed. A relative error under 1e-4 passes. Slow but certain — real framework developers test their autodiff the same way.

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level04_backprop_from_scratch
python3 main.py
```

Output, in order:

- **[2] Gradient verification**: prints the maximum relative error between backprop and numerical differentiation for each of W1, b1, W2, b2. All around 1e-7 — practically identical. An `assert` enforces < 1e-4, so if you edit the code and break a derivative, it's caught immediately.
- **[3] Training**: learning rate 0.5, 3000 epochs. Accuracy already hits 100% around epoch 100, and the loss keeps falling.
- **[4] Results**: the four inputs' predicted probabilities split confidently, around 0.001 / 0.999. The "75% wall of linear models" proved in level00 has been crossed.
- **[5] Hidden representation**: shows each input's hidden values. XOR, unsplittable by a line in the original coordinates, splits along a line in the new coordinates the hidden layer built — the "representation learning" foreshadowed in level00, actually happening.

The must-read part of the code is the `backward()` function (about 15 lines): the five rows of table 3-3 transcribed directly into numpy.

## 5. Try it yourself

1. **(Easy)** Reduce the hidden neurons from 8 to 2 (change the 8 in `init_params` to 2). Does XOR still get solved? What about 3? *Hint: in theory 2 hidden neurons suffice for XOR, but it can fail depending on initialization luck. Try different seeds and observe.*
2. **(Medium)** Delete `(1.0 - h ** 2)` from `dz1 = dh * (1.0 - h ** 2)` (i.e., omit the tanh derivative) and run. See how the verification in [2] fails. You'll feel why the numerical check is a "unit test."
3. **(Challenge)** Add another hidden layer, extending to 2-8-8-1. *Hint: add z2/h2 to forward, and repeat the dh2 → dz2 distribution step once more in backward. The moment you notice the pattern is identical, you've understood that a 100-layer network works on the same principle.*

## 6. Common mistakes

- **Transpose/shape mistakes.** Writing `h.T @ dz2` as `dz2 @ h.T` is the most common class of bug. Writing each array's shape in a comment as you code prevents most of them.
- **Not saving forward-pass intermediates.** Backprop reuses intermediates like h and p. Recomputing without saving is slow — and recomputing with different inputs makes the derivative wrong.
- **Making the numerical-diff step h too small.** At 1e-12, floating-point rounding error swallows the derivative. 1e-5 to 1e-6 is about right.
- **Loss stuck at 0.6931.** ln 2 ≈ 0.693 means "coin-flip level." The learning rate is too small, or the initial weights were all zero so the neurons move identically (the symmetry problem). Always initialize with small random numbers.

## Next level preview

Congratulations — today is the last day you write derivatives by hand. From level05 we use PyTorch. First a tour of PyTorch's basic ingredient, the **tensor** — what it shares with a numpy array and what's different (autodiff, the ability to move to a GPU) — then in level06 you'll watch today's 30 lines of backprop shrink to the single line `loss.backward()`.
