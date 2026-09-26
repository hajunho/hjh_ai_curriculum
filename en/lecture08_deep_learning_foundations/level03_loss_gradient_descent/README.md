# Lecture 08 · Level 03 — Loss Functions and Gradient Descent

> See with your own eyes, on a one-variable function, that training a model means "moving step by step in the direction that lowers the score on the grading sheet (the loss function)."

**Difficulty** ⭐⭐⭐ / **Prerequisites** level02 (activation functions) / **Estimated time** 45 min

## 1. Why learn this — the business view

The two phrases you hear most often in deep-learning project meetings are "the loss won't go down" and "let's lower the learning rate." Without knowing the loss function and gradient descent, you can't join that conversation. Conversely, knowing just these two, you can look at a training log a vendor sent over and say, "the loss is bouncing around — looks like the learning rate is too high." Whatever the model — a churn-prediction MLP or an LLM for a chatbot — the skeleton of training is always the same one sentence: "define a loss, walk down the gradient." The principle you learn here is reused, unchanged, in every level that follows.

## 2. Understanding through an analogy

**Descending a mountain in fog.** Imagine standing halfway up a mountain in thick fog. No map, no view of the summit or the valley. The only thing you can know is **the slope under your feet** — which way is downhill. There is exactly one sensible strategy: take a step downhill, check the ground again, take another step. That is all gradient descent is.

- **The mountain's altitude** = the loss: the score for how wrong the model is. Lower is better.
- **Your current position** = the model's parameter (weight) values.
- **The slope underfoot** = the gradient. Calculus tells you this.
- **Your stride** = the learning rate — the star of this level.

If your stride is too short, the sun sets before you reach the valley. If it's too long, you leap clear over the valley onto the far cliff, then somewhere higher still — you end up *climbing* the mountain. The hands-on section reproduces all three cases.

## 3. Key concepts

### 3-1. The loss function = the model's grading sheet

A loss function summarizes "how far the model's prediction is from the truth" in a single number. The standard grading sheet differs by problem type.

| Problem | Loss function | Intuition |
|---|---|---|
| Regression (revenue forecasting) | MSE (mean squared error) | Square the errors and average. Big mistakes get big penalties |
| Binary classification (churn or not) | Binary cross-entropy | The lower the probability assigned to the truth, the bigger the penalty |
| Multi-class classification (3 shape types) | Cross-entropy | Same as above, with multiple classes |

Why not use MSE for classification? Applied to probability outputs (0–1), MSE's penalty when wrong isn't steep enough, so training slows down. Cross-entropy grows the penalty geometrically for cases like "predicted the true class at probability 0.01," strongly correcting confident wrong answers.

### 3-2. The gradient = the slope meter under your feet

The gradient (derivative) is "if I increase x a tiny bit from here, how much does the loss change." A positive gradient means uphill is to the right, so move left; negative means move right. The update rule is this one line:

```
x_new = x_old - learning_rate × gradient
```

Why the minus sign: the gradient points "uphill," so to descend you go the other way.

### 3-3. The learning rate — the most important dial

| Learning rate | Symptom | What the log looks like |
|---|---|---|
| Too small | Convergence takes forever | Loss decreases very slowly but steadily |
| Just right | Reaches the valley in a few steps | Loss drops fast, then flattens |
| Too big | Keeps leaping over the valley | Loss oscillates or explodes (NaN) |

Practical tip: first sweep values 10x apart — 0.1, 0.01, 0.001 — then work near whichever drops fastest while staying stable.

### 3-4. Numerical differentiation — the gradient checker

Hand-implementing derivative formulas invites mistakes. So we cross-check with `(f(x+h) - f(x-h)) / 2h` (the central difference): "nudge a tiny bit and measure the actual change." In the next level (backpropagation), this checking technique becomes a core tool.

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level03_loss_gradient_descent
python3 main.py
```

Output, in order:

- **[1]–[2]** Introduces the loss function f(x) = (x-3)² + 0.7·sin(3x) and verifies that the analytic derivative and numerical differentiation agree (both +6.849769).
- **[3]** From the start point x=9, runs 30 steps of gradient descent at learning rates 0.01 / 0.15 / 1.05, printing position and loss per step.
  - 0.01: even after 30 steps, x≈6.3 — still halfway up the mountain.
  - 0.15: reaches the valley at x≈3.5 within 10 steps, then stays put.
  - 1.05: x leaps around — −2.9 → 11.4 → … → 103 — and the loss passes 10,000. Divergence.
- **[5]** Saves `outputs/gd_learning_rates.png`, drawing the three trajectories on the function curve. The blue square is the start point; the red dots trace the path.

The heart of the code is a single line in `gradient_descent()`: `x = x - lr * grad_f(x)`. The optimizers in deep-learning frameworks are, at bottom, fancy variations of this one line.

## 5. Try it yourself

1. **(Easy)** Run with learning rates 0.05 and 0.3 instead of 0.15. Compare how many steps it takes to reach the valley. *Hint: just change the numbers in the `settings` list.*
2. **(Medium)** Change the start point `x0` from 9.0 to −1.0. Same learning rate, but the destination can differ. Why? *Hint: the sin term creates several small dips (local minima). Gradient descent goes to the "nearest" valley.*
3. **(Challenge)** Replace f(x) with a two-variable function like f(x, y) = (x-1)² + 10·(y-2)², and implement gradient descent with the gradient vector (∂f/∂x, ∂f/∂y). Watch the trajectory zigzag when the slopes in the x and y directions differ by 10x. *Hint: have grad return a tuple and update x and y separately.*

## 6. Common mistakes

- **Setting the learning rate once and never touching it.** When training fails, suspect the learning rate before the model architecture. Adjust it in factors of 10.
- **Sign mistakes.** Write `x + lr * grad` and you climb the mountain. If the loss steadily *increases*, it's almost always a sign or learning-rate problem.
- **Local-minimum phobia.** Worrying "what if there are many valleys?" is 1–2 dimensional intuition. In real high-dimensional neural networks, saddle points and flat regions are the problem rather than local minima, and you usually land in a "good enough" valley.
- **Mistaking the loss value itself for performance.** Loss only tracks your progress downhill; business performance (accuracy, recall) must be measured separately.

## Next level preview

You've learned to compute the slope for "one variable." But a neural network has thousands to billions of weights. The algorithm that computes the gradients of all of them at once, efficiently, is **backpropagation**. In level04 you'll implement backpropagation from scratch in pure numpy and solve the XOR problem with it.
