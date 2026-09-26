# Lecture 08 · Level 01 — Building a Perceptron by Hand

> Implement the smallest unit of a neural network, the perceptron, in about 30 lines of numpy, and watch — epoch by epoch — the learning rule that nudges weights whenever it's wrong solve AND/OR and fail on XOR.

**Difficulty** ⭐⭐ / **Prerequisites** level00_why_neural_networks / **Estimated time** 50 min

## 1. Why learn this — the business view

Even a giant language model is, in the end, a combination of billions of very simple parts. One of those parts is the perceptron — an artificial neuron. Understand exactly how one part works, and phrases like "stacking layers," "training the weights," and "the model converged" stop sounding like magic.

From a practical standpoint, the real payoff of this level is seeing what the word *learning* actually is. When AI "learns," nothing mystical happens: it predicts, adjusts a few numbers in proportion to how wrong it was, and predicts again — on repeat. With that intuition, when a vendor reports "we retrained the model," you can judge what they did and how much, why training takes time, and why some problems never train no matter what (like XOR). The failure demonstration at the end matters most of all: only people who know a tool's limits can choose the right tool.

## 2. Understanding through an analogy

A perceptron is like weighted voting in a meeting. Several attendees vote yes (1) or no (0) on a proposal — but not everyone's voice counts the same. The team lead's opinion carries weight 3; the new hire's, 0.5. Multiply each vote by its speaking weight, sum them up, and if the total clears the passing threshold the proposal is approved; otherwise it's rejected. Here the attendees' votes are the inputs (x), the speaking weights are the weights (w), and the value that shifts the threshold is the bias (b).

So what is learning? It's auditing this committee's decisions against outcomes later. If it approved something that turned out badly (truth 0, prediction 1), you slightly reduce the speaking weight of those who voted yes. If it rejected what turned out to be a good proposal (truth 1, prediction 0), you slightly increase the weight of those who voted yes. If the decision was right, you change nothing. Repeat this simple personnel adjustment for every proposal (data point), and the distribution of speaking weights gradually drifts toward "a distribution that makes good decisions." That is the entire perceptron learning rule.

One caveat up front: this committee is a single-layer structure with only one panel, so it inherits the single-straight-line limit you saw in level00. Faced with a problem that no line can split — like XOR — no amount of reshuffling the speaking weights will ever settle the verdict; it swings forever.

## 3. Key concepts

### 3.1 What a perceptron computes

For inputs x1 and x2, a perceptron computes in two steps.

1. Weighted sum: `z = w1*x1 + w2*x2 + b`
2. Step function: output 1 if `z >= 0`, else 0

The formula is the same as level00's linear model. The difference is how the coefficients are found. In level00 we found the optimal line with a formula (least squares) and brute force; the perceptron finds it incrementally, looking at one data point at a time.

### 3.2 The perceptron learning rule

After seeing one data point (x, truth y) and producing prediction p, the update is a single line:

```
w <- w + lr * (y - p) * x
b <- b + lr * (y - p)
```

Here lr is the learning rate — the step size that decides how much to fix at once. `(y - p)` can only be 0 (correct), +1 (truth 1 predicted 0), or -1 (truth 0 predicted 1), so the rule reads: leave it alone when correct, push opposite to the direction of the mistake when wrong. One full pass over the data is called an epoch, and if a whole epoch goes by with no mistakes, training has converged.

### 3.3 The convergence theorem and its limit

The perceptron comes with one famous guarantee: if the data can be separated by a straight line (a hyperplane, in higher dimensions), this rule is guaranteed to converge within a finite number of corrections. AND and OR qualify. Conversely, for problems that no line can split — like XOR — convergence is impossible in principle. Under any weights at least one point stays wrong, and the update that tries to fix it makes another point wrong again, so the weights oscillate forever. In the hands-on section you'll watch that oscillation in actual numbers.

## 4. Hands-on — main.py

Run it like this:

```bash
cd lecture08_deep_learning_foundations/level01_perceptron_from_scratch
python3 main.py
```

The output has four stages.

- **[1]** Introduces the ~30-line `Perceptron` class and shows the initial weights (seeded random numbers).
- **[2]** Training AND: each epoch prints `w1, w2, b` and that epoch's misclassification count on one line. The moment misclassifications hit 0, it declares convergence and shows the final truth-table check.
- **[3]** Training OR: same procedure. You'll see it converge as fast as AND or faster.
- **[4]** Training XOR: even at the epoch cap (25), misclassifications never reach 0. The weights get pushed back and forth within each epoch only to end up where they started — trapped in an oscillation (a cycle) — and accuracy stays at 50%. It closes with the conclusion: "the structure has to change."

The spots to study in the code are the two update lines inside the `fit` method — the formula from section 3.2 transcribed directly into numpy. The early exit when `errors == 0` is the definition of "convergence" written as code. Follow how the weights move epoch by epoch in the output, and you can literally see that learning is just three numbers settling into place.

## 5. Try it yourself

1. **(Easy)** Change the learning rate `lr` from 0.1 to 0.5, then to 0.01, and rerun. How does the number of epochs to convergence change? Hint: a big step moves fast but can overshoot; a small step is stable but slow. This dataset is tiny, so the difference may be small.
2. **(Medium)** Change the seed in the `Perceptron` initializer from 7 to other values and run a few times. The final weights differ per run, yet the truth table is always fully correct. Confirm that "the correct line is not unique," then think about why we still fix the seed for reproducibility.
3. **(Challenge)** Add code after [3] that trains NAND (0 only when both inputs are 1). Then feed the outputs of your two hand-built perceptrons — the NAND result and the OR result — as new inputs to a third perceptron trained on AND, and the whole assembly computes XOR. Hint: it's the same structure as level00's hidden representation. Chain three perceptrons and you have a two-layer neural network.

## 6. Common mistakes

- **Suspecting a code bug when XOR fails.** Misclassifications refusing to drop to 0 is not a bug — it is this level's central result, the mathematical limit of a single-layer structure.
- **Confusing epochs with update counts.** One epoch is one full pass over all 4 data points; updates happen only for the points it got wrong.
- **Assuming a bigger learning rate always means faster.** It's less visible in this tiny example, but in practice a large learning rate is the main culprit behind divergence (values blowing up and never converging).
- **Initializing all weights to zero.** A single perceptron can still learn from zero, but in the multi-layer networks to come, all neurons would move identically and training breaks. That's why we build the habit of small random initialization starting now.
- **Forgetting the boundary case of the step function (`z = 0`).** Whether it's `>=` or `>` changes how points on the boundary are classified, so keep the implementation and the explanation consistent.

## Next level preview

The next level (level02_activation_functions) covers the activation functions that take the step function's place. You'll prove numerically, with matrix multiplication, that stacked layers without a nonlinear activation collapse into a single linear model, compare the shapes and slopes of sigmoid/tanh/ReLU in plots, and look at vanishing gradients and the dying-ReLU problem.
