# Lecture 08 · Level 02 — Activation Functions

> Prove numerically, with matrix multiplication, that stacking layers without a nonlinear activation is exactly the same as one linear model; compare the shapes and slopes of sigmoid/tanh/ReLU in plots; and understand the vanishing-gradient and dying-ReLU problems.

**Difficulty** ⭐⭐ / **Prerequisites** level01_perceptron_from_scratch / **Estimated time** 50 min

## 1. Why learn this — the business view

In levels 00–01 we established two things: a single line (a linear model) can't solve entangled problems like XOR, and a coordinate transformation (a hidden representation) solves them. The natural next thought is, "so why not stack several linear transformations?" This level's first conclusion is blunt: stack 100 linear layers and mathematically you still have exactly one linear layer. Only when you insert a nonlinear function — an activation function — between the layers does "depth" start to mean anything.

At work this knowledge pays off in two situations. First, when using deep-learning tools, names like ReLU and sigmoid appear all over settings screens and code samples; knowing each one's character, you'll understand why the defaults are what they are and stop changing them casually. Second, when someone reports "the model won't train," you'll recognize diagnostic terms like vanishing gradient and dead ReLU as candidate causes. What you learn today is also the story of a parts swap that genuinely changed the course of deep-learning history.

## 2. Understanding through an analogy

Think of an approval chain: a three-step sign-off by the staff member, the team lead, and the division head. Now suppose all three do only mechanical work — "multiply the incoming score by some factor, add something, pass it up." If the staffer doubles it, the team lead halves it, and the division head triples it, the combined effect of three steps is one multiplication: "times 3." Three stamps were applied, but judgment happened only once. That is a neural network made of nothing but linear layers.

For the sign-off chain to mean something, each step needs a judgment: "below the bar, reject it here," "above a certain amount, route it to a separate track." The moment a step responds differently depending on conditions, the three-step chain stops compressing into a single multiplication and becomes a genuine three steps. The activation function is that judgment. ReLU is the decisive reviewer who "cuts every negative score to zero"; sigmoid is the cautious reporter who restates any score as a degree of confidence between 0 and 1.

Vanishing gradients can be pictured like this. The sigmoid reviewer's answer, whether the score is very large or very small, hardens into "close to 1" or "close to 0." In that state, nudging the input barely changes the response — so to the question "which direction should we adjust to improve?" (the gradient), you get effectively no answer. Feedback shrinking as it climbs back down through the layers, never reaching the bottom ones — that is the vanishing gradient.

## 3. Key concepts

### 3.1 Why nonlinearity is necessary — a composition of linear maps is linear

In matrix form it's one line. If three layers are the matrix multiplications W1, W2, W3, the whole is `y = W3 @ (W2 @ (W1 @ x)) = (W3 @ W2 @ W1) @ x`. Matrix multiplication is associative, so `W3 @ W2 @ W1` can be premultiplied into a single matrix W, and the 3-layer network is exactly a 1-layer network. Insert a nonlinear function f between the layers and you get `W3 @ f(W2 @ f(W1 @ x))` — the compression becomes impossible, and only then does depth create expressive power. In the hands-on section you compare both cases in actual numbers.

### 3.2 The main activation functions compared

| Function | Formula | Output range | Character |
|------|-----|-----------|------|
| sigmoid | 1/(1+e^-z) | (0, 1) | Interpretable as a probability. Saturates at both ends |
| tanh | (e^z - e^-z)/(e^z + e^-z) | (-1, 1) | Zero-centered, so better than sigmoid. Also saturates |
| ReLU | max(0, z) | [0, ∞) | Cheap to compute, slope 1 on the positive side. The modern default |
| Leaky ReLU | 0.01z for z<0 | (-∞, ∞) | Keeps a small slope on the negative side too |

Saturation is the flattening of the output as the input grows large or small. Sigmoid's slope peaks at 0.25 (at z=0) and by z=5 has already plunged to about 0.0066.

### 3.3 Vanishing gradients and dead ReLUs

Training works by "passing the output error backwards, multiplying by each layer's slope along the way" (the full mechanics come in later levels). Put sigmoid — max slope 0.25 — into 10 layers, and the signal that gets through is at best 0.25 to the 10th power, roughly one millionth. The lower layers effectively stop learning. This is the core reason deep networks failed for so long, and ReLU — whose slope is always 1 on the positive side — largely relieved the problem and became the standard.

ReLU has its own weakness, though. If a neuron's weighted sum ends up negative across the entire data range, its output is 0 and its gradient is 0, so it is never updated again. That is a dead ReLU. Leaky ReLU, which keeps a small slope like 0.01 on the negative side, is a variant designed to reduce that risk.

## 4. Hands-on — main.py

Run it like this:

```bash
cd lecture08_deep_learning_foundations/level02_activation_functions
python3 main.py
```

The output has five stages.

- **[1]** Draws the shapes of sigmoid, tanh, ReLU, and Leaky ReLU in a 2x2 figure saved to `outputs/activations.png` and prints the path. Each panel shows the function (solid) and its derivative (dashed).
- **[2]** Saves `outputs/gradients.png`, overlaying just the four derivatives on one axis. You can compare how sigmoid/tanh gradients flatten to 0 at both ends while ReLU's gradient holds at 1 on the positive side.
- **[3]** Puts numbers on saturation: how far sigmoid'(0)=0.25 falls by sigmoid'(5), and what that value becomes multiplied across 10 layers — making the vanishing gradient tangible — plus the numeric difference between a dead ReLU and Leaky ReLU.
- **[4]** The key proof. It prints the maximum error between composing three linear layers with random matrices W1, W2, W3 and applying the single premultiplied matrix `W3 @ W2 @ W1` to the same input. The error is effectively zero (floating-point precision level).
- **[5]** Shows that inserting ReLU between the layers produces something no single linear matrix can imitate: the outputs differ substantially for the same input, and the basic property of linearity, `f(a+b) = f(a)+f(b)`, breaks — confirmed numerically.

The key code lives in `linear_stack_vs_single` (the associativity check) and `relu_breaks_linearity` (the broken-additivity check). The plotting code assigns a fixed color per function, and the derivatives are computed from analytic formulas, not approximations.

## 5. Try it yourself

1. **(Easy)** Change the x range in main.py from (-6, 6) to (-20, 20) and re-save the figures. The saturation of sigmoid and tanh looks far more dramatic. Think about why widening the range doesn't change the character of ReLU's shape.
2. **(Medium)** Change the "10 layers" in stage [3] to 30 and compute the sigmoid gradient product. How small does it get? Compare with the same calculation for ReLU (slope 1 on the positive side). Hint: 1 multiplied any number of times is still 1. That is why ReLU is beloved in deep networks.
3. **(Challenge)** Add a Leaky ReLU variant with the negative-side slope raised from 0.01 to 0.2 and plot it alongside. Then explain what function you get when the negative slope is 1.0, and whether the proof in [4] applies to that function. Hint: with slope 1 you get f(z)=z, i.e., the same as having no activation at all.

## 6. Common mistakes

- **Believing "we stacked many layers, so the model is complex."** Without activation functions, 100 layers are 1 layer. Depth only means something together with nonlinearity.
- **Using sigmoid as the default in hidden layers.** Sigmoid is still useful in the output layer for expressing probabilities, but in deep hidden layers it invites vanishing gradients. The hidden-layer default is the ReLU family.
- **Mistaking dead ReLUs for a data problem.** Some neurons outputting a constant 0 during training is common, and it worsens with too-large learning rates or bad initialization. Leaky ReLU or a learning-rate adjustment is the prescription.
- **Checking derivatives only by numerical differentiation.** For functions with a kink (z=0), like ReLU, numerical differentiation gives ambiguous values at that point. The safe habit is to cross-check against analytic formulas.
- **Reading a floating-point error of 1e-15 as "different."** The maximum error in [4] may come out as a tiny nonzero number — that's the limit of computer decimal arithmetic, not a sign that the two computations differ.

## Next level preview

The next level (level03 — Loss Functions and Gradient Descent) covers the loss function, which measures "how wrong we are" as a single number, and gradient descent, which nudges the weights in the direction that shrinks that loss. That's where today's gradient plots reveal why they matter so much — and where "learning" turns out to be, in the end, finding the way downhill.
