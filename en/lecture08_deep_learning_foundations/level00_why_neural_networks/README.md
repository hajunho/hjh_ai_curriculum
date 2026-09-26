# Lecture 08 · Level 00 — Why Neural Networks Emerged

> Verify with actual numbers that there is a problem (XOR) a linear model can never solve, and understand the starting point of deep learning: "instead of humans writing the rules, let the machine learn the representation."

**Difficulty** ⭐ / **Prerequisites** none / **Estimated time** 40 min

## 1. Why learn this — the business view

The chatbots, image recognition, and translation tools you see in the news every day all run on neural networks. To use a tool well, it turns out to be surprisingly important to know *why the tool was invented*. Neural networks aren't used because they're fashionable — they emerged because the previous generation of tools, linear models, had problems they could not solve even in principle.

Business data is full of such problems. A relationship where cause flows straight to effect — "increase ad spend and revenue goes up" — is fine for a linear model. But relationships where conditions interact — "delivery orders explode only on days that are both a weekend *and* rainy," or "cutting the price with no inventory just increases complaints" — cannot be explained by a single straight line. In this level you'll verify the smallest such example, the XOR problem, in code, and learn the insight that let neural networks climb over that wall. Understand this one story and the perceptrons, activation functions, and multi-layer networks of the following levels all connect into a single thread.

## 2. Understanding through an analogy

Think of conditional formatting in Excel. "If revenue is at least 1,000,000 KRW, color it green" is easy — you're drawing a single threshold line. That is exactly what a linear model does: lay a ruler on the plane, draw one straight line, and declare "above this line passes, below it fails."

But what about this rule? "Send it back for review when exactly one of the two approvers agrees." Both agree: it passes. Both disagree: it's rejected. Only a split opinion triggers special handling. Plot the four cases as points on paper, and the two "review" (1) points sit diagonally across from each other — no straight line you can draw separates the two groups in one stroke. This is the XOR (exclusive or) problem.

What would a person do? Instead of staring at the raw table, you'd add a new column: "number of approvals." If review happens exactly when the count is 1, one threshold now splits the cases cleanly. So the problem was never hard — the coordinates you viewed the data in were bad. This is the core insight of neural networks: rather than having humans hand-design good coordinates (representations), let the machine find them in the data by itself — this is called representation learning.

## 3. Key concepts

### 3.1 Linear models and decision boundaries

With two inputs (x1, x2), a linear model computes `score = w1*x1 + w2*x2 + b` and classifies as 1 if the score exceeds a threshold (e.g., 0), else 0. w1 and w2 are weights, b is the bias. The set of points where the score exactly equals the threshold forms a straight line on the plane, called the decision boundary. A linear model's ability rests entirely on one question: "can a single straight line separate this?"

### 3.2 AND, OR, XOR

For the four combinations of inputs 0 and 1, the three rules tabulate as follows.

| x1 | x2 | AND | OR | XOR |
|----|----|-----|----|-----|
| 0  | 0  | 0   | 0  | 0   |
| 0  | 1  | 0   | 1  | 1   |
| 1  | 0  | 0   | 1  | 1   |
| 1  | 1  | 1   | 1  | 0   |

For AND and OR, the points labeled 1 cluster in one corner, so a single line separates them. For XOR, the 1-points (0,1) and (1,0) face each other diagonally: whatever line you draw, at least one of the four points is always wrong. The accuracy ceiling of a linear model on XOR is therefore 75%. This is not a matter of effort or data volume but a mathematical limit — and the observation, made around 1969, sent early neural-network research into a long winter.

### 3.3 Change the representation and it's solved

In words, XOR is "OR but not AND." So instead of the original inputs (x1, x2), build two hidden representations — h1 = OR(x1,x2) and h2 = AND(x1,x2) — and move to (h1, h2) coordinates. XOR becomes simply `h1 - h2`, perfectly separable by one line. A multi-layer neural network is exactly this coordinate transformation (the hidden layer) fused with the final line-drawing (the output layer), and training adjusts the transformation itself to fit the data. This is the point where we crossed from the era of humans designing features to the era of models learning them.

## 4. Hands-on — main.py

Run it like this:

```bash
cd lecture08_deep_learning_foundations/level00_why_neural_networks
python3 main.py
```

The output proceeds in five stages.

- **[1]** Builds and prints the truth-table data for AND, OR, XOR.
- **[2]** Fits the best linear model for each problem by least squares and prints the accuracy. AND and OR reach 100%; XOR stays at 75% or below.
- **[3]** In case least squares missed a better line, it brute-force searches hundreds of thousands of weight/bias combinations. XOR's best accuracy still stops exactly at 75% — evidence that "a better line does not exist."
- **[4]** After switching to hand-designed hidden representations h1 (acting as OR) and h2 (acting as AND), the same linear model applied again reaches 100% accuracy on XOR.
- **[5]** Summarizes the conclusion: the model wasn't weak, the representation was bad — and learning representations is the reason neural networks exist.

There are two key spots in the code. `best_linear_accuracy` uses numpy broadcasting to evaluate predictions for the entire (w1, w2, b) grid at once and finds the best accuracy. `hidden_features` performs the coordinate transformation in just two lines — and those two lines do exactly the job of the two hidden-layer neurons you'll build later.

## 5. Try it yourself

1. **(Easy)** Change the XOR target array in main.py to NAND (0 only when both are 1) and run it. Does a linear model reach 100%? Hint: sketch on paper where NAND's 1-points sit on the plane and you can predict the answer.
2. **(Medium)** Widen the brute-force grid to -10..10 and increase the grid density. Does XOR accuracy ever exceed 75%? Hint: no matter how far you widen the range, the result doesn't change. Explain why using the table in section 3.2.
3. **(Challenge)** Could you solve XOR with only a single hidden representation, `h = x1 + x2`? Write down the correct label when h is 0, 1, and 2, and reason about whether one threshold can separate them. Hint: if you conclude you'd need two thresholds, that's correct — and it connects to why two hidden neurons are needed.

## 6. Common mistakes

- **Thinking "with more data, a linear model could solve XOR too."** XOR's limit is a matter of model structure, not sample size. With just four points — the entire dataset — the ceiling is already 75%.
- **Seeing 75% accuracy and shrugging "that's pretty good."** In a two-class problem, 75% here means one case out of four is always wrong. At work, too, build the habit of checking *which* cases are wrong instead of staring at the accuracy number alone.
- **Treating the hidden representation as a "trick."** The coordinate transformation is not a hack — it *is* the principle of neural networks. This time a human inserted it by hand; from the next levels on, the machine learns it.
- **Confusing the boundary handling of `>=` vs `>`.** Points whose score exactly equals the threshold can flip the result depending on which side they're assigned to, so always check the direction of the inequality in the code.

## Next level preview

In the next level (level01_perceptron_from_scratch) you'll build the smallest unit of a neural network, the perceptron, in about 30 lines of numpy. Today we found the "best line" via formulas and brute force; next time you'll implement a learning rule where the model nudges its weights a little each time it's wrong, finding the line by itself — and you'll watch it converge on AND/OR but never converge on XOR.
