# Lecture 08 · Level 07 — Building an MLP Classifier

> Assemble your own multi-layer perceptron (MLP) with nn.Module and pit it head-to-head against logistic regression on subscription-churn data. The result may not be what you expect.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level06 (autograd and optimizers), lecture06 level05 (logistic regression) / **Estimated time** 55 min

## 1. Why learn this — the business view

Here, for the first time, we apply deep learning to **business data** rather than a toy problem (XOR). Churn prediction for a subscription service is a classic real-world task, directly tied to how marketing budget is allocated. At the same time, this level plants an important dose of realism — on tabular data, deep learning does not always beat plain logistic regression. In the meeting where someone says "let's adopt deep learning too," you become the person who can say, "let's run a comparison against a simple model first." And the standard skeleton of production PyTorch code — writing an `nn.Module` class — is the base grammar for every lecture that follows (CNNs, transformers).

## 2. Understanding through an analogy

**nn.Module = LEGO assembly instructions.** A model class has two parts.

- `__init__`: **opening the parts box.** You declare the blocks you'll need (Linear layers, ReLU, Sigmoid) and set them in the box. The weights inside each Linear block are created automatically at this point, and `model.parameters()` hands them all to the optimizer at once.
- `forward`: **the assembly flowchart.** You write the order in which the input passes through the blocks. Call `model(X)` and PyTorch runs forward for you.

**Logistic regression = a neural network with zero hidden layers.** The logistic regression from lecture06 is really `Linear(6,1) + Sigmoid` — the smallest possible neural network. A neural network just bolts hidden layers (an automatic feature-engineering team) *in front of* logistic regression. So this showdown is also an experiment: "does hiring a new feature-engineering team improve results?"

## 3. Key concepts

### 3-1. A feel for hidden-layer design

- **Width**: a funnel that narrows — 6 inputs → 16 → 8 → 1 — is a safe starting shape.
- **Depth**: for tabular data, 1–2 hidden layers are usually enough. More layers add expressiveness but also overfitting and training difficulty (level09).
- **Parameter-count intuition**: logistic regression 7 parameters vs the MLP's 257. 37x the parameters does not mean 37x the performance.

### 3-2. Standardization and data leakage

When features live on wildly different scales (tenure 1–60 months, fees 9,900–29,900 KRW), gradient descent staggers. So we standardize to mean 0, standard deviation 1. But **the mean and standard deviation must be computed from the training data only.** Mixing in test data is like seeing the exam in advance (data leakage). Dropping `customer_id` from the features is the same instinct — a meaningless identifier is nothing but overfitting fuel.

### 3-3. model.eval() and no_grad

- `model.eval()`: switches to "it's exam time now" mode, turning off training-only behaviors such as dropout (level09).
- `torch.no_grad()`: turns off the derivative ledger during evaluation, saving memory and time.

They do different jobs, so the standard is to use **both** during evaluation.

### 3-4. Reading the scorecard on imbalanced data

With a churn rate of 13–18%, predicting "everyone stays" already scores around 85% accuracy. So never look at accuracy alone: check the churn (positive) class's **precision** (of those flagged as churn, how many really are) and **recall** (of the real churners, how many were caught). It's the "accuracy trap" from lecture06 level06 all over again.

### 3-5. So what should you use on tabular data?

The practical summary: for tabular data, the first-line candidates are logistic regression and tree ensembles (lecture06's random forests, boosting). Neural networks gain a clear edge on tables when (1) feature interactions are complex and data is plentiful, (2) you combine the table with unstructured data such as images or text, or (3) you need embeddings to compress ID-like features (product IDs and the like). The healthy process is: "build a baseline with a simple model first; the neural network is adopted only if it beats that baseline."

## 4. Hands-on — main.py

Run:

```bash
cd lecture08_deep_learning_foundations/level07_mlp_classifier
python3 main.py
```

Output, in order:

- **[1]** Splits the 2000-row churn_table 75:25 and standardizes it. Prints the "predict everyone stays" baseline (~87%) first.
- **[2]–[3]** Trains logistic regression (7 parameters) and the MLP (257) under identical conditions (Adam, 800 epochs, full batch). The MLP's train loss goes lower (naturally — it has more expressive power).
- **[4]** Test performance: both accuracies sit near the baseline (0.85–0.87), and both recalls are low. The train-loss advantage does not translate into a test advantage — this is the key observation.
- **[5]** Lowering the MLP's threshold from 0.5 → 0.3 gives up a little accuracy and jumps recall from 0.09 → 0.24. You can tune to the business goal (don't miss churning customers) without retraining the model.
- **[6]** Interpretation: this dataset's churn rule is nearly linear, so the MLP's advantage is small. "Deep learning doesn't always win" — confirmed with data.

The part of the code to study is the `ChurnMLP` class: parts declared in `__init__`, assembled in `forward`. This two-slot structure is the skeleton of every PyTorch model.

## 5. Try it yourself

1. **(Easy)** Grow the MLP's hidden sizes from (16, 8) to (64, 32). How do the train loss and the test performance each change? *Hint: train loss will clearly drop, but the test likely won't. That's the smell of overfitting.*
2. **(Medium)** Turn off standardization (set µ=0, σ=1) and train. How does the loss curve change? *Hint: the fee feature (in the tens of thousands) dominates the gradient and convergence gets visibly worse.*
3. **(Challenge)** Sweep the threshold from 0.1 to 0.9 and build a precision/recall table. Under the assumption "a missed churner costs 50,000 KRW, a coupon send costs 5,000 KRW," find the threshold that minimizes total cost. *Hint: just loop evaluate(threshold=...) in a for statement.*

## 6. Common mistakes

- **Calling forward directly.** Call `model(X)`, not `model.forward(X)`. The former correctly runs hooks and other internal machinery.
- **Computing standardization statistics on all the data.** A common leak. Always compute on the training set only and apply to the test set.
- **Mixing BCELoss with logits.** If your model ends in Sigmoid, use BCELoss; if not, BCEWithLogitsLoss. Mix them up and performance quietly degrades (the latter is numerically more stable and preferred in practice).
- **Declaring a winner by train loss alone.** The more expressive model almost always wins on train loss. The verdict must come from test/validation performance.

## Next level preview

Right now we trained all 2000 rows at once (full batch), but millions of rows won't fit in memory. In level08 you'll build the standard training loop that portions data onto plates (mini-batches) with Dataset/DataLoader, and learn to read train/valid loss curves to answer "is training going well?"
