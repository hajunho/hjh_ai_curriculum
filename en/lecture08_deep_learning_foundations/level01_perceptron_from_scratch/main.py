"""
Lecture 08 · Level 01 — Building a Perceptron by Hand
Implements a perceptron (one artificial neuron) in about 30 lines of numpy
and solves AND/OR with the learning rule "if wrong, nudge the weights."
Each epoch prints how the weights, bias, and misclassification count change,
and finally demonstrates that on XOR it never manages to converge.
"""

import numpy as np

np.random.seed(42)  # Global reproducibility (the class uses its own seeded Generator internally)

# Truth-table data: four input combinations and the answers per problem
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


class Perceptron:
    """Perceptron: a single neuron that outputs 1 when the weighted vote (weighted sum) of its inputs clears a threshold, else 0."""

    def __init__(self, n_inputs, lr=0.1, seed=7):
        rng = np.random.default_rng(seed)          # Fixed seed -> same initial values every run
        self.w = rng.normal(0.0, 0.1, size=n_inputs)  # Start the speaking weights as small random numbers
        self.b = 0.0                                # Bias that shifts the threshold
        self.lr = lr                                # Learning rate: the step size of each fix

    def predict(self, x):
        # Output 1 when the weighted sum is >= 0 (step function)
        return int(np.dot(self.w, x) + self.b >= 0.0)

    def fit(self, features, targets, max_epochs=20, verbose=True):
        """Perceptron learning rule. Returns the epoch count on convergence, or None on failure, plus the per-epoch error history."""
        error_history = []
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for x, y in zip(features, targets):
                pred = self.predict(x)
                update = self.lr * (y - pred)       # 0 when correct, +-lr when wrong
                if update != 0.0:
                    self.w = self.w + update * x    # Adjust speaking weights opposite to the mistake
                    self.b = self.b + update
                    errors += 1
            error_history.append(errors)
            if verbose:
                print(f"    epoch {epoch:2d}: w1={self.w[0]:+.3f}, w2={self.w[1]:+.3f}, "
                      f"b={self.b:+.3f}, misclassified {errors}/4")
            if errors == 0:                          # A full pass with no mistakes = convergence
                return epoch, error_history
        return None, error_history


def show_truth_table(model, features, targets):
    """Validate the trained model on the full truth table and print the result."""
    correct = 0
    for x, y in zip(features, targets):
        pred = model.predict(x)
        mark = "O" if pred == y else "X"
        correct += int(pred == y)
        print(f"      input ({int(x[0])}, {int(x[1])}) -> pred {pred}, truth {y}  [{mark}]")
    print(f"      accuracy {correct}/4 ({correct / 4 * 100:.0f}%)")


def train_and_report(step_no, name, max_epochs=20):
    print(f"[{step_no}] Training {name} — watching the weights and misclassification count each epoch")
    model = Perceptron(n_inputs=2, lr=0.1, seed=7)
    print(f"    initial : w1={model.w[0]:+.3f}, w2={model.w[1]:+.3f}, b={model.b:+.3f}")
    converged, history = model.fit(X, TARGETS[name], max_epochs=max_epochs)
    if converged is not None:
        print(f"    => Converged at epoch {converged} (0 misclassifications). Final truth-table check:")
    else:
        print(f"    => Failed to converge in {max_epochs} epochs. Misclassifications per epoch: {history}")
        print("       The error count never drops to 0. Within each epoch the weights get pushed")
        print("       around only to return to where they started — trapped in an oscillation (cycle). Final truth-table check:")
    show_truth_table(model, X, TARGETS[name])
    print()
    return converged


def main():
    print("[1] Meet the perceptron")
    print("    One neuron = a weighted vote: z = w1*x1 + w2*x2 + b, output 1 if z >= 0 (step function)")
    print("    Learning rule = fix it when wrong: w <- w + lr*(truth-pred)*x, b <- b + lr*(truth-pred)")
    print("    Below we train the three problems AND, OR, XOR from the same initial values (seed 7).")
    print()

    train_and_report(2, "AND")
    train_and_report(3, "OR")
    converged = train_and_report(4, "XOR", max_epochs=25)

    print("[5] Conclusion")
    if converged is None:
        print("    - AND/OR are separable by a line, so the perceptron converged within a finite number of epochs.")
        print("    - XOR cannot be split by a line (see level00), so the weights oscillate forever.")
        print("    - The fix is not training longer but changing the structure: stacking neurons")
        print("      into layers so they build representations. The prerequisite for that is the next level's activation functions.")


if __name__ == "__main__":
    main()
