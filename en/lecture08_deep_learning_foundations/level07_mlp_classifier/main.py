"""
Build your own MLP (multi-layer perceptron) classifier with nn.Module.
Split the subscription-churn data (churn_table) into train/test and train
(1) logistic regression (= a zero-hidden-layer network) and (2) an MLP (6-16-8-1)
under identical conditions in torch, comparing accuracy, precision, and recall.
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def load_data():
    """Convert churn_table into standardized tensors. customer_id is a meaningless column, so it's excluded."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)

    # Train/test split (shuffle, then 75:25)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.75)
    tr, te = idx[:cut], idx[cut:]

    # Standardization: mean/std must be computed 'from the training data only' (prevents leakage)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    X = (X - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(y[tr]), t(X[te]), t(y[te])


class LogisticRegression(nn.Module):
    """A zero-hidden-layer network = logistic regression (the model from lecture06 level05)."""

    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class ChurnMLP(nn.Module):
    """An MLP with two hidden layers (16, 8). Parts declared in __init__, assembled in forward."""

    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train(model, X_tr, y_tr, n_epochs=800, lr=0.01):
    """Full-batch Adam training. The data is small; batching comes in the next level."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        if epoch in (1, 200, 800):
            print(f"      epoch {epoch:3d}: train loss = {loss.item():.4f}")
    return model


def evaluate(model, X_te, y_te, threshold=0.5):
    """Test performance: accuracy plus the churn (1) class's precision/recall."""
    model.eval()
    with torch.no_grad():                       # derivative ledger OFF during evaluation
        p = model(X_te)
    pred = (p > threshold).float()
    acc = (pred == y_te).float().mean().item()
    tp = ((pred == 1) & (y_te == 1)).sum().item()
    fp = ((pred == 1) & (y_te == 0)).sum().item()
    fn = ((pred == 0) & (y_te == 1)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return acc, prec, rec


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("[1] Data: subscription churn churn_table (n=2000, churn rate about 18%)")
    X_tr, y_tr, X_te, y_te = load_data()
    print(f"    train {len(X_tr)} rows / test {len(X_te)} rows, {X_tr.shape[1]} features (customer_id excluded, standardized)")
    print(f"    test churn rate: {y_te.mean().item():.1%} -> predicting 'everyone stays' already gets accuracy ~{1-y_te.mean().item():.0%}\n")

    print("[2] Model A — logistic regression (a zero-hidden-layer network)")
    logreg = LogisticRegression(len(FEATURES))
    print(f"    parameter count: {count_params(logreg)}")
    train(logreg, X_tr, y_tr)

    print("\n[3] Model B — MLP (6-16-8-1, ReLU)")
    mlp = ChurnMLP(len(FEATURES))
    print(f"    parameter count: {count_params(mlp)}")
    train(mlp, X_tr, y_tr)

    print("\n[4] Test performance comparison (threshold 0.5, churn = positive)")
    print(f"    {'model':16s} {'accuracy':>8s} {'precision':>9s} {'recall':>8s}")
    results = {}
    for name, model in [("LogisticReg", logreg), ("MLP", mlp)]:
        acc, prec, rec = evaluate(model, X_te, y_te)
        results[name] = acc
        print(f"    {name:16s} {acc:8.3f} {prec:9.3f} {rec:8.3f}")

    print("\n[5] What if we lower the threshold 0.5 -> 0.3? (MLP)")
    acc3, prec3, rec3 = evaluate(mlp, X_te, y_te, threshold=0.3)
    print(f"    {'MLP(th=0.3)':16s} {acc3:8.3f} {prec3:9.3f} {rec3:8.3f}")
    print("    We give up a little accuracy in exchange for a big cut in 'missed churners' (recall).")

    print("\n[6] Interpretation")
    print("    - This dataset's churn rule was designed to be 'nearly linear', so the two models are similar.")
    print("    - Lesson 1: deep learning does not always win. For tabular data, start with simple models.")
    print("    - Lesson 2: still, the MLP has room to learn feature interactions automatically")
    print("      (e.g., plan changes x a spike in support calls), so the gap widens as relationships get nonlinear.")
    print("    - Lesson 3: if recall (missed churners) is low, lowering the 0.5 threshold is a legitimate practical option.")


if __name__ == "__main__":
    main()
