"""
Deliberately cause overfitting, then compare the effect of three prescriptions.
On the subscription-churn data we give a large MLP only 120 training rows and
let it memorize, then check — with a curve PNG and a table — how much
(1) dropout, (2) weight decay, and (3) early stopping each recover
validation (valid) performance.
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
N_EPOCHS = 600


def load_data():
    """Use 'a mere 120' of churn_table's 2000 rows for training -> induce overfitting.
    The remaining 1880 rows are for validation, making the report card very stable."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, va = idx[:120], idx[120:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8   # training-set statistics only
    X = (X - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(y[tr]), t(X[va]), t(y[va])


def build_model(dropout=0.0):
    """An MLP deliberately large for the data (120 rows): about 4.7k parameters = a memorization inducer."""
    layers = [nn.Linear(6, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))       # neurons randomly absent during training
    layers += [nn.Linear(64, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers)


def train(name, X_tr, y_tr, X_va, y_va, dropout=0.0, weight_decay=0.0):
    """Full-batch Adam training. Returns the per-epoch valid loss/accuracy history."""
    torch.manual_seed(1)                         # same initial weights for every condition
    model = build_model(dropout)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"tr_loss": [], "va_loss": [], "va_acc": []}
    for _ in range(N_EPOCHS):
        model.train()                            # dropout only operates in train mode
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        model.eval()                             # at evaluation, no absences — everyone reports for duty
        with torch.no_grad():
            logits = model(X_va)
            hist["va_loss"].append(loss_fn(logits, y_va).item())
            hist["va_acc"].append(((logits > 0) == y_va.bool()).float().mean().item())
        hist["tr_loss"].append(loss.item())
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(X_tr) > 0) == y_tr.bool()).float().mean().item()
    print(f"    {name:26s}: train accuracy {tr_acc:.3f} | final valid accuracy {hist['va_acc'][-1]:.3f}")
    return hist


def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] Experiment design: 120 training rows vs ~4,700 parameters — the conditions for memorization (overfitting)")
    X_tr, y_tr, X_va, y_va = load_data()
    print(f"    Subscription-churn binary classification, train {len(X_tr)} / valid {len(X_va)}, {N_EPOCHS} epochs full-batch\n")

    print("[2] Training per condition (same seed, same initial weights)")
    runs = {
        "baseline": train("baseline (unprotected)", X_tr, y_tr, X_va, y_va),
        "dropout": train("dropout p=0.5", X_tr, y_tr, X_va, y_va, dropout=0.5),
        "weight_decay": train("weight decay 3e-2", X_tr, y_tr, X_va, y_va, weight_decay=3e-2),
    }

    print("\n[3] Early stopping — what if we had stopped the baseline at its 'best moment'?")
    va = runs["baseline"]["va_loss"]
    best_epoch = int(np.argmin(va))              # epoch of lowest valid loss
    es_acc = runs["baseline"]["va_acc"][best_epoch]
    print(f"    valid loss minimum: epoch {best_epoch + 1} (out of 600!)")
    print(f"    valid accuracy at that point {es_acc:.3f} vs baseline run to the end {runs['baseline']['va_acc'][-1]:.3f}\n")

    print("[4] Final comparison table (valid accuracy is the report card)")
    rows = [("unprotected (all 600 epochs)", runs["baseline"]["va_acc"][-1]),
            ("dropout 0.5", runs["dropout"]["va_acc"][-1]),
            ("weight decay 3e-2", runs["weight_decay"]["va_acc"][-1]),
            (f"early stopping (epoch {best_epoch + 1})", es_acc)]
    print(f"    {'prescription':28s} {'valid accuracy':>14s}")
    for name, acc in rows:
        print(f"    {name:28s} {acc:>10.3f}")
    assert max(r[1] for r in rows[1:]) > rows[0][1] + 0.02, "The prescriptions' effect did not reproduce"

    print("\n[5] Saving the curve PNG")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, N_EPOCHS + 1)
    colors = {"baseline": "#d62728", "dropout": "#1f77b4", "weight_decay": "#2ca02c"}
    labels = {"baseline": "baseline", "dropout": "dropout 0.5", "weight_decay": "weight decay 3e-2"}
    for key, hist in runs.items():
        axes[0].plot(epochs, hist["va_loss"], label=f"{labels[key]} (valid)", color=colors[key])
        axes[1].plot(epochs, hist["va_acc"], label=labels[key], color=colors[key])
    axes[0].plot(epochs, runs["baseline"]["tr_loss"], "--", color="#d62728",
                 alpha=0.5, label="baseline (train)")
    axes[0].axvline(best_epoch + 1, color="gray", ls=":", label="early stop point")
    axes[0].set_title("Loss: train memorizes, valid gets worse")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend(fontsize=8)
    axes[1].axvline(best_epoch + 1, color="gray", ls=":")
    axes[1].set_title("Valid accuracy by remedy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("valid accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "regularization_compare.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"    saved: {png_path}")

    print("\n[6] Recap: overfitting prescriptions are all 'sabotage of memorization'.")
    print("    Absence training (dropout), a tax on large weights (weight decay), stopping at peak grades (early stopping).")
    print("    In practice you combine all three — and the best prescription is always 'collect more data'.")


if __name__ == "__main__":
    main()
