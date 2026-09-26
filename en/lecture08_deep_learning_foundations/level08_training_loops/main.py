"""
Learn Dataset/DataLoader and the standard training loop.
Wraps the subscription-churn data (churn_table) in a custom Dataset,
draws mini-batches with a DataLoader, implements the production-standard
train/validate-per-epoch template, then saves the train/valid loss curves as a PNG.
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
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


class ChurnDataset(Dataset):
    """Dataset = the 'warehouse librarian'. It only has to answer two questions:
    len (how many items total?) and getitem (give me item i)."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_datasets():
    """churn_table -> standardization -> two Datasets, train/valid."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8   # statistics from the training set only
    X = (X - mu) / sd
    return ChurnDataset(X[tr], y[tr]), ChurnDataset(X[va], y[va])


def build_model():
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(),
                         nn.Linear(16, 8), nn.ReLU(),
                         nn.Linear(8, 1))     # no Sigmoid: we use BCEWithLogitsLoss


def run_epoch(model, loader, loss_fn, opt=None):
    """One epoch = one pass over every plate (batch) in the warehouse. With opt: train; without: evaluate."""
    training = opt is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for xb, yb in loader:                     # the DataLoader serves the plates one by one
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                opt.zero_grad()                   # the standard 5-step loop (level06)
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(xb)   # weighted average by batch size
            total_correct += ((logits > 0) == yb.bool()).sum().item()
            total_n += len(xb)
    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] Data preparation — two Datasets (train/valid)")
    train_ds, valid_ds = load_datasets()
    print(f"    train {len(train_ds)} rows / valid {len(valid_ds)} rows")
    xb0, yb0 = train_ds[0]
    print(f"    train_ds[0] -> feature shape {tuple(xb0.shape)}, label {yb0.item():.0f}\n")

    print("[2] DataLoader — batch size 64, shuffled every epoch")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(0))
    valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False)  # no shuffle needed for evaluation
    n_steps = len(train_loader)
    print(f"    1 epoch = {len(train_ds)} rows / 64 = {n_steps} steps (batches)")
    print("    Terms: step = processing 1 batch, epoch = one pass over all the data\n")

    print("[3] Training — the standard template (one train pass + valid grading per epoch)")
    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()              # the stable fused version of Sigmoid+BCE
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    history = {"train": [], "valid": [], "acc": []}
    n_epochs = 40
    for epoch in range(1, n_epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, loss_fn, opt)
        va_loss, va_acc = run_epoch(model, valid_loader, loss_fn)
        history["train"].append(tr_loss)
        history["valid"].append(va_loss)
        history["acc"].append(va_acc)
        if epoch in (1, 5, 10, 20, 30, 40):
            print(f"    epoch {epoch:2d}: train loss {tr_loss:.4f} | "
                  f"valid loss {va_loss:.4f} | valid accuracy {va_acc:.3f}")

    print("\n[4] Saving the learning curve")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, n_epochs + 1)
    ax.plot(epochs, history["train"], label="train loss", color="#1f77b4")
    ax.plot(epochs, history["valid"], label="valid loss", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training curve (churn MLP, batch=64)")
    ax.legend()
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "training_curve.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"    saved: {png_path}")

    print("\n[5] How to read a learning curve")
    print("    - both curves falling together      -> training is healthy")
    print("    - only train falls, valid rebounds  -> overfitting begins (level09's topic)")
    print("    - both flat and high                -> underfitting: revisit model/epochs/learning rate")
    print(f"    This run: final train {history['train'][-1]:.4f}, valid {history['valid'][-1]:.4f}, "
          f"valid accuracy {history['acc'][-1]:.3f}")


if __name__ == "__main__":
    main()
