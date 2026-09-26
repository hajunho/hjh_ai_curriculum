"""
Dataset/DataLoader と標準的な学習ループを学びます。
サブスク解約データ(churn_table)をカスタム Dataset で包み、
DataLoader でミニバッチを取り出しながらエポック単位で学習/検証する
実務標準のテンプレートを実装したうえで、train/valid の loss 曲線を PNG に保存します。
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
    """Dataset = 「倉庫の司書」。答えるべき質問は 2 つだけです:
    len(全部で何件?) と getitem(i 番目の 1 件をください)。"""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_datasets():
    """churn_table -> 標準化 -> train/valid の Dataset を 2 つ。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8   # 統計量は学習セットだけで計算
    X = (X - mu) / sd
    return ChurnDataset(X[tr], y[tr]), ChurnDataset(X[va], y[va])


def build_model():
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(),
                         nn.Linear(16, 8), nn.ReLU(),
                         nn.Linear(8, 1))     # Sigmoid なし: BCEWithLogitsLoss を使うため


def run_epoch(model, loader, loss_fn, opt=None):
    """1 エポック = 倉庫のすべての皿(バッチ)を一周。opt があれば学習、なければ評価。"""
    training = opt is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for xb, yb in loader:                     # DataLoader が皿を 1 枚ずつ配膳
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                opt.zero_grad()                   # 標準の 5 段階ループ (level06)
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(xb)   # バッチサイズで重み付き平均
            total_correct += ((logits > 0) == yb.bool()).sum().item()
            total_n += len(xb)
    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] データの準備 — Dataset を 2 つ (train/valid)")
    train_ds, valid_ds = load_datasets()
    print(f"    train {len(train_ds)}件 / valid {len(valid_ds)}件")
    xb0, yb0 = train_ds[0]
    print(f"    train_ds[0] -> 特徴量の shape {tuple(xb0.shape)}、ラベル {yb0.item():.0f}\n")

    print("[2] DataLoader — バッチサイズ 64、毎エポックでシャッフル")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(0))
    valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False)  # 評価にシャッフルは不要
    n_steps = len(train_loader)
    print(f"    1 エポック = {len(train_ds)}件 / 64 = {n_steps}ステップ(バッチ)")
    print("    用語: ステップ=バッチ 1 個の処理、エポック=全データを一周\n")

    print("[3] 学習 — 標準テンプレート (エポックごとに train を一周 + valid で採点)")
    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()              # Sigmoid+BCE を合わせた安定版
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
                  f"valid loss {va_loss:.4f} | valid 正解率 {va_acc:.3f}")

    print("\n[4] 学習曲線の保存")
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
    print(f"    保存: {png_path}")

    print("\n[5] 学習曲線の読み方")
    print("    - 2 本の曲線が一緒に下がる       -> 正常に学習中")
    print("    - train だけ下がり valid が反転  -> 過学習の始まり (level09 の主題)")
    print("    - どちらも高いまま平ら           -> 過少適合: モデル/エポック/学習率を再検討")
    print(f"    今回の実行: 最終 train {history['train'][-1]:.4f}, valid {history['valid'][-1]:.4f}, "
          f"valid 正解率 {history['acc'][-1]:.3f}")


if __name__ == "__main__":
    main()
