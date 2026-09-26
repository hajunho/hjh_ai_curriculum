"""
過学習をわざと起こしたうえで、3 つの処方の効果を比較します。
サブスク解約データで学習データを 120 件だけ与えて大きな MLP に丸暗記させ、
(1) ドロップアウト (2) weight decay (3) 早期終了 がそれぞれ
検証(valid)性能をどれだけ回復させるかを曲線 PNG と表で確認します。
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
    """churn_table 2000 件のうち「たった 120 件」だけを学習に使用 -> 過学習を誘発。
    残りの 1880 件は検証用なので、成績表がとても安定します。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, va = idx[:120], idx[120:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8   # 学習セットの統計量のみ使用
    X = (X - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(y[tr]), t(X[va]), t(y[va])


def build_model(dropout=0.0):
    """データ(120 件)に対してわざと大きめの MLP (約 4.7 千パラメータ) = 丸暗記の誘発装置。"""
    layers = [nn.Linear(6, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))       # 学習中はニューロンがランダムに欠勤
    layers += [nn.Linear(64, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers)


def train(name, X_tr, y_tr, X_va, y_va, dropout=0.0, weight_decay=0.0):
    """full-batch の Adam 学習。エポックごとの valid loss/正解率の記録を返します。"""
    torch.manual_seed(1)                         # すべての条件に同じ初期重みを
    model = build_model(dropout)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"tr_loss": [], "va_loss": [], "va_acc": []}
    for _ in range(N_EPOCHS):
        model.train()                            # ドロップアウトは train モードでのみ働く
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        model.eval()                             # 評価のときは欠勤なしで全員出勤
        with torch.no_grad():
            logits = model(X_va)
            hist["va_loss"].append(loss_fn(logits, y_va).item())
            hist["va_acc"].append(((logits > 0) == y_va.bool()).float().mean().item())
        hist["tr_loss"].append(loss.item())
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(X_tr) > 0) == y_tr.bool()).float().mean().item()
    print(f"    {name:26s}: train 正解率 {tr_acc:.3f} | 最終 valid 正解率 {hist['va_acc'][-1]:.3f}")
    return hist


def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 実験の設計: 学習 120 件 vs パラメータ約 4,700 個 — 丸暗記(過学習)が起きる条件")
    X_tr, y_tr, X_va, y_va = load_data()
    print(f"    サブスク解約の二値分類、学習 {len(X_tr)}件 / 検証 {len(X_va)}件、{N_EPOCHS}エポックの full-batch\n")

    print("[2] 条件別の学習 (同じシード、同じ初期重み)")
    runs = {
        "baseline": train("baseline (無防備)", X_tr, y_tr, X_va, y_va),
        "dropout": train("dropout p=0.5", X_tr, y_tr, X_va, y_va, dropout=0.5),
        "weight_decay": train("weight decay 3e-2", X_tr, y_tr, X_va, y_va, weight_decay=3e-2),
    }

    print("\n[3] 早期終了(early stopping) — baseline を「一番よかった瞬間」で止めていたら?")
    va = runs["baseline"]["va_loss"]
    best_epoch = int(np.argmin(va))              # valid loss が最低のエポック
    es_acc = runs["baseline"]["va_acc"][best_epoch]
    print(f"    valid loss の最低点: epoch {best_epoch + 1} (600 エポック中!)")
    print(f"    その時点の valid 正解率 {es_acc:.3f} vs 最後まで回した baseline {runs['baseline']['va_acc'][-1]:.3f}\n")

    print("[4] 最終比較表 (valid 正解率が成績表)")
    rows = [("無防備 (600 エポック完走)", runs["baseline"]["va_acc"][-1]),
            ("ドロップアウト 0.5", runs["dropout"]["va_acc"][-1]),
            ("weight decay 3e-2", runs["weight_decay"]["va_acc"][-1]),
            (f"早期終了 (epoch {best_epoch + 1})", es_acc)]
    print(f"    {'処方':26s} {'valid 正解率':>12s}")
    for name, acc in rows:
        print(f"    {name:28s} {acc:>10.3f}")
    assert max(r[1] for r in rows[1:]) > rows[0][1] + 0.02, "処方の効果が再現されませんでした"

    print("\n[5] 曲線 PNG の保存")
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
    print(f"    保存: {png_path}")

    print("\n[6] まとめ: 過学習の処方は「暗記できないように邪魔すること」です。")
    print("    欠勤トレーニング(ドロップアウト)、大きな重みへの課税(weight decay)、成績の頂点で止める(早期終了)。")
    print("    実務では 3 つを組み合わせて使い、最高の処方はいつでも「データをもっと集めること」です。")


if __name__ == "__main__":
    main()
