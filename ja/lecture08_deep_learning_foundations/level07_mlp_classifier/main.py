"""
nn.Module で自分だけの MLP(多層パーセプトロン)分類モデルを作ります。
サブスク解約データ(churn_table)を学習用/テスト用に分け、
(1) ロジスティック回帰(= 隠れ層 0 個のニューラルネットワーク)と (2) MLP(6-16-8-1)を
同じ条件で torch で学習させ、正解率・適合率・再現率を比較します。
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
    """churn_table を標準化したテンソルに変換します。customer_id は意味のない列なので除外。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)

    # 学習用/テスト用の分割 (シャッフルしてから 75:25)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.75)
    tr, te = idx[:cut], idx[cut:]

    # 標準化: 平均/標準偏差は必ず「学習データだけで」計算 (リーク防止)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    X = (X - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(y[tr]), t(X[te]), t(y[te])


class LogisticRegression(nn.Module):
    """隠れ層 0 個のニューラルネットワーク = ロジスティック回帰 (lecture06 level05 のあのモデル)。"""

    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class ChurnMLP(nn.Module):
    """隠れ層 2 個(16, 8)の MLP。__init__ で部品を宣言し、forward で組み立てます。"""

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
    """full-batch の Adam 学習。小さなデータなのでバッチ分割は次のレベルで。"""
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
    """テスト性能: 正解率 + 解約(1)クラスの適合率/再現率。"""
    model.eval()
    with torch.no_grad():                       # 評価のときは微分の帳簿への記録を OFF
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

    print("[1] データ: サブスク解約の churn_table (n=2000、解約率は約 18%)")
    X_tr, y_tr, X_te, y_te = load_data()
    print(f"    学習 {len(X_tr)}件 / テスト {len(X_te)}件、特徴量 {X_tr.shape[1]}個 (customer_id は除外、標準化済み)")
    print(f"    テストの解約率: {y_te.mean().item():.1%} -> 「全員が継続」と答えるだけでも正解率は約 {1-y_te.mean().item():.0%}\n")

    print("[2] モデル A — ロジスティック回帰 (隠れ層 0 個のニューラルネットワーク)")
    logreg = LogisticRegression(len(FEATURES))
    print(f"    パラメータ数: {count_params(logreg)}")
    train(logreg, X_tr, y_tr)

    print("\n[3] モデル B — MLP (6-16-8-1、ReLU)")
    mlp = ChurnMLP(len(FEATURES))
    print(f"    パラメータ数: {count_params(mlp)}")
    train(mlp, X_tr, y_tr)

    print("\n[4] テスト性能の比較 (しきい値 0.5、解約=陽性)")
    print(f"    {'モデル':16s} {'正解率':>8s} {'適合率':>8s} {'再現率':>8s}")
    results = {}
    for name, model in [("LogisticReg", logreg), ("MLP", mlp)]:
        acc, prec, rec = evaluate(model, X_te, y_te)
        results[name] = acc
        print(f"    {name:16s} {acc:8.3f} {prec:8.3f} {rec:8.3f}")

    print("\n[5] しきい値を 0.5 -> 0.3 に下げると? (MLP)")
    acc3, prec3, rec3 = evaluate(mlp, X_te, y_te, threshold=0.3)
    print(f"    {'MLP(th=0.3)':16s} {acc3:8.3f} {prec3:8.3f} {rec3:8.3f}")
    print("    正解率を少し譲る代わりに、「取りこぼす解約顧客」(再現率)を大きく減らせます。")

    print("\n[6] 解釈")
    print("    - このデータは解約のルールが「ほぼ線形」に設計されているため、2 つのモデルは似た成績になります。")
    print("    - 教訓 1: ディープラーニングが常に勝つわけではありません。表形式のデータはまず単純なモデルから。")
    print("    - 教訓 2: それでも MLP は特徴量どうしの相互作用(例: プラン変更 x 問い合わせ急増)を")
    print("      自動で学ぶ余地があるので、関係が非線形であるほど差が開きます。")
    print("    - 教訓 3: 再現率(取りこぼした解約顧客)が低いなら、しきい値 0.5 を下げるのも実務の選択肢です。")


if __name__ == "__main__":
    main()
