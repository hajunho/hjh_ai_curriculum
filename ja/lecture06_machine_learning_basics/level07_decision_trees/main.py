"""
level07 — 決定木: 二十の質問ゲームで解約顧客を探す

churn_table で決定木を学習し、
  - ジニ不純度を直接計算して「良い質問」の基準を理解し
  - 学習された木を人が読めるルール文として出力して解釈し
  - 深さを変えながら過学習の分かれ道を観察します。
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def gini(labels: np.ndarray) -> float:
    """ジニ不純度: 部屋が 1 種類なら 0、半々なら 0.5。この 8 行が木の数学のすべて。"""
    if len(labels) == 0:
        return 0.0
    p = labels.mean()               # 解約の割合
    return 1.0 - (p ** 2 + (1 - p) ** 2)


def split_gain(y: np.ndarray, mask: np.ndarray) -> float:
    """質問 (mask) の後、2 つの部屋のサイズ加重平均の不純度がどれだけ減ったか。"""
    left, right = y[mask], y[~mask]
    after = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
    return gini(y) - after


if __name__ == "__main__":
    np.random.seed(0)

    # [1] データの準備 (木は標準化が不要: しきい値比較のモデル) -----------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] データ: 顧客 {len(df)}人, 解約率 {y.mean():.1%} — 訓練 {len(X_tr)} / テスト {len(X_te)}\n")

    # [2] ジニ不純度を直接計算: 良い質問とは? -------------------------------
    y_arr = y_tr.to_numpy()
    print("[2] ジニ不純度 — 「部屋がどれだけ混ざっているか」")
    print(f"    質問前の部屋全体の不純度: {gini(y_arr):.4f}")
    for feat, th in [("usage_days_30d", 9.5), ("monthly_fee", 15000), ("support_calls_30d", 1.5)]:
        gain = split_gain(y_arr, (X_tr[feat] <= th).to_numpy())
        print(f"    質問 \"{feat} <= {th}\" の不純度の減少: {gain:.4f}")
    print("    -> 木はすべての特徴量 x すべてのしきい値を試し、減少幅が最大の質問を選びます。")
    print("       (level00 のしきい値探索が再帰的に繰り返されるだけ — 魔法ではありません)\n")

    # [3] 深さ 3 の木を学習 + ルールをテキストで出力 ------------------------------
    # class_weight="balanced": 解約者 (15%) を継続者より重く扱うことで、
    # 少数派クラス (解約) 側のルールが木に現れるようにします。
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                                  class_weight="balanced", random_state=0)
    tree.fit(X_tr, y_tr)
    print("[3] 深さ 3 の木のルール全文 (export_text)")
    print(export_text(tree, feature_names=FEATURES))

    # 葉 (leaf) の部屋を日本語の報告文に翻訳
    leaf_id = tree.apply(X_tr)
    print("    葉の部屋のまとめ (訓練データ基準):")
    rows = []
    for leaf in np.unique(leaf_id):
        members = y_tr[leaf_id == leaf]
        rows.append((leaf, len(members), members.mean()))
    for leaf, n, rate in sorted(rows, key=lambda r: -r[2])[:3]:
        print(f"      リスク上位の部屋 #{leaf}: {n}人, 解約率 {rate:.1%}")
    print("    -> 上位の部屋への経路 (上のルール木でたどれる) はそのまま CRM ルールへ逆輸入可能。\n")

    # [4] 深さ別の訓練/テスト成績 — 過学習の分かれ道 ----------------------------
    print("[4] 深さと過学習 (level04 実験 A の再訪)")
    print("    深さ      訓練の正解率   テストの正解率   テストの再現率")
    for depth in [1, 3, 5, 10, None]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        rec = recall_score(y_te, t.predict(X_te))
        label = "無制限" if depth is None else f"{depth:>4}"
        print(f"    {label:>6}      {t.score(X_tr, y_tr):6.1%}       {t.score(X_te, y_te):6.1%}        {rec:6.1%}")
    print("    -> 深さ無制限 = 訓練 100% の丸暗記、テストではむしろ損。\n")

    # [5] 特徴量の重要度 ---------------------------------------------------------
    print("[5] 特徴量の重要度 (不純度の減少への寄与割合, 合計 1)")
    order = np.argsort(-tree.feature_importances_)
    for i in order:
        bar = "#" * int(tree.feature_importances_[i] * 40)
        print(f"    {FEATURES[i]:18s} {tree.feature_importances_[i]:.3f} {bar}")
    print("    -> 「よく使われた特徴量」であって「原因のランキング」ではありません (解釈に注意, level11)。")
