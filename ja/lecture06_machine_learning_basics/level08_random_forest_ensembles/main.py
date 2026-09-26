"""
level08 — ランダムフォレストとアンサンブル: 集団知性の実験

churn_table で単一の決定木 vs ランダムフォレストを比較します。
  [2] バギング (ブートストラップ + 投票) を 15 行で自前実装し「平均の力」を確認
  [3] 性能比較 (AUC / 再現率)
  [4] 安定性比較: 分割を 12 回変えて AUC の揺れ (標準偏差) を測定
  [5] 特徴量の重要度: 1 本の木 vs 森
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def bagged_predict(X_tr, y_tr, X_te, n_trees: int, seed: int = 0) -> np.ndarray:
    """ミニバギングの自前実装: 復元抽出したデータで深い木を育て、確率の平均 (投票)。
    sklearn の RandomForest = これに「分岐ごとの特徴量ランダム選択」を足したもの。"""
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    probas = []
    for i in range(n_trees):
        idx = rng.choice(n, n, replace=True)          # ブートストラップ: 同サイズの復元抽出
        tree = DecisionTreeClassifier(random_state=i)  # 深さ無制限 (低バイアス・高バリアンス)
        tree.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        probas.append(tree.predict_proba(X_te)[:, 1])
    return np.mean(probas, axis=0)                     # 平均 = 揺れ(分散)の相殺


if __name__ == "__main__":
    np.random.seed(0)

    # [1] データの準備 --------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] データ: 顧客 {len(df)}人, 解約率 {y.mean():.1%}\n")

    # [2] ミニバギングの自前実装 -------------------------------------------------
    single = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    auc_single = roc_auc_score(y_te, single.predict_proba(X_te)[:, 1])
    print("[2] バギングを自前で実装 — 牛の体重当てコンテストの原理")
    print(f"    深い木 1 本               テスト AUC = {auc_single:.3f}")
    for n_trees in [5, 25]:
        auc_bag = roc_auc_score(y_te, bagged_predict(X_tr, y_tr, X_te, n_trees))
        print(f"    同じ木 {n_trees:>2}本の投票        テスト AUC = {auc_bag:.3f}")
    print("    -> 一人ひとり (木) は過学習しても、違う経験をした多数の平均は強くなります。\n")

    # [3] 性能比較: 単一の木 vs ランダムフォレスト --------------------------------
    print("[3] 性能比較 (テストセット)")
    models = {
        "木(深さ 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "木(無制限)": DecisionTreeClassifier(random_state=0),
        "ランダムフォレスト(300本)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
    }
    print("    モデル                     AUC     再現率(しきい値 0.5)")
    forest = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        rec = recall_score(y_te, m.predict(X_te))
        print(f"    {name:20s} {auc:.3f}      {rec:6.1%}")
        if isinstance(m, RandomForestClassifier):
            forest = m
    print()

    # [4] 安定性比較: 分割を変えながら揺れを測定 ------------------------------
    print("[4] 安定性比較 — 分割を 12 回変えて AUC を反復測定")
    aucs_tree, aucs_rf = [], []
    for rep in range(12):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25,
                                          random_state=rep, stratify=y)
        t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xa, ya)
        f = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(Xa, ya)
        aucs_tree.append(roc_auc_score(yb, t.predict_proba(Xb)[:, 1]))
        aucs_rf.append(roc_auc_score(yb, f.predict_proba(Xb)[:, 1]))
    aucs_tree, aucs_rf = np.array(aucs_tree), np.array(aucs_rf)
    print(f"    単一の木          AUC 平均 {aucs_tree.mean():.3f} ± 標準偏差 {aucs_tree.std():.3f}")
    print(f"    ランダムフォレスト  AUC 平均 {aucs_rf.mean():.3f} ± 標準偏差 {aucs_rf.std():.3f}")
    print(f"    フォレストが勝った回数: {int((aucs_rf > aucs_tree).sum())}/12")
    print("    -> 平均性能だけでなく「揺れ (標準偏差)」が小さいことが実務の信頼の核心。\n")

    # [5] 特徴量の重要度: 1 本の木 vs 森 ------------------------------------------
    print("[5] 特徴量の重要度比較 (不純度減少への寄与, 合計 1)")
    tree5 = models["木(深さ 5)"]
    print("    特徴量               木1本      フォレスト300本")
    for i in np.argsort(-forest.feature_importances_):
        print(f"    {FEATURES[i]:18s}   {tree5.feature_importances_[i]:.3f}       "
              f"{forest.feature_importances_[i]:.3f}")
    print("    -> 森の重要度は多数の木の平均なので、より安定しています。")
    print("       (それでも「よく使われた度合い」であって因果ではない — 公正な測定は level11)")
