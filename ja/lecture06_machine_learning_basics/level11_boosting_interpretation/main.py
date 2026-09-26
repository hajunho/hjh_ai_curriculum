"""
level11 — 勾配ブースティングとモデル解釈

churn_table でブースティングモデルを作り、「なぜ?」まで答えます。
  [2] ロジスティック vs ランダムフォレスト vs HistGradientBoosting の性能比較
  [3] 学習率 (誤答ノートの反映強度) x 木の数のミニ実験
  [4] 順列重要度: 特徴量 1 つをシャッフルすると AUC がどれだけ崩れるか
  [5] 部分依存: 特徴量の値に応じた予測確率の形 (テキストグラフ)
  [6] 個別予測の説明: 特徴量を平均値に差し替えながら根拠を分解
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_JA = {"tenure_months": "契約月数", "monthly_fee": "月額料金",
              "usage_days_30d": "利用日数", "support_calls_30d": "問い合わせ数",
              "plan_changes": "プラン変更", "auto_pay": "自動決済"}


def explain_one(model, x_row: pd.DataFrame, baseline: pd.Series) -> list[tuple[str, float]]:
    """個別予測の説明器 (12 行): 特徴量 1 つを「平均的な顧客」の値に差し替えたとき、
    解約確率がどれだけ変わるかを特徴量ごとに測る。変化幅の大きい特徴量 = 判定の根拠。"""
    p0 = model.predict_proba(x_row)[0, 1]
    contribs = []
    for f in FEATURES:
        x_mod = x_row.copy()
        x_mod[f] = baseline[f]
        p_mod = model.predict_proba(x_mod)[0, 1]
        contribs.append((f, p0 - p_mod))     # + ならこの特徴量がリスクを高めた根拠
    return sorted(contribs, key=lambda t: -abs(t[1]))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] データ -------------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] データ: 顧客 {len(df)}人, 解約率 {y.mean():.1%}\n")

    # [2] 3 つのモデルの卒業試験 ----------------------------------------------------
    print("[2] これまで学んだモデルたちの性能比較 (テスト AUC)")
    models = {
        "ロジスティック回帰 (level05)": Pipeline([
            ("s", StandardScaler()), ("c", LogisticRegression(random_state=0))]),
        "ランダムフォレスト (level08)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=0, early_stopping=True),
    }
    boost = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {name:24s} AUC = {auc:.3f}")
        if isinstance(m, HistGradientBoostingClassifier):
            boost = m
    print("    -> 意外にもロジスティックが 1 位! この合成データの本当の構造が『線形の採点表』")
    print("       (ログオッズの線形式) で作られているからです。教訓は 2 つ:")
    print("       (1) 強いモデルが常に勝つのではなく、データ構造に合ったモデルが勝つ。")
    print("       (2) 非線形・相互作用の多い現実のデータでは、ブースティングが先行することが多い。\n")

    # [3] 学習率 x 木の数のミニ実験 -------------------------------------------
    print("[3] 学習率 (誤答ノートの反映強度) の実験 — ブレーキと授業コマ数の取引")
    print("    学習率   最大の木   テスト AUC")
    for lr, n_iter in [(1.0, 50), (0.3, 100), (0.1, 200), (0.03, 500)]:
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=n_iter, random_state=0,
            early_stopping=True).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {lr:<6}    {n_iter:>5}      {auc:.3f}")
    print("    -> 学習率を下げると木は多く必要になりますが、結果が安定する傾向。")
    print("       (learning_rate は level09 の正則化の哲学がブースティングに来たもの)\n")

    # [4] 順列重要度 ----------------------------------------------------------
    print("[4] 順列重要度 — 特徴量 1 つをシャッフルしたときのテスト AUC の下落幅")
    perm = permutation_importance(boost, X_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=0)
    print("    特徴量         AUC の下落 (平均±標準偏差)")
    for i in np.argsort(-perm.importances_mean):
        bar = "#" * max(0, int(perm.importances_mean[i] * 200))
        print(f"    {FEATURE_JA[FEATURES[i]]:10s}   {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}  {bar}")
    print("    -> 『完成したモデルをテストデータで尋問する』方式なのでモデルの種類を問わず、")
    print("       不純度の重要度 (level07〜08) より報告用として正直です。\n")

    # [5] 部分依存 -------------------------------------------------------------
    print("[5] 部分依存 — 他の条件を固定し、特徴量 1 つだけ動かしたときの平均解約確率")
    for feat in ["usage_days_30d", "support_calls_30d"]:
        # method="brute": predict_proba 基準で計算 -> 結果が「確率」の単位になる
        pd_res = partial_dependence(boost, X_te, [feat], kind="average",
                                    grid_resolution=7, method="brute")
        grid = pd_res["grid_values"][0]
        avg = pd_res["average"][0]
        print(f"    {FEATURE_JA[feat]} ({feat})")
        for g, v in zip(grid, avg):
            bar = "#" * int(v * 60)
            print(f"      値 {g:6.1f} -> 平均確率 {v:5.1%} {bar}")
    print("    -> 曲線の『形』(どの区間で急変するか) が、ロジスティックの係数 1 つよりも")
    print("       豊かな情報をくれます。ただし関連であって因果ではありません。\n")

    # [6] 個別予測の説明 --------------------------------------------------------
    proba_te = boost.predict_proba(X_te)[:, 1]
    idx = int(np.argmax(proba_te))                     # テストで最もリスクの高い顧客
    x_row = X_te.iloc[[idx]]
    baseline = X_tr.mean()                             # 「平均的な顧客」
    print("[6] 個別予測の説明 — 最高リスク顧客 1 人の判定根拠を分解")
    print(f"    この顧客の解約確率: {proba_te[idx]:.1%} (平均的な顧客の基準値との比較)")
    print("    特徴量 (顧客の値 -> 平均値に差し替え)      確率の変化")
    for f, delta in explain_one(boost, x_row, baseline):
        if abs(delta) < 0.005:
            direction = "影響わずか"
        else:
            direction = "リスクを高めた根拠" if delta > 0 else "リスクを下げた要素"
        print(f"    {FEATURE_JA[f]:10s} ({x_row[f].iloc[0]:>8.1f} -> {baseline[f]:>8.1f})   "
              f"{delta:+6.1%}p  {direction}")
    print("    -> 『問い合わせを平均水準に戻せば確率が大きく下がる』といった文章が、")
    print("       フォロー担当のトークスクリプトの根拠になります。(専門ツール SHAP も同じ原理の精緻化)")
    print()
    print("[7] lecture06 完走! 性能 (ブースティング) と説明 (重要度・部分依存・事例の説明) を")
    print("    セットで報告できて初めて、モデルは組織に採用されます。")
