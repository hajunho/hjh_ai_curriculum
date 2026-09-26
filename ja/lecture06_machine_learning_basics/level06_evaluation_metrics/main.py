"""
level06 — 評価指標: 正解率の落とし穴

不正比率 約 1.5% のカード取引データで
  - 無条件に「正常」と叫ぶだけの張りぼてモデルが正解率 98.5% になる罠を実演し
  - 混同行列 / 適合率 / 再現率 / F1 / ROC-AUC でモデルを正しく読みます。
AUC の確率的解釈 (「無作為な不正・正常のペアで不正側の確率が高くなる確率」) を
無作為ペア抽出のシミュレーションで直接検証します。
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# tx_count_1h はこの合成データで不正を「完璧に」切り分けてしまう特徴量なので除外。
# (現実でこんな完璧な特徴量が見えたら、性能を祝う前にリーク (leakage) を疑うこと!)
FEATURES = ["amount", "hour", "is_foreign"]


def print_metrics(name: str, y_true, y_pred) -> None:
    """混同行列と 4 大指標を一度に出力。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"    {name}")
    print(f"      混同行列: TP={tp:>3} (よく捕まえた)   FN={fn:>3} (見逃し!)")
    print(f"                FP={fp:>3} (空振り警報)    TN={tn:>4} (通過)")
    print(f"      正解率 {acc:.1%} / 適合率 {prec:.1%} / 再現率 {rec:.1%} / F1 {f1:.3f}")


def auc_by_sampling(y_true, proba, n_pairs: int = 10_000, seed: int = 0) -> float:
    """AUC の定義をシミュレーションで検証:
    不正 1 件と正常 1 件を無作為に取り出し、「不正側の確率の方が高い」割合。"""
    rng = np.random.default_rng(seed)
    pos = proba[y_true == 1]
    neg = proba[y_true == 0]
    p = rng.choice(pos, n_pairs)
    n = rng.choice(neg, n_pairs)
    return float(np.mean((p > n) + 0.5 * (p == n)))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] データの準備 --------------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=8000, seed=11))
    X, y = df[FEATURES].astype(float), df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)  # 不均衡比率を保つ分割
    print(f"[1] データ: カード取引 {len(df):,}件, 不正の比率 {y.mean():.2%}")
    print(f"    訓練 {len(X_tr):,}件 / テスト {len(X_te):,}件 (stratify で比率を維持)\n")

    # [2] 張りぼてモデル — 全部正常と予測 ------------------------------------------
    dummy_pred = np.zeros(len(y_te), dtype=int)
    print("[2] 張りぼてモデル: 無条件に「正常」と言い張る")
    print_metrics("全部正常と予測:", y_te, dummy_pred)
    print("      -> 正解率 98.5% なのに不正の検挙は 0 件。『正解率の落とし穴』の正体です。")
    print("         不均衡データでは、正解率は 1 行目ではなく脚注に書く指標。\n")

    # [3] 本物のモデル — ロジスティック回帰 -------------------------------------------
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print("[3] 本物のモデル: ロジスティック回帰 (しきい値 0.5)")
    print_metrics("ロジスティック回帰:", y_te, (proba >= 0.5).astype(int))
    print("      -> 正解率は張りぼてと数 %p の差ですが、混同行列はまったく別世界です。\n")

    # [4] しきい値のシーソー: 適合率 vs 再現率 ---------------------------------------
    print("[4] しきい値をなぞって見る適合率-再現率のシーソー")
    print("    しきい値   警報件数   適合率    再現率")
    for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
        pred = (proba >= th).astype(int)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        print(f"     {th:.1f}      {pred.sum():>4}     {prec:6.1%}   {rec:6.1%}")
    print("    -> 感度を上げると (しきい値↓) 見逃しは減り、空振り警報は増えます。")
    print("       どこに座るかは『見逃しコスト vs 空振りコスト』を知る人が決めます。\n")

    # [5] ROC-AUC: しきい値と無関係な分別力 -------------------------------------
    auc_dummy = roc_auc_score(y_te, np.zeros(len(y_te)))
    auc_model = roc_auc_score(y_te, proba)
    auc_sim = auc_by_sampling(y_te.to_numpy(), proba)
    print("[5] ROC-AUC — しきい値を決める前の『モデルの格』")
    print(f"    張りぼてモデル AUC : {auc_dummy:.3f} (コイン投げの水準)")
    print(f"    ロジスティック AUC : {auc_model:.3f}")
    print(f"    シミュレーション検証: 無作為な (不正, 正常) ペア 10,000 組のうち")
    print(f"                      不正側の確率が高かった割合 = {auc_sim:.3f}  (AUC と一致)")
    print()
    print("[6] まとめ: 不均衡問題の報告書に必須の 3 点セット")
    print("    (1) ベースライン(張りぼてモデル)の成績  (2) 運用しきい値の混同行列  (3) AUC")
    print("    『正解率 98.5%』の 1 行報告を見たら、この 3 つを要求しましょう。")
