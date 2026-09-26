"""
level05 — ロジスティック回帰: サブスク解約確率モデル

churn_table (顧客 2000人) で「解約確率」を出力する分類モデルを作ります。
  - シグモイド: 線形スコア z を 0〜1 の確率に変えるじょうご
  - 係数 -> オッズ比 (odds ratio) の翻訳: 「問い合わせ 1 単位増 -> 解約オッズ N 倍」
  - しきい値 (threshold) はモデルではなくビジネスが決める
"""

import math
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_JA = {"tenure_months": "契約月数", "monthly_fee": "月額料金",
              "usage_days_30d": "利用日数(30日)", "support_calls_30d": "問い合わせ数(30日)",
              "plan_changes": "プラン変更", "auto_pay": "自動決済の有無"}


def sigmoid(z: float) -> float:
    """確率のじょうご: どんなに大きいスコアも 0〜1 の間に押し込む。"""
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] データの準備 + 分割 -------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X = df[FEATURES].astype(float)     # customer_id は意味のない列なので除外
    y = df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)  # 解約率を保った分割
    print("[1] データ: サブスク顧客 2000人、解約率 {:.1%}".format(y.mean()))
    print(f"    訓練 {len(X_tr)}人 / テスト {len(X_te)}人 (level04 の原則を順守)\n")

    # [2] シグモイドのじょうごを観察 ----------------------------------------------
    print("[2] シグモイド: 線形スコア z -> 確率 p")
    for z in [-4, -2, 0, 2, 4]:
        print(f"    z = {z:+d}  ->  p = {sigmoid(z):5.1%}")
    print("    -> スコア 0 点なら五分五分 (50%)、±4 点ならほぼ確定。\n")

    # [3] 学習 ---------------------------------------------------------------
    # Pipeline: スケーラーが訓練セットだけで fit -> リークを自動防止 (level04)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]        # 解約確率
    pred_05 = (proba_te >= 0.5).astype(int)
    print("[3] ロジスティック回帰の学習完了 — テスト成績 (しきい値 0.5)")
    print(f"    正解率 {accuracy_score(y_te, pred_05):.1%} / "
          f"適合率 {precision_score(y_te, pred_05):.1%} / "
          f"再現率 {recall_score(y_te, pred_05):.1%}\n")

    # [4] 係数 -> オッズ比の翻訳 --------------------------------------------------
    clf = model.named_steps["clf"]
    print("[4] 係数の解釈 (標準化した特徴量基準: 「1 標準偏差の増加」の効果)")
    print("    特徴量              係数      オッズ比    解釈")
    order = np.argsort(-np.abs(clf.coef_[0]))
    for i in order:
        coef = clf.coef_[0][i]
        orat = math.exp(coef)
        direction = "解約リスク増加" if coef > 0 else "解約リスク減少"
        print(f"    {FEATURE_JA[FEATURES[i]]:14s} {coef:+7.3f}   {orat:6.2f}倍   {direction}")
    print("    -> データジェネレーターに仕込んだ本物のシグナル (利用日数↓, 問い合わせ↑, 自動決済-) と")
    print("       方向が一致しているか確認してみてください。(ただし相関であって因果の証明ではない)\n")

    # [5] 個別予測 + しきい値の実験 ---------------------------------------------
    print("[5] 個別顧客の予測と、しきい値というビジネス判断")
    samples = pd.DataFrame([
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 3, "monthly_fee": 29900, "usage_days_30d": 2,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])[FEATURES].astype(float)
    names = ["優良利用の顧客", "新規+低利用+問い合わせ殺到", "ごく平均的な顧客"]
    for name, p in zip(names, model.predict_proba(samples)[:, 1]):
        print(f"    {name:16s} 解約確率 {p:5.1%}")
    print()
    print("    しきい値を変えれば、同じモデルでも違う決定を下します:")
    print("    しきい値   リスク分類の人数   適合率   再現率")
    for th in [0.5, 0.3]:
        pred = (proba_te >= th).astype(int)
        print(f"     {th:.1f}        {pred.sum():>4}人      "
              f"{precision_score(y_te, pred):6.1%}  {recall_score(y_te, pred):6.1%}")
    print("    -> しきい値を下げると見逃す解約者 (再現率↑) は減るが、空振りのフォロー (適合率↓) が増える。")
    print("       0.5 は慣例にすぎない — フォローのコストと顧客価値がしきい値を決めるべきです。")
