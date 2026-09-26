"""
交差検証の実験 3 種。
(1) 単一分割のスコアが分割の運でどれだけ揺れるかを 30 回の実験で確認し
(2) 5-fold 交差検証で「平均 ± 標準偏差」のレポートを作り
(3) 時系列データで無作為分割がスコアを水増しすることを TimeSeriesSplit と比較します。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold, TimeSeriesSplit)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def main() -> None:
    print("=" * 62)
    print(" 交差検証: 性能の数字の「運まかせ度」を測定して飼いならす")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

    # [1] 単一分割 30 回: スコアは確率変数 --------------------------------
    print("\n[1] 同じモデル、同じデータ — 分割 seed だけ 30 回変えて AUC を測定")
    scores = []
    for seed in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=seed, stratify=y)
        pipe.fit(X_tr, y_tr)
        scores.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
    scores = np.array(scores)
    print(f"    最小 {scores.min():.4f} / 最大 {scores.max():.4f} / "
          f"幅 {scores.max()-scores.min():.4f} / 標準偏差 {scores.std():.4f}")
    # 簡易ヒストグラム
    bins = np.linspace(scores.min(), scores.max() + 1e-9, 7)
    counts, _ = np.histogram(scores, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], counts):
        print(f"    {lo:.3f}~{hi:.3f} | {'#' * c}")
    print("    => 「単一スコア 0.87」の裏には、これだけの運が隠れています。")

    # [2] 5-fold 交差検証 --------------------------------------------------
    print("\n[2] 5-fold StratifiedKFold 交差検証 (パイプラインごと投入)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("    フォールド別 AUC:", " ".join(f"{s:.4f}" for s in cv_scores))
    print(f"    報告形式 => AUC {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # [3] 2 モデルの公正比較 ----------------------------------------------
    print("\n[3] モデル比較: ロジスティック回帰 vs ランダムフォレスト (同じ CV)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"    ロジスティック回帰 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    ランダムフォレスト : {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    diff = abs(cv_scores.mean() - rf_scores.mean())
    noise = max(cv_scores.std(), rf_scores.std())
    verdict = "意味のある差とは言いにくいです (平均の差 < ばらつき)" if diff < noise \
        else "差がばらつきより大きく、意味がありそうです"
    print(f"    平均の差 {diff:.4f} vs ばらつき {noise:.4f} => {verdict}")

    # [4] 時系列の落とし穴: 無作為 KFold vs TimeSeriesSplit ---------------------
    print("\n[4] 時系列データ: 未来で過去を予測するとスコアが水増しされます")
    sales = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    sales = sales.dropna(subset=["revenue"])
    sales = sales[sales["revenue"] > 0]
    daily = sales.groupby("day_index")["revenue"].sum().reset_index()
    # ラグ(lag)特徴量: 前日・移動平均 -> 無作為分割では検証情報が学習の特徴量に染み込む
    daily["lag1"] = daily["revenue"].shift(1)
    daily["ma7"] = daily["revenue"].shift(1).rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)
    Xs, ys = daily[["lag1", "ma7"]], daily["revenue"]

    model = Ridge(alpha=1.0)
    shuffled = cross_val_score(model, Xs, ys, scoring="r2",
                               cv=KFold(n_splits=5, shuffle=True, random_state=42))
    tssplit = TimeSeriesSplit(n_splits=5)
    ordered = cross_val_score(model, Xs, ys, scoring="r2", cv=tssplit)
    print(f"    無作為 KFold(shuffle)      R2: {shuffled.mean():.4f} ± {shuffled.std():.4f}")
    print(f"    TimeSeriesSplit(順序を維持) R2: {ordered.mean():.4f} ± {ordered.std():.4f}")
    print("    (R2 < 0 = 「平均で当てずっぽう」より悪いという意味。前日売上だけでは未来の予測が")
    print("     難しいという正直な成績表であり、無作為分割はその事実を隠していたのです。)")
    print("    TimeSeriesSplit のフォールド構造 (学習は常に検証より過去):")
    for i, (tr, te) in enumerate(tssplit.split(Xs)):
        print(f"      fold{i+1}: 学習 day {tr.min()}~{tr.max()} -> 検証 day {te.min()}~{te.max()}")
    print("\n    教訓: 性能は 1 回測れば「運」、何回も測れば「実力 ± 誤差」になります。")
    print("          そして時間軸があるなら、試験問題は必ず未来から持ってきてください。")


if __name__ == "__main__":
    main()
