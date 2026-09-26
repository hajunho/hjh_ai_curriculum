"""
スケーリングとパイプラインの実験。
単位がバラバラの churn_table 特徴量で KNN の解約予測を行いながら
(1) スケーリングなし vs あり、(2) 前処理を交差検証の外で vs パイプラインの中で
という 2 つの比較を通じて「なぜ前処理はパイプラインの中へ」なのかを確認します。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def report(name: str, y_true, proba) -> None:
    """AUC と「リスク上位100人中の実際の解約者数」(キャンペーン観点の指標)を出力。"""
    auc = roc_auc_score(y_true, proba)
    top100 = np.asarray(y_true)[np.argsort(proba)[::-1][:100]].sum()
    print(f"    {name:<26} AUC {auc:.3f} / リスク上位100人中の実際の解約者 {int(top100)}名")


def main() -> None:
    print("=" * 62)
    print(" スケーリングとパイプライン: 単位の統一 + リーケージの根絶")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)

    # [1] 特徴量の単位確認 --------------------------------------------------
    print("\n[1] 特徴量のスケール (同じものさしで測れているか?)")
    for c in FEATURES:
        print(f"    {c:<18} 範囲 {X[c].min():>7,.0f} ~ {X[c].max():>7,.0f}")
    print("    => monthly_fee が他の特徴量より数百〜数千倍大きい: 距離計算を独占します。")

    # [2] スケーリングなしの KNN ----------------------------------------------
    print("\n[2] スケーリングなしの KNN (k=15)")
    knn_raw = KNeighborsClassifier(n_neighbors=15)
    knn_raw.fit(X_tr, y_tr)
    report("KNN (スケーリングなし)", y_te, knn_raw.predict_proba(X_te)[:, 1])
    base = int(y_te.sum() / len(y_te) * 100)
    print(f"    (参考: 無作為に100人選んでも平均 {base}名は解約者です)")
    print("    (料金の数百ウォンの差が、利用日数30日の差より大きく扱われている状態)")

    # [3] パイプライン: StandardScaler + KNN --------------------------------
    print("\n[3] パイプライン = 洗浄(スケーリング) -> 組み立て(モデル) のベルトコンベア")
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    pipe.fit(X_tr, y_tr)          # スケーラーの fit は自動的に学習データにだけ
    report("KNN + StandardScaler", y_te, pipe.predict_proba(X_te)[:, 1])
    print("    => 同じモデル・同じデータで、単位をそろえただけなのに性能が変わります。")

    # [4] 間違った順序 vs 正しい順序 (交差検証で) -----------------------
    print("\n[4] 前処理をどこで行うか: 交差検証 5-fold、指標=AUC")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 間違った方式: 全データを「先に」スケーリング -> フォールドの外の情報が漏れ込む
    X_scaled_all = pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES)
    bad = cross_val_score(KNeighborsClassifier(n_neighbors=15),
                          X_scaled_all, y, cv=cv, scoring="roc_auc")

    # 正しい方式: パイプラインごと渡す -> フォールドごとに学習部分だけで fit し直す
    good = cross_val_score(make_pipeline(StandardScaler(),
                                         KNeighborsClassifier(n_neighbors=15)),
                           X, y, cv=cv, scoring="roc_auc")
    print(f"    間違った順序(全体をスケーリング後に CV): {bad.mean():.4f} ± {bad.std():.4f}")
    print(f"    正しい順序(パイプラインごと CV)       : {good.mean():.4f} ± {good.std():.4f}")
    print("    => スケーリング程度なら差は小さく見えても、ターゲットエンコーディング・欠損補完・特徴量選択のように")
    print("       正解/分布を濃く使う前処理では、スコアが大きく水増しされることがあります。")
    print("       ルールはひとつ:「前処理はすべてパイプラインの中へ」。")

    # [5] デプロイ観点: オブジェクト1つで予測 -------------------------------------
    print("\n[5] デプロイのシミュレーション: パイプラインオブジェクト1つ = 前処理 + モデル")
    new_customers = pd.DataFrame([
        {"tenure_months": 2, "monthly_fee": 29900, "usage_days_30d": 3,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])
    probs = pipe.predict_proba(new_customers)[:, 1]
    for i, p in enumerate(probs):
        print(f"    新規顧客 {i+1}: 解約確率 {p:.1%}")
    print("    => デプロイコードは pipe.predict_proba(新しいデータ) の1行。前処理の入れ忘れ事故が不可能です。")
    print("\n    教訓: パイプラインは便利機能ではなく、リーケージ・不一致事故を防ぐ安全装置です。")


if __name__ == "__main__":
    main()
