"""
実践 2 — 顧客チャーン予測の完結プロジェクト。
問題仕様 -> ミニ EDA -> 特徴量 -> 交差検証 -> 最終モデル -> 係数の解釈 ->
「リスク顧客トップ10 + 主な原因 + 推奨アクション」の表まで、
マーケティングチームがそのまま実行できる成果物を作ります。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score

BASE = ["tenure_months", "monthly_fee", "usage_days_30d",
        "support_calls_30d", "plan_changes", "auto_pay"]
DERIVED = ["usage_per_tenure", "calls_plus_changes"]
FEAT_JA = {
    "tenure_months": "契約月数", "monthly_fee": "月額料金",
    "usage_days_30d": "直近30日の利用日数", "support_calls_30d": "直近30日の問い合わせ",
    "plan_changes": "プラン変更", "auto_pay": "自動決済",
    "usage_per_tenure": "期間に対する活動性", "calls_plus_changes": "不満シグナル合計",
}
# 主な原因 -> 推奨アクション (モデルの外側で人間が設計する対応ルール)
ACTION_MAP = {
    "usage_days_30d": "再訪を促すコンテンツ + 7日間無料利用券",
    "usage_per_tenure": "再訪を促すコンテンツ + 7日間無料利用券",
    "support_calls_30d": "CS 優先相談の割り当て、不満原因の解決",
    "calls_plus_changes": "CS 優先相談の割り当て、不満原因の解決",
    "plan_changes": "料金プランのパーソナライズ相談",
    "monthly_fee": "料金プランのパーソナライズ相談",
    "auto_pay": "自動決済への切り替えで1か月20%割引",
    "tenure_months": "オンボーディングガイド + 初月特典の案内",
}


def main() -> None:
    print("=" * 70)
    print(" 実践 2: 顧客チャーン予測 — リスト・理由・アクションまで完結させる")
    print("=" * 70)

    # [1] 問題仕様 --------------------------------------------------------
    print("\n[1] 問題仕様 (コードより先に)")
    print("    予測対象: 今月のサブスク解約の有無 / 活用: 毎週リスク上位顧客に CRM キャンペーン")
    print("    目標指標: 再現率 55%+ のもとで適合率 25%+ / ベースライン: 無作為送付(的中率=解約率 約15%)")

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    # [2] ミニ EDA ---------------------------------------------------------
    print(f"\n[2] ミニ EDA: {len(df)}名, 解約率 {df['churned'].mean():.1%}")
    grp = df.groupby("churned")[BASE].mean()
    gap = ((grp.loc[1] - grp.loc[0]) / grp.loc[0]).sort_values(key=abs, ascending=False)
    print("    解約グループが残留グループと最も違う信号 (平均差の比率):")
    for name, v in gap.head(3).items():
        print(f"      {FEAT_JA[name]:<14} {v:+.0%}")

    # [3] 特徴量の準備 --------------------------------------------------------
    print("\n[3] 特徴量: 元の6個 + 派生2個")
    df["usage_per_tenure"] = df["usage_days_30d"] / (df["tenure_months"] + 1)
    df["calls_plus_changes"] = df["support_calls_30d"] + df["plan_changes"]
    features = BASE + DERIVED
    X, y = df[features], df["churned"]

    # [4] 交差検証 + 最終モデル ----------------------------------------------
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\n[4] 5-fold 交差検証 AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    idx_te = X_te.index
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rec, prec = recall_score(y_te, pred), precision_score(y_te, pred)
    print(f"    テスト性能: AUC {roc_auc_score(y_te, proba):.3f} / "
          f"再現率 {rec:.1%} / 適合率 {prec:.1%}")
    goal = "達成" if (rec >= 0.55 and prec >= 0.25) else "未達 -> しきい値・特徴量を再検討"
    print(f"    目標指標(再現率 55%+, 適合率 25%+) の判定: {goal}")
    print(f"    (無作為送付ベースラインの的中率 {y_te.mean():.1%} に対し適合率 {prec/y_te.mean():.1f}倍)")

    # [5] 解釈: 標準化係数 -------------------------------------------------
    print("\n[5] モデルの解釈 — 何がリスクを高めるのか (標準化係数)")
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    for name, c in sorted(zip(features, coefs), key=lambda t: -abs(t[1])):
        arrow = "リスク増加" if c > 0 else "リスク減少"
        print(f"    {FEAT_JA[name]:<14} {c:+.2f} ({arrow}) {'#' * int(abs(c) * 6)}")

    # [6] 成果物: リスクトップ10 + 原因 + アクション -----------------------------------
    print("\n[6] 最終成果物 — 解約リスクトップ10リスト (テスト顧客が基準)")
    # 個々の顧客の「主な原因」 = 標準化特徴量値 x 係数 のうちリスク方向の寄与が最大の特徴量
    Z = scaler.transform(X_te)                      # 標準化された特徴量値
    contrib = Z * coefs                             # 顧客別 x 特徴量別のリスク寄与度
    top10 = np.argsort(proba)[::-1][:10]
    print(f"    {'顧客ID':<9} {'解約確率':>7}  {'主な原因':<16} 推奨アクション")
    print("    " + "-" * 66)
    for i in top10:
        cust_id = df.loc[idx_te[i], "customer_id"]
        main_feat = features[int(np.argmax(contrib[i]))]
        action = ACTION_MAP[main_feat]
        print(f"    {cust_id:<9} {proba[i]:>6.1%}  {FEAT_JA[main_feat]:<16} {action}")
    print("\n    キャンペーン運用メモ:")
    print("      - 確率 80% 以上: 電話相談 / 50~80%: クーポン+メッセージ (確率区間別の強度調節)")
    print("      - 効果検証: リスク顧客の一部を無作為の対照群として残し、解約率を比較")
    print("\n    教訓: プロジェクトの完成は AUC ではなく、「月曜の朝、マーケティングチームが")
    print("          そのまま実行できる表」です。")


if __name__ == "__main__":
    main()
