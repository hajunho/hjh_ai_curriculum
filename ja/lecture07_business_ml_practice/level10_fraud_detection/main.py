"""
実践 3 — 不正取引検知: 教師あり学習と教師なし(IsolationForest)の並行運用。
同じアラート予算で 2 つのアプローチの検知力を公正に比較し、
「新種疑いゾーン」(教師ありは低く、教師なしは高く見た取引)を確認したうえで、
調査チームのアラート予算ごとの再現率の限界効用表で運用しきい値を設計します。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# level07 と同じ仮定: 履歴集計(tx_count_1h)システムがまだない状態で、
# 決済の瞬間のフィールド + level03 の派生特徴量で勝負します。
FEATURES = ["log_amount", "hour", "is_foreign", "is_night"]


def topk_stats(scores, y_true, k):
    """疑いスコア上位 k 件だけアラートしたときの (捕まえた不正, 適合率, 再現率)。"""
    idx = np.argsort(scores)[::-1][:k]
    caught = int(np.asarray(y_true)[idx].sum())
    total = int(np.asarray(y_true).sum())
    return caught, caught / k, caught / total


def main() -> None:
    print("=" * 68)
    print(" 実践 3: 不正取引検知 — 写真帳(教師あり) + 勘(教師なし) + 予算(運用)")
    print("=" * 68)

    # [1] データと特徴量 -----------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n[1] 取引 {len(df)}件 (不正 {y.mean():.2%}) / テスト {len(y_te)}件のうち不正 {y_te.sum()}件")
    print(f"    特徴量: {FEATURES}")

    # [2] 教師あり学習: ラベル(写真帳)で学ぶベテラン ---------------------------
    print("\n[2] 教師あり学習 — 過去の摘発ラベルで学習 (ロジスティック回帰)")
    sup = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=42, class_weight="balanced",
                                           max_iter=1000))
    sup.fit(X_tr, y_tr)
    sup_score = sup.predict_proba(X_te)[:, 1]
    print("    -> 取引ごとに不正確率(疑いスコア)を算出しました。")

    # [3] 教師なし: ラベルなしで「普段との違い」を探す新人 -----------------------
    print("\n[3] 教師なし — IsolationForest、ラベルをまったく使わない")
    iso = IsolationForest(n_estimators=200, contamination=0.015, random_state=42)
    iso.fit(X_tr)                      # y_tr なし! 正常取引の形だけを学習
    iso_score = -iso.score_samples(X_te)   # 大きいほど異常 (符号を反転)
    print("    -> 「何回で孤立するか」で異常スコアを算出しました。")

    # [4] 同じアラート予算で公正比較 --------------------------------------
    budget = 20
    print(f"\n[4] 公正比較: テスト {len(y_te)}件中、アラート予算 {budget}件のとき")
    for name, sc in [("教師あり(ラベル使用)", sup_score), ("IsolationForest(ラベルなし)", iso_score)]:
        caught, prec, rec = topk_stats(sc, y_te, budget)
        print(f"    {name:<26} 捕まえた不正 {caught:2d}件 | 適合率 {prec:5.1%} | 再現率 {rec:5.1%}")
    print("    => ラベルは強力です。しかし教師なしは「ラベルにない手口」という")
    print("       別の角度を見ています。勝者総取りではなく役割分担です。")

    # [5] 新種疑いゾーン: 教師ありは低く、教師なしだけ高く見た取引 ---------------
    print("\n[5] 新種疑いゾーン — 教師ありスコア下位 50% なのに教師なしスコア上位 5%")
    sup_rank = pd.Series(sup_score).rank(pct=True)
    iso_rank = pd.Series(iso_score).rank(pct=True)
    novel = (sup_rank < 0.5) & (iso_rank > 0.95)
    zone = X_te.reset_index(drop=True)[novel]
    zone_y = y_te.reset_index(drop=True)[novel]
    print(f"    該当取引 {novel.sum()}件 (うち実際の不正 {int(zone_y.sum())}件)")
    if len(zone) > 0:
        show = zone.head(3).copy()
        show["実際の不正"] = zone_y.head(3).values
        print(show.to_string())
    print("    => このゾーンは少量サンプル調査のキューへ送ります。調査結果が新しいラベルとなり、")
    print("       教師ありモデルを再教育する好循環が、実務運用の核心です。")

    # [6] アラート予算別の限界効用と運用しきい値 -------------------------------
    print("\n[6] アラート予算を増やすとどれだけ多く捕まえられるか (教師ありモデル基準)")
    print(f"    {'予算':>4} | {'捕捉不正':>6} | {'適合率':>6} | {'再現率':>6} | 予算に対応するしきい値")
    prev_caught = 0
    for k in [10, 20, 30, 40, 60]:
        caught, prec, rec = topk_stats(sup_score, y_te, k)
        th = np.sort(sup_score)[::-1][k - 1]
        gain = caught - prev_caught
        print(f"    {k:>4} | {caught:>5}件 | {prec:6.1%} | {rec:6.1%} | "
              f"スコア {th:.3f} 以上 (直前比 +{gain}件)")
        prev_caught = caught
    print("\n    報告文の例:")
    c20, p20, r20 = topk_stats(sup_score, y_te, 20)
    c40, p40, r40 = topk_stats(sup_score, y_te, 40)
    print(f'    「現在の人員(アラート20件)で不正の {r20:.0%} を捕捉しています。調査人員を2倍に')
    print(f'     増やすと(40件)、再現率は {r40:.0%} になりますが適合率は {p40:.0%} に下がります。」')
    print("\n    教訓: しきい値は統計からは生まれません。人員計画(予算)から生まれます。")


if __name__ == "__main__":
    main()
