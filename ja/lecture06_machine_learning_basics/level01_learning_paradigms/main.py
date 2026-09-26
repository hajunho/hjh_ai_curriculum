"""
level01 — 教師あり・教師なし・強化学習のミニ体験

3 つの学習パラダイムを、おもちゃサイズの例で一つずつ動かしてみます。
  [1] 教師あり学習 = 家庭教師     (正解ラベルを与えて解約予測を訓練)
  [2] 教師なし学習 = 自主ゼミ     (正解なしで顧客をグループ分け)
  [3] 強化学習     = 犬のしつけ   (報酬だけを頼りに最適クーポンを探すバンディット)
"""

import pathlib
import random
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "usage_days_30d", "support_calls_30d", "auto_pay"]


def demo_supervised() -> None:
    """[1] 教師あり学習: 問題(顧客の特徴)と正解(churned)をセットで与えて教える。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["churned"] for r in rows])

    scaler = StandardScaler()
    model = LogisticRegression(random_state=0)
    model.fit(scaler.fit_transform(X), y)          # <- 正解 y を渡す (家庭教師)

    print("[1] 教師あり学習 = 家庭教師 (正解を見ながらパターンを学習)")
    print("    訓練: 顧客 2000人の特徴 + 解約したかどうかの正解")
    print("    試験: 初めて見る顧客 3人の解約確率を予測")
    new_customers = [
        ("優良利用の顧客", [36, 28, 0, 1]),
        ("低利用 + 問い合わせ殺到", [3, 2, 5, 0]),
        ("ごく平均的な顧客", [18, 15, 1, 1]),
    ]
    for name, feat in new_customers:
        p = model.predict_proba(scaler.transform([feat]))[0, 1]
        verdict = "解約リスクあり" if p >= 0.5 else "継続の見込み"
        print(f"      {name:12s} -> 解約確率 {p:5.1%} ({verdict})")
    print()


def demo_unsupervised() -> None:
    """[2] 教師なし学習: 正解の列を外し、「似た顧客同士でまとめてみて」とだけ頼む。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    cols = ["tenure_months", "usage_days_30d", "support_calls_30d", "monthly_fee"]
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)  # 正解なし!

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0)
    labels = km.fit_predict(Xs)                     # <- y がない (自主ゼミ)

    print("[2] 教師なし学習 = 自主ゼミ (正解なしで構造を見つける)")
    print("    k-means が顧客 2000人を 3グループに分けた。グループ別の平均プロフィール:")
    print("      グループ  人数   契約月数  利用日数  問い合わせ  月額料金")
    for g in range(3):
        member = X[labels == g]
        m = member.mean(axis=0)
        print(f"      {g:>2}   {len(member):>4}   {m[0]:7.1f}  {m[1]:7.1f}  {m[2]:6.2f}  {m[3]:7.0f}")
    print("    -> 各グループに名前を付けるのは (例: 「優良」「休眠リスク」) 人間の仕事です。")
    print()


def demo_reinforcement() -> None:
    """[3] 強化学習: どのクーポンが良いか知らないまま、送ってみて反応(報酬)から学ぶ。
    多腕バンディット + epsilon-greedy を純粋な Python で実装。"""
    rng = random.Random(42)
    true_rates = {"A. 5%割引": 0.05, "B. 送料無料": 0.12, "C. 1+1": 0.08}
    coupons = list(true_rates)
    counts = {c: 0 for c in coupons}       # 各クーポンを送った回数
    wins = {c: 0 for c in coupons}         # 各クーポンが使われた回数
    EPSILON = 0.1                          # 10% の確率で探索(でたらめに試す)
    history = []

    for _ in range(1000):
        if rng.random() < EPSILON or not any(counts.values()):
            choice = rng.choice(coupons)                       # 探索
        else:
            choice = max(coupons, key=lambda c: wins[c] / max(counts[c], 1))  # 活用
        reward = 1 if rng.random() < true_rates[choice] else 0  # 顧客の反応 = おやつ
        counts[choice] += 1
        wins[choice] += reward
        history.append(reward)

    print("[3] 強化学習 = 犬のしつけ (試す -> 報酬 -> 行動を改善)")
    print("    クーポン 3種の本当の利用率は秘密。1000人に送りながら自力で学習:")
    for c in coupons:
        est = wins[c] / max(counts[c], 1)
        print(f"      {c:10s} 発送 {counts[c]:>4}回, 推定利用率 {est:5.1%} (真実 {true_rates[c]:.0%})")
    early = sum(history[:100]) / 100
    late = sum(history[-100:]) / 100
    print(f"    序盤 100回の平均報酬 {early:.2f}  ->  最後の 100回の平均報酬 {late:.2f}")
    print("    -> 正解を一度も教えていないのに、最善の行動 (B) に収束します。")
    print()


if __name__ == "__main__":
    np.random.seed(0)   # 再現性
    demo_supervised()
    demo_unsupervised()
    demo_reinforcement()

    print("[4] まとめ")
    print("    パラダイム       与えられるもの        学ぶもの")
    print("    教師あり学習     入力 + 正解ラベル     入力 -> 正解の対応 (家庭教師)")
    print("    教師なし学習     入力のみ              隠れた構造/グループ (自主ゼミ)")
    print("    強化学習         環境 + 報酬           報酬を最大化する行動 (犬のしつけ)")
    print("    実務の ML の大半は教師あり学習で、level03 から本格的に扱います。")
