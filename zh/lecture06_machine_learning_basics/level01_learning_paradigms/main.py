"""
level01 — 监督、无监督、强化学习迷你体验

用玩具大小的例子把三种学习范式各跑一遍。
  [1] 监督学习   = 一对一辅导 (给出正确标签，训练流失预测)
  [2] 无监督学习 = 自习小组   (没有答案，把客户分组)
  [3] 强化学习   = 训练小狗   (只看奖励找最优优惠券的老虎机)
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
    """[1] 监督学习: 把题目(客户特征)和答案(churned)一起给它教。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["churned"] for r in rows])

    scaler = StandardScaler()
    model = LogisticRegression(random_state=0)
    model.fit(scaler.fit_transform(X), y)          # <- 传入答案 y (辅导)

    print("[1] 监督学习 = 一对一辅导 (看着答案学习规律)")
    print("    训练: 2000 名客户的特征 + 是否流失的答案")
    print("    考试: 预测 3 位没见过的新客户的流失概率")
    new_customers = [
        ("勤恳使用的客户", [36, 28, 0, 1]),
        ("低使用 + 投诉不断", [3, 2, 5, 0]),
        ("普普通通的客户", [18, 15, 1, 1]),
    ]
    for name, feat in new_customers:
        p = model.predict_proba(scaler.transform([feat]))[0, 1]
        verdict = "流失风险" if p >= 0.5 else "预计留存"
        print(f"      {name:12s} -> 流失概率 {p:5.1%} ({verdict})")
    print()


def demo_unsupervised() -> None:
    """[2] 无监督学习: 去掉答案列，只吩咐'把相似的客户归到一起'。"""
    rows = hjh_data.churn_table(n=2000, seed=7)
    cols = ["tenure_months", "usage_days_30d", "support_calls_30d", "monthly_fee"]
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)  # 没有答案!

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=0)
    labels = km.fit_predict(Xs)                     # <- 没有 y (自习)

    print("[2] 无监督学习 = 自习小组 (没有答案，自己找结构)")
    print("    k-means 把 2000 名客户分成 3 组。各组平均画像:")
    print("      组别  人数   入网月数  使用天数  来电数  月费")
    for g in range(3):
        member = X[labels == g]
        m = member.mean(axis=0)
        print(f"      {g:>2}   {len(member):>4}   {m[0]:7.1f}  {m[1]:7.1f}  {m[2]:6.2f}  {m[3]:7.0f}")
    print("    -> 给每组起名字 (例: '优质'、'休眠风险') 是人的活儿。")
    print()


def demo_reinforcement() -> None:
    """[3] 强化学习: 不知道哪种优惠券好，发出去看反应(奖励)来学。
    用纯 Python 实现多臂老虎机 + epsilon-greedy。"""
    rng = random.Random(42)
    true_rates = {"A. 95折券": 0.05, "B. 免运费": 0.12, "C. 买一送一": 0.08}
    coupons = list(true_rates)
    counts = {c: 0 for c in coupons}       # 每种券发出的次数
    wins = {c: 0 for c in coupons}         # 每种券被使用的次数
    EPSILON = 0.1                          # 10% 的概率探索(随便试一张)
    history = []

    for _ in range(1000):
        if rng.random() < EPSILON or not any(counts.values()):
            choice = rng.choice(coupons)                       # 探索
        else:
            choice = max(coupons, key=lambda c: wins[c] / max(counts[c], 1))  # 利用
        reward = 1 if rng.random() < true_rates[choice] else 0  # 客户反应 = 零食
        counts[choice] += 1
        wins[choice] += reward
        history.append(reward)

    print("[3] 强化学习 = 训练小狗 (尝试 -> 奖励 -> 改进行为)")
    print("    3 种优惠券的真实使用率保密。给 1000 人发券的过程中自己学:")
    for c in coupons:
        est = wins[c] / max(counts[c], 1)
        print(f"      {c:10s} 发出 {counts[c]:>4} 次, 估计使用率 {est:5.1%} (真实 {true_rates[c]:.0%})")
    early = sum(history[:100]) / 100
    late = sum(history[-100:]) / 100
    print(f"    前 100 次平均奖励 {early:.2f}  ->  最后 100 次平均奖励 {late:.2f}")
    print("    -> 从没有人教过它正确答案，却收敛到了最优行为(B)。")
    print()


if __name__ == "__main__":
    np.random.seed(0)   # 可复现
    demo_supervised()
    demo_unsupervised()
    demo_reinforcement()

    print("[4] 总结")
    print("    范式         给定的东西           学到的东西")
    print("    监督学习     输入 + 正确标签      输入 -> 答案的对应 (辅导)")
    print("    无监督学习   只有输入             隐藏结构/分组      (自习)")
    print("    强化学习     环境 + 奖励          奖励最大化的行为   (训练小狗)")
    print("    实战 ML 的大多数是监督学习，从 level03 开始正式展开。")
