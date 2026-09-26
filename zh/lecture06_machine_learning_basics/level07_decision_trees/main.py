"""
level07 — 决策树: 用"二十个问题"找流失客户

用 churn_table 训练决策树，
  - 手算基尼不纯度，理解'好问题'的标准
  - 把学到的树打印成人能读懂的规则语句来解读
  - 换着深度观察过拟合的岔路口。
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def gini(labels: np.ndarray) -> float:
    """基尼不纯度: 房间里全是一种人就是 0，一半一半就是 0.5。这 8 行是树的全部数学。"""
    if len(labels) == 0:
        return 0.0
    p = labels.mean()               # 流失比例
    return 1.0 - (p ** 2 + (1 - p) ** 2)


def split_gain(y: np.ndarray, mask: np.ndarray) -> float:
    """提问(mask)之后，两个房间按人数加权的平均不纯度降了多少。"""
    left, right = y[mask], y[~mask]
    after = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
    return gini(y) - after


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 数据准备 (树不需要标准化: 阈值比较模型) -----------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] 数据: 客户 {len(df)} 名, 流失率 {y.mean():.1%} — 训练 {len(X_tr)} / 测试 {len(X_te)}\n")

    # [2] 手算基尼不纯度: 什么是好问题? -------------------------------
    y_arr = y_tr.to_numpy()
    print("[2] 基尼不纯度 — '房间有多混杂'")
    print(f"    提问前整个房间的不纯度: {gini(y_arr):.4f}")
    for feat, th in [("usage_days_30d", 9.5), ("monthly_fee", 15000), ("support_calls_30d", 1.5)]:
        gain = split_gain(y_arr, (X_tr[feat] <= th).to_numpy())
        print(f"    问题 \"{feat} <= {th}\" 带来的不纯度下降: {gain:.4f}")
    print("    -> 树会把所有特征 x 所有阈值试一遍，选降幅最大的问题。")
    print("       (只是 level00 的阈值搜索被递归重复 — 不是魔法)\n")

    # [3] 训练深度 3 的树 + 把规则打印成文本 ------------------------------
    # class_weight="balanced": 把流失者(15%)看得比留存者更重，
    # 让少数类(流失)一侧的规则在树里显形。
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                                  class_weight="balanced", random_state=0)
    tree.fit(X_tr, y_tr)
    print("[3] 深度 3 树的规则全文 (export_text)")
    print(export_text(tree, feature_names=FEATURES))

    # 把叶子(leaf)房间翻译成中文汇报语句
    leaf_id = tree.apply(X_tr)
    print("    叶子房间汇总 (按训练数据):")
    rows = []
    for leaf in np.unique(leaf_id):
        members = y_tr[leaf_id == leaf]
        rows.append((leaf, len(members), members.mean()))
    for leaf, n, rate in sorted(rows, key=lambda r: -r[2])[:3]:
        print(f"      高危房间 #{leaf}: {n} 名, 流失率 {rate:.1%}")
    print("    -> 高危房间的路径(在上面的规则树里追踪)可以原样反向输入为 CRM 规则。\n")

    # [4] 各深度的训练/测试成绩 — 过拟合的岔路口 ----------------------------
    print("[4] 深度与过拟合 (level04 实验 A 重访)")
    print("    深度      训练准确率   测试准确率    测试召回率")
    for depth in [1, 3, 5, 10, None]:
        t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        rec = recall_score(y_te, t.predict(X_te))
        label = "无限制" if depth is None else f"{depth:>4}"
        print(f"    {label:>6}      {t.score(X_tr, y_tr):6.1%}       {t.score(X_te, y_te):6.1%}        {rec:6.1%}")
    print("    -> 深度无限制 = 训练 100% 死记硬背，测试反而吃亏。\n")

    # [5] 特征重要性 ---------------------------------------------------------
    print("[5] 特征重要性 (对不纯度下降的贡献比例, 合计 1)")
    order = np.argsort(-tree.feature_importances_)
    for i in order:
        bar = "#" * int(tree.feature_importances_[i] * 40)
        print(f"    {FEATURES[i]:18s} {tree.feature_importances_[i]:.3f} {bar}")
    print("    -> 是'被用得多的特征'，不是'原因排行' (解读要小心, level11)。")
