"""
level08 — 随机森林与集成学习: 集体智慧实验

用 churn_table 比较单棵决策树 vs 随机森林。
  [2] 用 15 行亲手实现装袋(自助抽样 + 投票)，确认'平均的力量'
  [3] 性能比较 (AUC / 召回率)
  [4] 稳定性比较: 换 12 次划分，测量 AUC 的晃动(标准差)
  [5] 特征重要性: 一棵树 vs 森林
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def bagged_predict(X_tr, y_tr, X_te, n_trees: int, seed: int = 0) -> np.ndarray:
    """亲手实现迷你装袋: 用有放回抽样的数据养深树们，以概率平均(投票)作答。
    sklearn RandomForest = 在此之上再加'每次分叉的特征随机挑选'。"""
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    probas = []
    for i in range(n_trees):
        idx = rng.choice(n, n, replace=True)          # 自助法: 同样大小的有放回抽样
        tree = DecisionTreeClassifier(random_state=i)  # 深度无限制(低偏差·高方差)
        tree.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        probas.append(tree.predict_proba(X_te)[:, 1])
    return np.mean(probas, axis=0)                     # 平均 = 抵消晃动(方差)


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 数据准备 --------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] 数据: 客户 {len(df)} 名, 流失率 {y.mean():.1%}\n")

    # [2] 亲手实现迷你装袋 -------------------------------------------------
    single = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
    auc_single = roc_auc_score(y_te, single.predict_proba(X_te)[:, 1])
    print("[2] 亲手实现装袋 — 猜牛体重大赛的原理")
    print(f"    1 棵深树                 测试 AUC = {auc_single:.3f}")
    for n_trees in [5, 25]:
        auc_bag = roc_auc_score(y_te, bagged_predict(X_tr, y_tr, X_te, n_trees))
        print(f"    同样的树 {n_trees:>2} 棵投票        测试 AUC = {auc_bag:.3f}")
    print("    -> 个体(树)尽管过拟合，经历各异的多数取平均就变强。\n")

    # [3] 性能比较: 单棵树 vs 随机森林 --------------------------------
    print("[3] 性能比较 (测试集)")
    models = {
        "树(深度 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
        "树(无限制)": DecisionTreeClassifier(random_state=0),
        "随机森林(300 棵)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
    }
    print("    模型                     AUC     召回率(阈值 0.5)")
    forest = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        rec = recall_score(y_te, m.predict(X_te))
        print(f"    {name:20s} {auc:.3f}      {rec:6.1%}")
        if isinstance(m, RandomForestClassifier):
            forest = m
    print()

    # [4] 稳定性比较: 换着划分测量晃动 ------------------------------
    print("[4] 稳定性比较 — 换 12 次划分反复测量 AUC")
    aucs_tree, aucs_rf = [], []
    for rep in range(12):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25,
                                          random_state=rep, stratify=y)
        t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xa, ya)
        f = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1).fit(Xa, ya)
        aucs_tree.append(roc_auc_score(yb, t.predict_proba(Xb)[:, 1]))
        aucs_rf.append(roc_auc_score(yb, f.predict_proba(Xb)[:, 1]))
    aucs_tree, aucs_rf = np.array(aucs_tree), np.array(aucs_rf)
    print(f"    单棵树      AUC 平均 {aucs_tree.mean():.3f} ± 标准差 {aucs_tree.std():.3f}")
    print(f"    随机森林    AUC 平均 {aucs_rf.mean():.3f} ± 标准差 {aucs_rf.std():.3f}")
    print(f"    森林赢的次数: {int((aucs_rf > aucs_tree).sum())}/12")
    print("    -> 实战信任的核心不只是平均性能，还有'晃动(标准差)'小。\n")

    # [5] 特征重要性: 一棵树 vs 森林 ------------------------------------------
    print("[5] 特征重要性比较 (对不纯度下降的贡献, 合计 1)")
    tree5 = models["树(深度 5)"]
    print("    特征                 1 棵树     森林 300 棵")
    for i in np.argsort(-forest.feature_importances_):
        print(f"    {FEATURES[i]:18s}   {tree5.feature_importances_[i]:.3f}       "
              f"{forest.feature_importances_[i]:.3f}")
    print("    -> 森林的重要性是多棵树的平均，所以更稳定。")
    print("       (依然只是'被用得多的程度'，不是因果 — 公正的测量见 level11)")
