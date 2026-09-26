"""
level11 — 梯度提升与模型解释

用 churn_table 建提升模型，并连'为什么?'一起回答。
  [2] 逻辑回归 vs 随机森林 vs HistGradientBoosting 性能比较
  [3] 学习率(错题本吸收强度) x 树数迷你实验
  [4] 置换重要性: 洗牌一个特征, AUC 塌多少
  [5] 部分依赖: 预测概率随特征值变化的形状 (文本图)
  [6] 单条预测解释: 逐特征换成平均值, 分解判定依据
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
FEATURE_ZH = {"tenure_months": "入网月数", "monthly_fee": "月费",
              "usage_days_30d": "使用天数", "support_calls_30d": "来电数",
              "plan_changes": "套餐变更", "auto_pay": "自动扣款"}


def explain_one(model, x_row: pd.DataFrame, baseline: pd.Series) -> list[tuple[str, float]]:
    """单条预测解释器(12 行): 把一个特征换成'平均客户'的值，
    逐特征量流失概率变了多少。变化幅度大的特征 = 判定的依据。"""
    p0 = model.predict_proba(x_row)[0, 1]
    contribs = []
    for f in FEATURES:
        x_mod = x_row.copy()
        x_mod[f] = baseline[f]
        p_mod = model.predict_proba(x_mod)[0, 1]
        contribs.append((f, p0 - p_mod))     # 为 + 表示该特征是推高风险的依据
    return sorted(contribs, key=lambda t: -abs(t[1]))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 数据 -------------------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES].astype(float), df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    print(f"[1] 数据: 客户 {len(df)} 名, 流失率 {y.mean():.1%}\n")

    # [2] 三个模型的毕业考试 ----------------------------------------------------
    print("[2] 学过的模型们的性能比较 (测试 AUC)")
    models = {
        "逻辑回归 (level05)": Pipeline([
            ("s", StandardScaler()), ("c", LogisticRegression(random_state=0))]),
        "随机森林 (level08)": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=0, early_stopping=True),
    }
    boost = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {name:24s} AUC = {auc:.3f}")
        if isinstance(m, HistGradientBoostingClassifier):
            boost = m
    print("    -> 出人意料: 逻辑回归第一! 因为这份合成数据的真实结构就是")
    print("       '线性评分表'(对数几率的线性式)。两条教训:")
    print("       (1) 赢的不是强模型, 而是契合数据结构的模型。")
    print("       (2) 在非线性、交互多的现实数据上, 提升常常领先。\n")

    # [3] 学习率 x 树数迷你实验 -------------------------------------------
    print("[3] 学习率(错题本吸收强度)实验 — 刹车与课时数的交易")
    print("    学习率   最大树数   测试 AUC")
    for lr, n_iter in [(1.0, 50), (0.3, 100), (0.1, 200), (0.03, 500)]:
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=n_iter, random_state=0,
            early_stopping=True).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"    {lr:<6}    {n_iter:>5}      {auc:.3f}")
    print("    -> 压低学习率需要更多棵树, 但结果趋于稳定。")
    print("       (learning_rate 就是 level09 的正则化哲学来到了提升)\n")

    # [4] 置换重要性 ----------------------------------------------------------
    print("[4] 置换重要性 — 洗牌一个特征时测试 AUC 的跌幅")
    perm = permutation_importance(boost, X_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=0)
    print("    特征           AUC 跌幅(平均±标准差)")
    for i in np.argsort(-perm.importances_mean):
        bar = "#" * max(0, int(perm.importances_mean[i] * 200))
        print(f"    {FEATURE_ZH[FEATURES[i]]:10s}   {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}  {bar}")
    print("    -> '拿测试数据审问训练完的模型'的方式, 不挑模型种类,")
    print("       比不纯度重要性(level07~08)更适合诚实汇报。\n")

    # [5] 部分依赖 -------------------------------------------------------------
    print("[5] 部分依赖 — 其他条件固定, 只动一个特征时的平均流失概率")
    for feat in ["usage_days_30d", "support_calls_30d"]:
        # method="brute": 按 predict_proba 计算 -> 结果以'概率'为单位
        pd_res = partial_dependence(boost, X_te, [feat], kind="average",
                                    grid_resolution=7, method="brute")
        grid = pd_res["grid_values"][0]
        avg = pd_res["average"][0]
        print(f"    {FEATURE_ZH[feat]} ({feat})")
        for g, v in zip(grid, avg):
            bar = "#" * int(v * 60)
            print(f"      值 {g:6.1f} -> 平均概率 {v:5.1%} {bar}")
    print("    -> 曲线的'形状'(哪个区间骤变)比逻辑回归的一个系数")
    print("       信息丰富得多。不过是关联, 不是因果。\n")

    # [6] 单条预测解释 --------------------------------------------------------
    proba_te = boost.predict_proba(X_te)[:, 1]
    idx = int(np.argmax(proba_te))                     # 测试集中风险最高的客户
    x_row = X_te.iloc[[idx]]
    baseline = X_tr.mean()                             # '平均客户'
    print("[6] 单条预测解释 — 分解风险最高的 1 位客户的判定依据")
    print(f"    这位客户的流失概率: {proba_te[idx]:.1%} (相对平均客户基准值)")
    print("    特征(客户值 -> 换成平均值)          概率变化")
    for f, delta in explain_one(boost, x_row, baseline):
        if abs(delta) < 0.005:
            direction = "影响甚微"
        else:
            direction = "推高风险的依据" if delta > 0 else "压低风险的因素"
        print(f"    {FEATURE_ZH[f]:10s} ({x_row[f].iloc[0]:>8.1f} -> {baseline[f]:>8.1f})   "
              f"{delta:+6.1%}p  {direction}")
    print("    -> '把来电数拉回平均水平, 概率会大幅下降'这类句子,")
    print("       就是回访话术的依据。(专业工具 SHAP 也是同一原理的精细化)")
    print()
    print("[7] lecture06 完结! 把性能(提升)和解释(重要性·部分依赖·案例说明)")
    print("    成套汇报出来, 模型才会被组织采纳。")
