"""
level05 — 逻辑回归: 订阅流失概率模型

用 churn_table(2000 名客户)做一个输出'流失概率'的分类模型。
  - Sigmoid: 把线性分数 z 换算成 0~1 概率的漏斗
  - 系数 -> 优势比(odds ratio)翻译: "来电每 +1 单位 -> 流失几率 N 倍"
  - 阈值(threshold)不由模型定，由业务定
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
FEATURE_ZH = {"tenure_months": "入网月数", "monthly_fee": "月费",
              "usage_days_30d": "使用天数(30天)", "support_calls_30d": "来电数(30天)",
              "plan_changes": "套餐变更", "auto_pay": "自动扣款"}


def sigmoid(z: float) -> float:
    """概率漏斗: 再大的分数也压进 0~1 之间。"""
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 数据准备 + 划分 -------------------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X = df[FEATURES].astype(float)     # customer_id 是无意义的列，剔除
    y = df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)  # 保持流失率的划分
    print("[1] 数据: 订阅客户 2000 名, 流失率 {:.1%}".format(y.mean()))
    print(f"    训练 {len(X_tr)} 名 / 测试 {len(X_te)} 名 (遵守 level04 原则)\n")

    # [2] 观察 Sigmoid 漏斗 ----------------------------------------------
    print("[2] Sigmoid: 线性分数 z -> 概率 p")
    for z in [-4, -2, 0, 2, 4]:
        print(f"    z = {z:+d}  ->  p = {sigmoid(z):5.1%}")
    print("    -> 0 分就是一半一半(50%)，±4 分基本就是定局。\n")

    # [3] 训练 ---------------------------------------------------------------
    # Pipeline: 缩放器只用训练集 fit -> 自动防泄漏 (level04)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba_te = model.predict_proba(X_te)[:, 1]        # 流失概率
    pred_05 = (proba_te >= 0.5).astype(int)
    print("[3] 逻辑回归训练完成 — 测试成绩 (阈值 0.5)")
    print(f"    准确率 {accuracy_score(y_te, pred_05):.1%} / "
          f"精确率 {precision_score(y_te, pred_05):.1%} / "
          f"召回率 {recall_score(y_te, pred_05):.1%}\n")

    # [4] 系数 -> 优势比翻译 --------------------------------------------------
    clf = model.named_steps["clf"]
    print("[4] 系数解读 (基于标准化特征: '增加 1 个标准差'的效应)")
    print("    特征                系数      优势比    解读")
    order = np.argsort(-np.abs(clf.coef_[0]))
    for i in order:
        coef = clf.coef_[0][i]
        orat = math.exp(coef)
        direction = "流失风险增加" if coef > 0 else "流失风险降低"
        print(f"    {FEATURE_ZH[FEATURES[i]]:14s} {coef:+7.3f}   {orat:6.2f}倍   {direction}")
    print("    -> 检查方向是否与数据生成器里埋的真实信号一致")
    print("       (使用天数↓, 来电↑, 自动扣款-)。(是相关，不是因果证明)\n")

    # [5] 单条预测 + 阈值实验 ---------------------------------------------
    print("[5] 单个客户的预测与阈值这个业务决策")
    samples = pd.DataFrame([
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 3, "monthly_fee": 29900, "usage_days_30d": 2,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])[FEATURES].astype(float)
    names = ["勤恳使用的客户", "新客+低使用+投诉不断", "普普通通的客户"]
    for name, p in zip(names, model.predict_proba(samples)[:, 1]):
        print(f"    {name:16s} 流失概率 {p:5.1%}")
    print()
    print("    换个阈值，同一个模型也会做出不同的决定:")
    print("    阈值    判为风险人数    精确率   召回率")
    for th in [0.5, 0.3]:
        pred = (proba_te >= th).astype(int)
        print(f"     {th:.1f}        {pred.sum():>4} 名      "
              f"{precision_score(y_te, pred):6.1%}  {recall_score(y_te, pred):6.1%}")
    print("    -> 调低阈值，漏掉的流失者变少(召回率↑)，白跑的回访变多(精确率↓)。")
    print("       0.5 只是惯例 — 该由回访成本和客户价值来定阈值。")
