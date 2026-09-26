"""
不平衡数据(欺诈 1.5%)应对手段对比实验。
默认模型 / class_weight / 过采样 / 阈值调整 四种方式,
用表格确认召回率-精确率的取舍关系,
并在'一天的告警处理预算'约束下挑一个合理的阈值。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             average_precision_score, precision_recall_curve)

# 假设历史聚合系统还没上线, 只能用'刷卡瞬间就知道的字段'
# (加上聚合特征 tx_count_1h 之后问题会变得多简单, 留给'动手试试')
FEATURES = ["amount", "hour", "is_foreign"]


def new_model(**kw):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, max_iter=1000, **kw))


def show(name, y_true, y_pred):
    print(f"    {name:<28} 准确率 {accuracy_score(y_true, y_pred):.3f} | "
          f"召回率 {recall_score(y_true, y_pred):.3f} | "
          f"精确率 {precision_score(y_true, y_pred, zero_division=0):.3f} | "
          f"告警 {int(y_pred.sum())}单")


def main() -> None:
    print("=" * 66)
    print(" 不平衡数据: 抓住 1.5% 欺诈的四种方法")
    print("=" * 66)

    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    print(f"\n    数据: {len(df)}笔, 欺诈 {y.sum()}笔({y.mean():.2%}) / "
          f"测试 {len(y_te)}笔里有欺诈 {y_te.sum()}笔")

    # [1] 默认设置: 躺平的模型 -----------------------------------------
    print("\n[1] 默认设置 (阈值 0.5, 不加权)")
    base = new_model()
    base.fit(X_tr, y_tr)
    show("默认逻辑回归", y_te, base.predict(X_te))
    missed = int(y_te.sum()) - int((base.predict(X_te) & y_te).sum())
    print(f"    => 准确率 99% 上下, 看着很漂亮, 但 {int(y_te.sum())}笔欺诈里漏掉了 {missed}笔。")
    print("       (极端点说'全判正常'的准确率也有 98.6% — 准确率不是这个问题的指标)")

    # [2] class_weight: 改罚分表 -------------------------------------
    print("\n[2] class_weight='balanced' — 对少数类失误重罚")
    weighted = new_model(class_weight="balanced")
    weighted.fit(X_tr, y_tr)
    show("加权模型", y_te, weighted.predict(X_te))

    # [3] 过采样: 只对训练数据! ------------------------------------
    print("\n[3] 手动过采样 — 划分'之后'只复制训练数据里的欺诈行")
    rng = np.random.default_rng(42)
    pos_idx = y_tr[y_tr == 1].index
    ratio = int((y_tr == 0).sum() / (y_tr == 1).sum())  # 大致能达到平衡的倍数
    dup_idx = rng.choice(pos_idx, size=len(pos_idx) * (ratio - 1), replace=True)
    X_bal = pd.concat([X_tr, X_tr.loc[dup_idx]])
    y_bal = pd.concat([y_tr, y_tr.loc[dup_idx]])
    print(f"    训练数据 {len(y_tr)}笔 -> {len(y_bal)}笔 (欺诈比例 {y_bal.mean():.1%})")
    over = new_model()
    over.fit(X_bal, y_bal)
    show("过采样模型", y_te, over.predict(X_te))
    print("    => 和 [2] 效果相仿。两者都是'改罚分表'的变体。")
    print("       (测试比例绝对不许动 — 考试要按真实世界的比例来)")

    # [4] 阈值调整: 同一个模型, 不同运营点 ------------------------------
    print("\n[4] 阈值调整 — 就用默认模型这一个, 只挪动运营点")
    proba = base.predict_proba(X_te)[:, 1]
    print(f"    {'阈值':>6} | {'召回率':>6} | {'精确率':>6} | 告警笔数")
    rows = []
    for th in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
        pred = (proba >= th).astype(int)
        r = recall_score(y_te, pred)
        p = precision_score(y_te, pred, zero_division=0)
        rows.append((th, r, p, int(pred.sum())))
        print(f"    {th:6.2f} | {r:6.1%} | {p:6.1%} | {int(pred.sum()):4d}单")
    print("    => 越往下召回率越涨, 精确率越跌。天下没有免费的午餐。")

    # [5] PR 曲线摘要与告警预算 -----------------------------------------
    print("\n[5] PR 曲线摘要 + 用'告警预算'挑运营点")
    pr_auc = average_precision_score(y_te, proba)
    prec_c, rec_c, th_c = precision_recall_curve(y_te, proba)
    print(f"    PR-AUC(平均精确率) = {pr_auc:.3f}  (阈值全区间的综合得分)")
    budget = 15  # 调查团队在这段测试期里能处理的告警单数
    order = np.argsort(proba)[::-1]
    top = order[:budget]
    caught = int(y_te.iloc[top].sum())
    th_budget = proba[order[budget - 1]]
    print(f"    约束: 调查团队可处理量 = {budget}单")
    print(f"    => 只对可疑分数最高的 {budget}笔告警 (对应阈值约 {th_budget:.2f})")
    print(f"       抓到欺诈 {caught}笔 / 共 {int(y_te.sum())}笔 "
          f"(召回率 {caught / y_te.sum():.1%}, 精确率 {caught / budget:.1%})")
    print("\n    教训: 选哪个运营点不是模型决定的, 是业务(人力·成本)决定的。")
    print("          模型的活儿, 到'画出一条好曲线'为止。")


if __name__ == "__main__":
    main()
