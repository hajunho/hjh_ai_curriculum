"""
level06 — 评估指标: 准确率的陷阱

在欺诈比例约 1.5% 的银行卡交易数据上
  - 演示无条件喊"正常"的空壳模型拿到 98.5% 准确率的陷阱
  - 用混淆矩阵 / 精确率 / 召回率 / F1 / ROC-AUC 正确读懂模型。
用随机抽对模拟直接验证 AUC 的概率解释
("随机欺诈·正常对中欺诈方概率更高的概率")。
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

# tx_count_1h 在这份合成数据里是能'完美'分开欺诈的特征，故剔除。
# (现实中看到这么完美的特征，先别庆祝性能，先怀疑泄漏(leakage)!)
FEATURES = ["amount", "hour", "is_foreign"]


def print_metrics(name: str, y_true, y_pred) -> None:
    """一次输出混淆矩阵和四大指标。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"    {name}")
    print(f"      混淆矩阵: TP={tp:>3} (抓对了)   FN={fn:>3} (漏掉!)")
    print(f"                FP={fp:>3} (误报)     TN={tn:>4} (放行)")
    print(f"      准确率 {acc:.1%} / 精确率 {prec:.1%} / 召回率 {rec:.1%} / F1 {f1:.3f}")


def auc_by_sampling(y_true, proba, n_pairs: int = 10_000, seed: int = 0) -> float:
    """用模拟验证 AUC 的定义:
    随机抽 1 笔欺诈和 1 笔正常，统计'欺诈方概率更高'的比例。"""
    rng = np.random.default_rng(seed)
    pos = proba[y_true == 1]
    neg = proba[y_true == 0]
    p = rng.choice(pos, n_pairs)
    n = rng.choice(neg, n_pairs)
    return float(np.mean((p > n) + 0.5 * (p == n)))


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 数据准备 --------------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=8000, seed=11))
    X, y = df[FEATURES].astype(float), df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)  # 保持不平衡比例的划分
    print(f"[1] 数据: 银行卡交易 {len(df):,} 笔, 欺诈比例 {y.mean():.2%}")
    print(f"    训练 {len(X_tr):,} 笔 / 测试 {len(X_te):,} 笔 (用 stratify 保持比例)\n")

    # [2] 空壳模型 — 全部预测为正常 ------------------------------------------
    dummy_pred = np.zeros(len(y_te), dtype=int)
    print("[2] 空壳模型: 无条件硬说'正常'")
    print_metrics("全预测正常:", y_te, dummy_pred)
    print("      -> 准确率 98.5% 却 0 笔欺诈落网。这就是'准确率的陷阱'的真面目。")
    print("         不平衡数据上，准确率是写在脚注而不是第一行的指标。\n")

    # [3] 真模型 — 逻辑回归 -------------------------------------------
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(random_state=0))])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    print("[3] 真模型: 逻辑回归 (阈值 0.5)")
    print_metrics("逻辑回归:", y_te, (proba >= 0.5).astype(int))
    print("      -> 准确率和空壳只差几个百分点，混淆矩阵却是两个世界。\n")

    # [4] 阈值跷跷板: 精确率 vs 召回率 ---------------------------------------
    print("[4] 扫阈值看精确率-召回率跷跷板")
    print("    阈值    警报笔数   精确率    召回率")
    for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
        pred = (proba >= th).astype(int)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        print(f"     {th:.1f}      {pred.sum():>4}     {prec:6.1%}   {rec:6.1%}")
    print("    -> 调高灵敏度(阈值↓)，漏检变少，误报变多。")
    print("       坐在跷跷板哪一头，由知道'漏检成本 vs 误报成本'的人决定。\n")

    # [5] ROC-AUC: 与阈值无关的分辨力 -------------------------------------
    auc_dummy = roc_auc_score(y_te, np.zeros(len(y_te)))
    auc_model = roc_auc_score(y_te, proba)
    auc_sim = auc_by_sampling(y_te.to_numpy(), proba)
    print("[5] ROC-AUC — 定阈值之前'模型的档次'")
    print(f"    空壳模型 AUC    : {auc_dummy:.3f} (掷硬币水平)")
    print(f"    逻辑回归 AUC    : {auc_model:.3f}")
    print(f"    模拟验证        : 随机 (欺诈, 正常) 对 10,000 组中")
    print(f"                      欺诈方概率更高的比例 = {auc_sim:.3f}  (与 AUC 一致)")
    print()
    print("[6] 总结: 不平衡问题报告的必备三件套")
    print("    (1) 基线(空壳模型)成绩  (2) 运营阈值下的混淆矩阵  (3) AUC")
    print("    看到'准确率 98.5%'一行字的汇报，就要求这三样。")
