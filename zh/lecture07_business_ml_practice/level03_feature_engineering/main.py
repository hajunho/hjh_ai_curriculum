"""
证明特征工程威力的对照实验。
在 fraud_table 上钉死同一个模型、同一份划分，
比较'仅原始特征' vs '原始+衍生特征'的性能 (AUC、PR-AUC)，
最后再演示数据泄漏特征制造出来的虚假性能。
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
from sklearn.metrics import roc_auc_score, average_precision_score

# '原始' = 支付授权报文里直接送达的字段 (金额、时刻、是否境外)
RAW = ["amount", "hour", "is_foreign"]
# '衍生' = 人用领域知识造出来附加的特征
# (交易历史聚合特征 tx_count_1h 留作'动手试试'的作业)
DERIVED = ["log_amount", "is_night", "foreign_night"]


def evaluate(X_tr, X_te, y_tr, y_te, label: str) -> tuple[float, float]:
    """钉死模型和预处理, 只换特征组来测性能 (对照实验)。"""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    pr_auc = average_precision_score(y_te, proba)
    print(f"    {label:<24} AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    return auc, pr_auc


def main() -> None:
    print("=" * 62)
    print(" 原始特征 vs 衍生特征 — 用同一个模型公平竞技")
    print("=" * 62)

    # [1] 原始数据 ----------------------------------------------------
    df = pd.DataFrame(hjh_data.fraud_table(n=5000, seed=11))
    print(f"\n[1] fraud_table {len(df)} 笔, 欺诈比例 {df['is_fraud'].mean():.2%}")
    print(f"    原始特征(支付瞬间即知的值): {RAW}")

    # [2] 领域知识 -> 衍生特征 ---------------------------------------
    print("\n[2] 把领域知识翻译成数字列 (3 个衍生特征)")
    df["log_amount"] = np.log1p(df["amount"])                    # 偏斜的金额 -> 倍率刻度
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(int)  # 凌晨标志
    df["foreign_night"] = df["is_foreign"] * df["is_night"]       # 境外 x 凌晨交互项
    recipes = {
        "log_amount": "对数变换 — 让'金额大 2 倍'对应相同的间隔",
        "is_night": "区间标志 — '凌晨交易可疑'这条业务常识",
        "foreign_night": "交互项 — 境外'同时'又是凌晨才危险",
    }
    for k, v in recipes.items():
        print(f"    - {k:<14}: {v}")

    # [3] 对照实验: 钉死划分和模型, 只换特征 -------------------------
    print("\n[3] 性能比较 (逻辑回归, 同一划分, 正类=欺诈)")
    y = df["is_fraud"]
    idx_tr, idx_te = train_test_split(df.index, test_size=0.3,
                                      random_state=42, stratify=y)
    _, pr_raw = evaluate(df.loc[idx_tr, RAW], df.loc[idx_te, RAW],
                         y.loc[idx_tr], y.loc[idx_te], "原始 3 个")
    _, pr_full = evaluate(df.loc[idx_tr, RAW + DERIVED], df.loc[idx_te, RAW + DERIVED],
                          y.loc[idx_tr], y.loc[idx_te], "原始 + 衍生共 6 个")
    print(f"    => PR-AUC {pr_raw:.4f} -> {pr_full:.4f} "
          f"({(pr_full - pr_raw) / pr_raw * 100:+.1f}%) — 模型一个字都没改")

    # [4] 哪些特征在干活 -------------------------------------------
    print("\n[4] 衍生特征组模型的标准化系数 (绝对值越大影响越大)")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    )
    cols = RAW + DERIVED
    model.fit(df.loc[idx_tr, cols], y.loc[idx_tr])
    coefs = model.named_steps["logisticregression"].coef_[0]
    for name, c in sorted(zip(cols, coefs), key=lambda t: -abs(t[1])):
        bar = "#" * int(abs(c) * 4)
        print(f"    {name:<14} {c:+.2f} {bar}")

    # [5] 泄漏演示: 把答案的影子塞进特征会怎样 -----------------------
    print("\n[5] 数据泄漏演示 — 名叫'调查结果得分'的假特征")
    rng = np.random.default_rng(0)
    # 只有在欺诈判定'之后'才可能知道的值: 答案 + 一点噪声 = 典型的泄漏特征
    df["inspection_score"] = df["is_fraud"] * 0.9 + rng.normal(0, 0.1, len(df))
    _, pr_leak = evaluate(df.loc[idx_tr, cols + ["inspection_score"]],
                          df.loc[idx_te, cols + ["inspection_score"]],
                          y.loc[idx_tr], y.loc[idx_te], "衍生 + 泄漏特征")
    print(f"    => PR-AUC {pr_leak:.4f}: 完美得不真实 = 不是喜讯, 是'bug 警报'。")
    print("       判别问题: \"预测时点知道这个值吗?\" — 不知道就立刻剔除。")
    print("\n    教训: 性能的钥匙不是换模型, 而是把领域知识翻译成特征。")


if __name__ == "__main__":
    main()
