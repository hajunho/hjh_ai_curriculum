"""
交叉验证实验三连。
(1) 用 30 次实验确认单次划分的分数随划分手气晃动多少
(2) 用 5-fold 交叉验证做出'平均 ± 标准差'的报告
(3) 与 TimeSeriesSplit 对比, 看随机划分如何吹胀时间序列数据的分数。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold, TimeSeriesSplit)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def main() -> None:
    print("=" * 62)
    print(" 交叉验证: 测量并驯服性能数字里的'手气'")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

    # [1] 单次划分 30 回: 分数是随机变量 --------------------------------
    print("\n[1] 同一模型、同一数据 — 只换 30 次划分 seed 来测 AUC")
    scores = []
    for seed in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=seed, stratify=y)
        pipe.fit(X_tr, y_tr)
        scores.append(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
    scores = np.array(scores)
    print(f"    最小 {scores.min():.4f} / 最大 {scores.max():.4f} / "
          f"跨度 {scores.max()-scores.min():.4f} / 标准差 {scores.std():.4f}")
    # 简易直方图
    bins = np.linspace(scores.min(), scores.max() + 1e-9, 7)
    counts, _ = np.histogram(scores, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], counts):
        print(f"    {lo:.3f}~{hi:.3f} | {'#' * c}")
    print("    => '单一分数 0.87' 的背后, 藏着这么大的运气成分。")

    # [2] 5-fold 交叉验证 --------------------------------------------------
    print("\n[2] 5-fold StratifiedKFold 交叉验证 (整条流水线投入)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("    各折 AUC:", " ".join(f"{s:.4f}" for s in cv_scores))
    print(f"    汇报格式 => AUC {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # [3] 两个模型的公平比较 ----------------------------------------------
    print("\n[3] 模型比较: 逻辑回归 vs 随机森林 (同一 CV)")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"    逻辑回归 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    随机森林 : {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    diff = abs(cv_scores.mean() - rf_scores.mean())
    noise = max(cv_scores.std(), rf_scores.std())
    verdict = "很难视为有意义的差异 (平均差 < 波动)" if diff < noise \
        else "差异大于波动, 看起来有意义"
    print(f"    平均差 {diff:.4f} vs 波动 {noise:.4f} => {verdict}")

    # [4] 时间序列陷阱: 随机 KFold vs TimeSeriesSplit ---------------------
    print("\n[4] 时间序列数据: 用未来预测过去, 分数就会膨胀")
    sales = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    sales = sales.dropna(subset=["revenue"])
    sales = sales[sales["revenue"] > 0]
    daily = sales.groupby("day_index")["revenue"].sum().reset_index()
    # 滞后(lag)特征: 前一日·移动平均 -> 随机划分时验证信息渗入训练特征
    daily["lag1"] = daily["revenue"].shift(1)
    daily["ma7"] = daily["revenue"].shift(1).rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)
    Xs, ys = daily[["lag1", "ma7"]], daily["revenue"]

    model = Ridge(alpha=1.0)
    shuffled = cross_val_score(model, Xs, ys, scoring="r2",
                               cv=KFold(n_splits=5, shuffle=True, random_state=42))
    tssplit = TimeSeriesSplit(n_splits=5)
    ordered = cross_val_score(model, Xs, ys, scoring="r2", cv=tssplit)
    print(f"    随机 KFold(shuffle)      R2: {shuffled.mean():.4f} ± {shuffled.std():.4f}")
    print(f"    TimeSeriesSplit(保持顺序) R2: {ordered.mean():.4f} ± {ordered.std():.4f}")
    print("    (R2 < 0 = 还不如'押平均值'。这是'仅凭前一日销售额很难预测未来'")
    print("     的诚实成绩单, 而随机划分把这个事实藏了起来。)")
    print("    TimeSeriesSplit 的折结构 (训练永远早于验证):")
    for i, (tr, te) in enumerate(tssplit.split(Xs)):
        print(f"      fold{i+1}: 训练 day {tr.min()}~{tr.max()} -> 验证 day {te.min()}~{te.max()}")
    print("\n    教训: 性能测一次是'运气', 测多次才是'实力 ± 误差'。")
    print("          另外, 有时间轴的话, 考卷一定要从未来拿。")


if __name__ == "__main__":
    main()
