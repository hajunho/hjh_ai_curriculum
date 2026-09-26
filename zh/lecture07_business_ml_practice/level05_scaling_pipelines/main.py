"""
缩放与流水线实验。
用单位五花八门的 churn_table 特征做 KNN 流失预测,
通过 (1) 不缩放 vs 缩放, (2) 预处理放在交叉验证之外 vs 放进流水线里
两组比较, 确认为什么'预处理全部进流水线'。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def report(name: str, y_true, proba) -> None:
    """输出 AUC 和'风险前 100 名中实际流失人数'(营销活动视角的指标)。"""
    auc = roc_auc_score(y_true, proba)
    top100 = np.asarray(y_true)[np.argsort(proba)[::-1][:100]].sum()
    print(f"    {name:<26} AUC {auc:.3f} / 风险前 100 名中实际流失 {int(top100)} 人")


def main() -> None:
    print("=" * 62)
    print(" 缩放与流水线: 统一单位 + 根除泄漏")
    print("=" * 62)

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)

    # [1] 确认特征单位 --------------------------------------------------
    print("\n[1] 各特征的尺度 (是在用同一把尺子量吗?)")
    for c in FEATURES:
        print(f"    {c:<18} 范围 {X[c].min():>7,.0f} ~ {X[c].max():>7,.0f}")
    print("    => monthly_fee 比其他特征大几百~几千倍: 它会垄断距离计算。")

    # [2] 不缩放直接 KNN ----------------------------------------------
    print("\n[2] 不缩放的 KNN (k=15)")
    knn_raw = KNeighborsClassifier(n_neighbors=15)
    knn_raw.fit(X_tr, y_tr)
    report("KNN (无缩放)", y_te, knn_raw.predict_proba(X_te)[:, 1])
    base = int(y_te.sum() / len(y_te) * 100)
    print(f"    (参考: 随机抽 100 人平均也有 {base} 人是流失者)")
    print("    (现在月费差几百韩元被看得比使用天数差 30 天还重)")

    # [3] 流水线: StandardScaler + KNN --------------------------------
    print("\n[3] 流水线 = 清洗(缩放) -> 组装(模型) 的传送带")
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    pipe.fit(X_tr, y_tr)          # 缩放器的 fit 自动只用训练数据
    report("KNN + StandardScaler", y_te, pipe.predict_proba(X_te)[:, 1])
    print("    => 同一模型·同一数据, 只是统一了单位, 性能就变了。")

    # [4] 错误顺序 vs 正确顺序 (在交叉验证里) -----------------------
    print("\n[4] 预处理放在哪里: 交叉验证 5-fold, 指标=AUC")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 错误做法: '先'对全量数据缩放 -> 折外信息渗入
    X_scaled_all = pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES)
    bad = cross_val_score(KNeighborsClassifier(n_neighbors=15),
                          X_scaled_all, y, cv=cv, scoring="roc_auc")

    # 正确做法: 塞整条流水线 -> 每一折只用训练部分重新 fit
    good = cross_val_score(make_pipeline(StandardScaler(),
                                         KNeighborsClassifier(n_neighbors=15)),
                           X, y, cv=cv, scoring="roc_auc")
    print(f"    错误顺序(全量缩放后再 CV): {bad.mean():.4f} ± {bad.std():.4f}")
    print(f"    正确顺序(整条流水线进 CV) : {good.mean():.4f} ± {good.std():.4f}")
    print("    => 缩放本身差距看着不大, 但换成目标编码·缺失填补·特征选择这类")
    print("       重度使用答案/分布的预处理, 分数可能被大幅吹胀。")
    print("       规则只有一条: '预处理全部进流水线'。")

    # [5] 部署视角: 一个对象搞定预测 -------------------------------------
    print("\n[5] 部署模拟: 一个流水线对象 = 预处理 + 模型")
    new_customers = pd.DataFrame([
        {"tenure_months": 2, "monthly_fee": 29900, "usage_days_30d": 3,
         "support_calls_30d": 4, "plan_changes": 2, "auto_pay": 0},
        {"tenure_months": 36, "monthly_fee": 9900, "usage_days_30d": 28,
         "support_calls_30d": 0, "plan_changes": 0, "auto_pay": 1},
        {"tenure_months": 12, "monthly_fee": 14900, "usage_days_30d": 15,
         "support_calls_30d": 1, "plan_changes": 1, "auto_pay": 1},
    ])
    probs = pipe.predict_proba(new_customers)[:, 1]
    for i, p in enumerate(probs):
        print(f"    新客户 {i+1}: 流失概率 {p:.1%}")
    print("    => 部署代码就一行 pipe.predict_proba(新数据)。漏做预处理的事故不可能发生。")
    print("\n    教训: 流水线不是便利功能, 而是防泄漏·防不一致的安全装置。")


if __name__ == "__main__":
    main()
