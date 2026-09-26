"""
实战 2 — 客户流失预测的完结项目。
问题说明书 -> 迷你 EDA -> 特征 -> 交叉验证 -> 最终模型 -> 系数解释 ->
一直做到'风险客户 Top10 + 主要原因 + 推荐动作'这张表,
产出市场部拿到就能直接执行的交付物。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score

BASE = ["tenure_months", "monthly_fee", "usage_days_30d",
        "support_calls_30d", "plan_changes", "auto_pay"]
DERIVED = ["usage_per_tenure", "calls_plus_changes"]
FEAT_ZH = {
    "tenure_months": "入网月数", "monthly_fee": "月费",
    "usage_days_30d": "近30日使用天数", "support_calls_30d": "近30日来电",
    "plan_changes": "套餐变更", "auto_pay": "自动付款",
    "usage_per_tenure": "时长折算活跃度", "calls_plus_changes": "不满信号合计",
}
# 主要原因 -> 推荐动作 (模型之外由人设计的应对规则)
ACTION_MAP = {
    "usage_days_30d": "回访引流内容 + 7天免费体验券",
    "usage_per_tenure": "回访引流内容 + 7天免费体验券",
    "support_calls_30d": "客服优先接待, 解决不满根因",
    "calls_plus_changes": "客服优先接待, 解决不满根因",
    "plan_changes": "套餐定制推荐咨询",
    "monthly_fee": "套餐定制推荐咨询",
    "auto_pay": "转自动付款可享1个月8折",
    "tenure_months": "新手引导 + 首月权益告知",
}


def main() -> None:
    print("=" * 70)
    print(" 实战 2: 客户流失预测 — 把名单、理由、动作一次做完")
    print("=" * 70)

    # [1] 问题说明书 --------------------------------------------------------
    print("\n[1] 问题说明书 (比代码更先)")
    print("    预测对象: 本月是否退订 / 用途: 每周对高风险客户做 CRM 活动")
    print("    目标指标: 召回率 55%+ 时精确率 25%+ / 基准线: 随机群发(命中率=流失率约 15%)")

    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    # [2] 迷你 EDA ---------------------------------------------------------
    print(f"\n[2] 迷你 EDA: {len(df)}人, 流失率 {df['churned'].mean():.1%}")
    grp = df.groupby("churned")[BASE].mean()
    gap = ((grp.loc[1] - grp.loc[0]) / grp.loc[0]).sort_values(key=abs, ascending=False)
    print("    流失组与留存组差得最远的信号 (平均值差异比例):")
    for name, v in gap.head(3).items():
        print(f"      {FEAT_ZH[name]:<14} {v:+.0%}")

    # [3] 特征准备 --------------------------------------------------------
    print("\n[3] 特征: 原始 6 个 + 衍生 2 个")
    df["usage_per_tenure"] = df["usage_days_30d"] / (df["tenure_months"] + 1)
    df["calls_plus_changes"] = df["support_calls_30d"] + df["plan_changes"]
    features = BASE + DERIVED
    X, y = df[features], df["churned"]

    # [4] 交叉验证 + 最终模型 ----------------------------------------------
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(random_state=42, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\n[4] 5-fold 交叉验证 AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    idx_te = X_te.index
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rec, prec = recall_score(y_te, pred), precision_score(y_te, pred)
    print(f"    测试性能: AUC {roc_auc_score(y_te, proba):.3f} / "
          f"召回率 {rec:.1%} / 精确率 {prec:.1%}")
    goal = "达成" if (rec >= 0.55 and prec >= 0.25) else "未达 -> 重新检视阈值和特征"
    print(f"    目标指标(召回率 55%+、精确率 25%+) 判定: {goal}")
    print(f"    (相对随机群发基准线的命中率 {y_te.mean():.1%}, 精确率是它的 {prec/y_te.mean():.1f}倍)")

    # [5] 解释: 标准化系数 -------------------------------------------------
    print("\n[5] 模型解释 — 什么在推高风险 (标准化系数)")
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    for name, c in sorted(zip(features, coefs), key=lambda t: -abs(t[1])):
        arrow = "风险上升" if c > 0 else "风险下降"
        print(f"    {FEAT_ZH[name]:<14} {c:+.2f} ({arrow}) {'#' * int(abs(c) * 6)}")

    # [6] 交付物: 风险 Top10 + 原因 + 动作 -----------------------------------
    print("\n[6] 最终交付物 — 流失风险 Top10 名单 (取自测试客户)")
    # 单个客户的'主要原因' = 标准化特征值 x 系数中, 朝风险方向贡献最大的那个特征
    Z = scaler.transform(X_te)                      # 标准化后的特征值
    contrib = Z * coefs                             # 每位客户 x 每个特征的风险贡献
    top10 = np.argsort(proba)[::-1][:10]
    print(f"    {'客户ID':<9} {'流失概率':>7}  {'主要原因':<16} 推荐动作")
    print("    " + "-" * 66)
    for i in top10:
        cust_id = df.loc[idx_te[i], "customer_id"]
        main_feat = features[int(np.argmax(contrib[i]))]
        action = ACTION_MAP[main_feat]
        print(f"    {cust_id:<9} {proba[i]:>6.1%}  {FEAT_ZH[main_feat]:<16} {action}")
    print("\n    活动运营备忘:")
    print("      - 概率 80% 以上: 电话回访 / 50~80%: 优惠券+短信 (按概率区间调节力度)")
    print("      - 效果验证: 随机留出一部分风险客户当对照组, 比较流失率")
    print("\n    教训: 项目的完成不是 AUC, 而是'周一早上市场部拿过去")
    print("          就能照着执行的那张表'。")


if __name__ == "__main__":
    main()
