"""
把混淆矩阵换算成韩元金额的计算器。
给流失预测模型 (逻辑回归) 的 TP/FP/FN/TN 贴上业务单价，
计算'这个模型值多少钱'、'召回率 1 个百分点值多少韩元'。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# ---- 业务单价 (假设是从财务/市场部门要来的数字) ----
VALUE_V = 179_000   # 客户留存价值: 留住即获得的营收 (韩元/人)
COST_C = 12_000     # 干预成本: 优惠券 + 回访成本 (韩元/人)
SUCCESS_S = 0.30    # 干预成功率: 收到优惠券的准流失客户留下来的概率

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def profit_of(tp: int, fp: int) -> float:
    """混淆矩阵 -> 期望收益(韩元)。TP 救回一部分人，FP 纯粹花钱。"""
    return tp * (VALUE_V * SUCCESS_S - COST_C) - fp * COST_C


def main() -> None:
    print("=" * 62)
    print(" 混淆矩阵 -> 韩元翻译器: 流失模型值多少钱")
    print("=" * 62)

    # [1] 数据准备与模型训练 ------------------------------------
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))
    X, y = df[FEATURES], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    # class_weight="balanced": 加权避免漏掉流失(少数类) (level07 详细讲)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(random_state=42, class_weight="balanced"))
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"\n[1] 逻辑回归训练完成 (训练 {len(X_tr)} 人 / 测试 {len(X_te)} 人, "
          f"测试集流失率 {y_te.mean():.1%})")

    # [2] 混淆矩阵 ----------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    rec = recall_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    print("\n[2] 测试集混淆矩阵 (正类 = 流失)")
    print(f"    TP(命中流失)={tp:4d}  FN(漏掉流失)={fn:4d}")
    print(f"    FP(误报)    ={fp:4d}  TN(正常放行)={tn:4d}")
    print(f"    召回率 {rec:.1%} / 精确率 {prec:.1%}")

    # [3] 给每个格子贴单价 -----------------------------------------
    print("\n[3] 用业务单价换算")
    print(f"    单价: 客户价值 V={VALUE_V:,}韩元, 干预成本 C={COST_C:,}韩元, 成功率 s={SUCCESS_S:.0%}")
    unit_tp = VALUE_V * SUCCESS_S - COST_C
    print(f"    1 个 TP 的价值 = V*s - C = {unit_tp:+,.0f}韩元")
    print(f"    1 个 FP 的价值 = -C      = {-COST_C:+,}韩元")
    print(f"    1 个 FN = 支出 0 韩元, 但机会损失 V*s = {VALUE_V*SUCCESS_S:,.0f}韩元")
    model_profit = profit_of(tp, fp)
    print(f"    => 模型期望收益(按测试 {len(X_te)} 人计): {model_profit:+,.0f}韩元")

    # [4] 对照组: 什么都不做 vs 全员发券 ------------------------
    print("\n[4] 策略比较 (基于同一批测试客户)")
    n_pos = int(y_te.sum())
    do_nothing = profit_of(0, 0)
    give_all = profit_of(n_pos, len(y_te) - n_pos)  # 全员干预: 流失者全是 TP, 其余全是 FP
    print(f"    A. 什么都不做            : {do_nothing:+13,.0f}韩元")
    print(f"    B. 给全员发优惠券        : {give_all:+13,.0f}韩元")
    print(f"    C. 只发给模型点名的人    : {model_profit:+13,.0f}韩元")
    best = max([("A", do_nothing), ("B", give_all), ("C", model_profit)], key=lambda t: t[1])
    print(f"    => 最佳策略: {best[0]} — 模型的价值在于'挑对花钱对象的能力'。")

    # [5] 召回率 1 个百分点的价值 ------------------------------------------
    print("\n[5] '召回率 1 个百分点'值多少韩元")
    n_customers = 100_000            # 扩展到真实服务规模
    churn_rate = float(y.mean())     # 从数据估计的流失率
    value_1pp = n_customers * churn_rate * 0.01 * unit_tp
    print(f"    公式: N * 流失率 * 0.01 * (V*s - C)")
    print(f"        = {n_customers:,}人 * {churn_rate:.1%} * 1%p * {unit_tp:,.0f}韩元")
    print(f"        = 约 {value_1pp:,.0f}韩元 (若按月做活动, 一年约 {value_1pp*12:,.0f}韩元)")
    print("\n    汇报句式示例:")
    print(f'    "把召回率从 {rec:.0%} 提到 {rec+0.05:.0%}, 提升 5 个百分点,')
    print(f'     每年可额外获得约 {value_1pp*5*12/1e8:.1f}亿韩元的流失防御收益。"')
    print("\n    教训: ML 指标一旦贴上单价, 就变成了预算语言。")


if __name__ == "__main__":
    main()
