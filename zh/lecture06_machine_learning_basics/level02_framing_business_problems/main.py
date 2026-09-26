"""
level02 — 业务问题 -> ML 问题说明书检查清单引擎

把"想降低流失"这类愿望写成五要素说明书(对象/单位/时点/指标/行动)，
本工具就自动检查空白项、泄漏风险与 ML 适配度。
这是唯一不建模的一关，却是实战中最常用的技术。
"""

from dataclasses import dataclass, field


@dataclass
class Feature:
    """一条预测材料(特征)。available_at: 能拿到这个值的时点。"""
    name: str
    available_at: str  # "预测时点之前" 或 "结果确定之后"


@dataclass
class MLSpec:
    """ML 问题说明书 — 写在会议室白板上的五要素 + 基线。"""
    business_goal: str = ""    # 业务目标 (愿望)
    target: str = ""           # 1. 预测对象: 可度量的定义
    unit: str = ""             # 2. 预测单位: 一行是什么
    timing: str = ""           # 3. 预测时点
    metric: str = ""           # 4. 评估指标
    action: str = ""           # 5. 预测后的行动
    baseline: str = ""         # 不用模型、按现有做法的成绩
    features: list[Feature] = field(default_factory=list)


def check_completeness(spec: MLSpec) -> list[str]:
    """检查五要素 + 基线是否填齐。"""
    problems = []
    checks = [
        (spec.target, "预测对象(target)为空。不要写'流失'，要写成'未来 30 天内退订(1/0)'这样可度量的定义。"),
        (spec.unit, "预测单位(unit)为空。先定清楚一行是客户、账户还是订单。"),
        (spec.timing, "预测时点(timing)为空。定清楚什么时候按下预测按钮。"),
        (spec.metric, "评估指标(metric)为空。请选一个和业务盈亏挂钩的指标。"),
        (spec.action, "行动(action)为空。没有行动的预测就是摆设。"),
        (spec.baseline, "基线(baseline)为空。先量一量不用模型、按现有做法的成绩。"),
    ]
    for value, msg in checks:
        if not value.strip():
            problems.append(msg)
    return problems


def check_leakage(spec: MLSpec) -> list[str]:
    """泄漏检查: '按下预测按钮的那一刻，这个值能拿到吗?'"""
    return [
        f"疑似泄漏: '{f.name}' 是在{f.available_at}才产生的值。"
        f"在预测时点({spec.timing})它并不存在，必须从材料中剔除。"
        for f in spec.features if f.available_at != "预测时点之前"
    ]


def validate(title: str, spec: MLSpec) -> None:
    """检查一份说明书并输出结果。"""
    print(f"  ■ 说明书: {title}")
    print(f"    目标: {spec.business_goal}")
    issues = check_completeness(spec) + check_leakage(spec)
    if not issues:
        print(f"    [通过] 五要素齐备 — 对象: {spec.target} / 单位: {spec.unit}")
        print(f"           时点: {spec.timing} / 指标: {spec.metric}")
        print(f"           行动: {spec.action}")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"    [批注 {i}] {msg}")
    print(f"    得分: {6 + len(spec.features) - len(issues)} / {6 + len(spec.features)}\n")


def judge_ml_fitness() -> None:
    """[4] 能解的问题 / 解不了的问题自动分类。
    标准: 反复发生、足够的标注数据、可能存在规律、允许误差。"""
    candidates = [
        ("各门店明日三明治需求预测", dict(repeats=True, labels=3000, pattern=True, error_ok=True)),
        ("增值税 10% 自动计算", dict(repeats=True, labels=100000, pattern=True, error_ok=False)),
        ("本公司 M&A 成败预测", dict(repeats=False, labels=12, pattern=True, error_ok=True)),
        ("银行卡异常交易实时检测", dict(repeats=True, labels=50000, pattern=True, error_ok=True)),
    ]
    for name, c in candidates:
        reasons = []
        if not c["repeats"]:
            reasons.append("不是反复发生的事 (案例攒不起来)")
        if c["labels"] < 500:
            reasons.append(f"正确答案的数据只有 {c['labels']} 条 (达不到可学习的量)")
        if not c["error_ok"]:
            reasons.append("误差容忍度为 0 -> 已有明确规则就用规则 (level00)")
        verdict = "值得用 ML 解" if not reasons else "不适合 ML"
        print(f"    {name:28s} -> {verdict}")
        for r in reasons:
            print(f"        理由: {r}")


if __name__ == "__main__":
    # [1] 烂说明书: 只有愿望，全是空白 --------------------------------
    print("[1] 烂说明书 — 把'想降低流失'原样提交的情况")
    validate("愿望原样", MLSpec(business_goal="想降低流失"))

    # [2] 好说明书: 五要素填齐的翻译完成版 -----------------------------
    print("[2] 好说明书 — 把同一个愿望翻译成 ML 问题的情况")
    good = MLSpec(
        business_goal="想降低订阅流失",
        target="以本月底为基准，未来 30 天内退订 (1/0)",
        unit="1 位活跃订阅客户 (每月 1 日快照)",
        timing="每月 1 日上午，只用截至上月末已确定的数据",
        metric="风险前 10% 名单的精确率/召回率 (不是准确率，参见 level06)",
        action="对风险前 10% 的客户，客服团队 1 周内进行挽留回访 + 提供优惠",
        baseline="现行做法是'给 3 个月未登录者群发短信' — 响应率 2%",
        features=[
            Feature("最近 30 天使用天数", "预测时点之前"),
            Feature("最近 30 天客服来电数", "预测时点之前"),
            Feature("入网月数", "预测时点之前"),
        ],
    )
    validate("流失预测 v1", good)

    # [3] 泄漏陷阱: 结果确定之后才产生的列混进了材料 -----------
    print("[3] 泄漏检查 — 混入了偷看未来的材料的说明书")
    leaky = MLSpec(
        business_goal="想降低订阅流失",
        target=good.target, unit=good.unit, timing=good.timing,
        metric=good.metric, action=good.action, baseline=good.baseline,
        features=[
            Feature("最近 30 天使用天数", "预测时点之前"),
            Feature("退订违约金账单金额", "结果确定之后"),   # <- 退订之后才产生的值!
            Feature("退订原因问卷回答", "结果确定之后"),
        ],
    )
    validate("流失预测 v2 (陷阱)", leaky)
    print("    -> 泄漏列只会让考试成绩完美，实战中根本用不上。\n")

    # [4] 能解的问题 / 解不了的问题分类 -----------------------------------------
    print("[4] 4 个候选问题的 ML 适配度判定")
    judge_ml_fitness()
    print()
    print("[5] 总结: 翻译顺序 = 愿望 -> (对象/单位/时点/指标/行动) -> 泄漏检查 -> 基线")
    print("    通过这份清单之后，建模(level03~)才值得开始。")
