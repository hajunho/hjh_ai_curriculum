"""
把数据项目的五个阶段 (问题定义→数据→模型→部署→监控)
逐一通过关口 (gate) 检查清单的模拟。
以虚构的"订阅流失防御"项目为例，
观察某个阶段的产出物一旦留空，关口就会给出 STOP 判定。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def gate(stage_name: str, checklist: dict) -> bool:
    """检查清单里只要有一项的值为空，就无法通过关口。"""
    print(f"\n  [{stage_name}] 关口检查")
    ok = True
    for item, value in checklist.items():
        filled = value not in ("", None, [])
        mark = "OK " if filled else "缺失"
        shown = value if filled else "(空白)"
        print(f"    - {mark} | {item}: {shown}")
        if not filled:
            ok = False
    print(f"    => 判定: {'PASS - 进入下一阶段' if ok else 'STOP - 填齐之前禁止推进'}")
    return ok


def main() -> None:
    print("=" * 62)
    print(" 数据项目生命周期模拟: '订阅流失防御'项目")
    print("=" * 62)
    results = {}

    # ------------------------------------------------------------------
    print("\n[1] 问题定义 — 预测什么、为什么、成功标准是?")
    problem_spec = {
        "预测对象(标签定义)": "下个月未续费的客户 = 流失(1)",
        "预测结果的使用方式": "每周一向风险最高的 200 人提供回访/优惠券",
        "目标指标(数字)": "6 个月内月流失率 18% -> 15%",
        "基准线(现行做法)": "向全体客户中随机 100 人发放优惠券",
    }
    results["1.问题定义"] = gate("问题定义", problem_spec)

    # ------------------------------------------------------------------
    print("\n[2] 数据 — 有能回答这个问题的数据吗? (执行真实审计)")
    rows = hjh_data.churn_table(n=2000, seed=7)
    n_total = len(rows)
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    n_churn = sum(r["churned"] for r in rows)
    churn_rate = n_churn / n_total
    min_minority = 200  # 少数类(流失)最少样本的关口标准

    print(f"    行数: {n_total} / 含缺失的行: {n_missing}")
    print(f"    流失客户: {n_churn} 人 (流失率 {churn_rate:.1%})")
    data_audit = {
        "样本规模确认": f"{n_total} 行",
        "缺失检查": f"{n_missing} 行 (在允许范围内)",
        "少数类样本": f"{n_churn} 人" if n_churn >= min_minority else "",
        "标签定义一致性确认": "以扣费日志为准，已与客服团队达成一致",
    }
    results["2.数据"] = gate("数据", data_audit)

    # ------------------------------------------------------------------
    print("\n[3] 模型 — 先算出必须跨过的基准线 (baseline)")
    # 不用任何模型、只押'多数类(未流失)'的基准线准确率
    majority_acc = 1 - churn_rate
    print(f"    多数票基准线准确率: {majority_acc:.1%}  <- 光喊'没人会流失'就能拿到这个数")
    print("    => 之后要做的模型必须靠真正找出流失者的召回率/精确率")
    print("       来打败这条基准线，而不是靠'准确率' (level01、level07 继续讲)。")
    model_report = {
        "基准线性能记录": f"多数票准确率 {majority_acc:.1%}",
        "验证方法共识": "交叉验证 5-fold (level06 学习)",
        "模型性能报告": "(将在后续关卡完成)",  # 演示用，视为已填写
    }
    results["3.模型"] = gate("模型", model_report)

    # ------------------------------------------------------------------
    print("\n[4] 部署 — 业务方真的能用吗?")
    deploy_plan = {
        "结果交付渠道": "每周一把风险名单上传至 CRM 系统",
        "接收人与业务流程": "",  # 故意留空: 为了观察 STOP 判定
        "故障应对(模型失效时)": "",
    }
    results["4.部署"] = gate("部署", deploy_plan)

    # ------------------------------------------------------------------
    print("\n[5] 监控 — 性能还保持着吗?")
    monitor_plan = {
        "性能跟踪指标": "每周召回率/精确率、活动后的实际流失率",
        "数据分布偏移监测": "",  # 故意留空
        "重训标准": "召回率连续 2 周下降超过 5 个百分点即重训",
    }
    results["5.监控"] = gate("监控", monitor_plan)

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("[6] 最终汇总 — 这个项目现在该停在哪里")
    print("=" * 62)
    first_stop = None
    for stage, ok in results.items():
        print(f"    {stage:<10} : {'PASS' if ok else 'STOP'}")
        if not ok and first_stop is None:
            first_stop = stage
    if first_stop:
        print(f"\n    => 第一个 STOP 点: {first_stop}")
        print("       比起继续拉高模型性能，先把这道关口的空白填上才是当务之急。")
    print("\n    教训: 失败的项目不是倒在代码上，而是倒在'揣着空白往前冲'上。")


if __name__ == "__main__":
    main()
