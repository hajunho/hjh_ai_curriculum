"""
提示词工程: 亲身体验同一个模型、不同指令书带来的质量差异。
- 把"差提示词"和"好提示词"(含角色·情境·格式·示例) 并排运行对比
- 用提示词要素检查器 (清单) 给自己的指令书打分
MockLLM 被设计成"角色/格式/示例写得越清楚，回答越结构化"，
以此复现真实 LLM 上观察到的质量差异。
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mock_llm import MockLLM

MEETING_NOTE = (
    "本周库存会议决定把物流仓库自动化投资推迟到明年第一季度。"
    "作为替代方案，从11月起招聘20名旺季临时人员。"
    "库存管理系统的报错件数比上月减少了40%。"
    "下次会议将重新确定各门店的安全库存标准。"
    "会议室空调的维修申请已转交总务部。"
)

COMPLAINT = "我用信用卡支付的，可是退款两周了还没退，客服电话也一直打不通。"

# 提示词四大要素: 角色(Role)、情境(Context)、格式(Format)、示例(Example)
CHECK_ITEMS = [
    ("角色", ["你是", "角色"], "指定模型要像哪种专家一样行动"),
    ("情境", ["情境", "目标读者", "背景"], "交代为什么做、给谁做的背景"),
    ("格式", ["格式", "要点", "JSON", "条列"], "明确输出的样子 (表格/列表/JSON...)"),
    ("示例", ["示例"], "给出期望的输入输出样本 (few-shot)"),
]


def audit_prompt(prompt: str) -> tuple[int, list[str]]:
    """给提示词打分: 四大要素各自有没有写进去。"""
    passed = []
    for name, keywords, _ in CHECK_ITEMS:
        if any(k in prompt for k in keywords):
            passed.append(name)
    return len(passed), passed


def show_case(title: str, prompt: str, llm: MockLLM) -> None:
    score, passed = audit_prompt(prompt)
    print(f"\n  ({title}) 包含要素 {score}/4: {', '.join(passed) if passed else '无'}")
    print("  --- 提示词 ---")
    for line in prompt.strip().splitlines():
        print(f"  > {line}")
    print("  --- MockLLM 回答 ---")
    for line in llm.complete(prompt).splitlines():
        print(f"  | {line}")


def main() -> None:
    llm = MockLLM()

    print("=" * 62)
    print("Level 02 | 提示词工程 — 怎么把指令书写好")
    print("=" * 62)

    # [1] 提示词四大要素
    print("\n[1] 好指令书的四大要素")
    for name, _, desc in CHECK_ITEMS:
        print(f"    {name}: {desc}")

    # [2] 案例 A — 会议纪要总结: 差提示词 vs 好提示词
    print("\n[2] 案例 A: 会议纪要总结")
    bad = "总结一下。\n正文: " + MEETING_NOTE
    good = (
        "你是一名负责整理会议纪要、供高管阅读的秘书。\n"
        "情境: 忙碌的高管要在30秒内读完。\n"
        "请把下面的会议纪要按要点格式总结，重点放在决议事项上。\n"
        "正文: " + MEETING_NOTE
    )
    show_case("差提示词", bad, llm)
    show_case("好提示词", good, llm)
    print("\n    -> 差提示词只会甩回来一句话，")
    print("       好提示词连'谁为什么要读'都讲清楚了，于是拿到结构化的回答。")

    # [3] 案例 B — 客诉分类: 示例 (few-shot) 的威力
    print("\n[3] 案例 B: 客诉分类 (few-shot)")
    bad2 = "这是什么客诉？分类一下。\n正文: " + COMPLAINT
    good2 = (
        "你是客服中心负责客诉分类的专员。\n"
        "请把下面的客诉分类。格式: '分类: <类别>' 一行 + 依据关键词一行。\n"
        "示例: '快递太慢了' -> 分类: 物流配送 / 依据关键词: 快递\n"
        "正文: " + COMPLAINT
    )
    show_case("差提示词", bad2, llm)
    show_case("好提示词", good2, llm)
    print("\n    -> 给了示例 (few-shot)，输出格式就被固定住，可以直接接到")
    print("       下一道工序 (自动处理)。格式忽高忽低，自动化就会断链。")

    # [4] 各要素的效果小结
    print("\n[4] 各要素效果小结")
    effects = [
        ("角色", "语气和专业程度定下来了 ('像秘书一样'、'像律师一样')"),
        ("情境", "有了取舍标准 (给高管看 vs 给执行层看)"),
        ("格式", "输出变得可预测，后续自动化更容易"),
        ("示例", "用几个样本传达那些说不清楚的风格"),
    ]
    for name, effect in effects:
        print(f"    {name}: {effect}")

    print("\n总结: 提示词就是'给新员工的工作指令书'。")
    print("      把角色、情境、格式、示例这4项填满，结果质量会阶梯式上升。")

    # ------------------------------------------------------------------
    # [参考] 真实 API 的话: 角色写进系统提示词，任务写进用户消息
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5",
    #     max_tokens=1024,
    #     system="你是一名负责整理会议纪要、供高管阅读的秘书。",
    #     messages=[{"role": "user", "content": "请把下面的会议纪要按要点格式总结..."}],
    # )
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
