"""
用模拟 LLM 体验公司里使用 LLM 的三个代表性场景。
1) 撰写邮件初稿  2) 总结会议纪要  3) 客户投诉(咨询)自动分类
最后打印引入 LLM 时必须逐项检查的风险清单。
不需要 API 密钥，靠规则式 MockLLM 运行；真实 API 调用示例
以注释形式提供。
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mock_llm import MockLLM

# 办公应用全景图: 各类型的代表任务与风险等级
USE_CASE_MAP = [
    ("总结", "会议纪要·报告·合同摘要", "低 (可对照原文)"),
    ("初稿", "邮件·通知·方案的第一版", "低 (由人做最终修改)"),
    ("分类", "客诉路由·邮件打标签", "中 (需监控误分类)"),
    ("抽取", "从文档里提取日期·金额·条款", "中 (必须有校验逻辑)"),
    ("翻译", "海外客户邮件·产品手册", "中 (合同文件需专家复核)"),
    ("代码", "生成 Excel 公式·SQL·脚本", "中 (执行前先审查)"),
    ("客服", "面向客户的问答机器人", "高 (幻觉·责任问题 -> 需要 RAG)"),
]

RISK_CHECKLIST = [
    "这项业务一旦答错，损失有多大？(金额·法律责任)",
    "流程设计里有没有'人做最终审核'这一步？",
    "公司机密、客户个人信息可以发给外部 API 吗？(安全合规)",
    "答案的依据可以追溯吗？(用 RAG 的话要引用出处)",
    "模型出错时有没有回退流程 (回滚·更正公告)？",
    "成本: 调用量 × token 单价，真的比人工便宜吗？",
]

MEETING_NOTE = (
    "三季度销售额同比增长12%，但物流成本上升导致营业利润小幅下降。"
    "市场部申请把新的销售活动预算增加50万元。"
    "开发部承诺在下个月之前完成订单系统的升级。"
    "下次会议定在10月15日上午10点。"
    "午餐菜单的话题以后再议。"
)

COMPLAINTS = [
    "快递一个星期了还没到货，运单号也查不到。",
    "申请了退款却一直取消不了，还被重复扣款了。",
    "产品发出奇怪的噪音，从昨天开始就不能用了。",
    "客服等待了40分钟才接通，态度还特别冷淡。",
]


def main() -> None:
    llm = MockLLM()

    print("=" * 62)
    print("Level 01 | LLM 办公应用全景图 — 三种任务演示")
    print("=" * 62)

    # [1] 办公应用全景图
    print("\n[1] LLM 办公应用全景图 (类型 | 代表任务 | 风险)")
    for kind, desc, risk in USE_CASE_MAP:
        print(f"    {kind:<4}| {desc:<24}| {risk}")

    # [2] 任务演示 1 — 邮件初稿
    print("\n[2] 演示 1: 撰写邮件初稿")
    prompt = (
        "你是一名秘书。请根据下面的信息撰写一封得体的工作邮件初稿。格式: 问候-事由-收尾\n"
        "收件人: 销售部王经理\n"
        "目的: 请求共享三季度销售业绩资料\n"
        "期限: 本周五"
    )
    print("    --- 指令(提示词)要点: 给出收件人/目的/期限，请求初稿 ---")
    for line in llm.complete(prompt).splitlines():
        print(f"    | {line}")
    print("    -> 初稿 30 秒完成，人只负责'审核和签名'。这就是初稿 (draft) 用法。")

    # [3] 任务演示 2 — 会议纪要总结
    print("\n[3] 演示 2: 会议纪要总结 (5句 -> 核心3行)")
    summary = llm.complete("请把下面的会议纪要总结成要点格式。\n正文: " + MEETING_NOTE)
    for line in summary.splitlines():
        print(f"    {line}")
    print("    -> 像午餐菜单这类不重要的句子是否被剔除，需要人来确认。")

    # [4] 任务演示 3 — 客诉自动分类
    print("\n[4] 演示 3: 客诉自动分类 (路由到负责部门)")
    for text in COMPLAINTS:
        result = llm.complete("请把下面的客户投诉分类。格式: 分类/依据关键词\n正文: " + text)
        cat = result.splitlines()[0].replace("分类: ", "")
        basis = result.splitlines()[1].replace("依据关键词: ", "")
        print(f"    \"{text[:16]}...\"")
        print(f"      -> 分类: {cat:<8} (依据: {basis})")
    print("    -> 每天 500 条客诉，在人读之前就先按部门分好类。")

    # [5] 引入风险清单
    print("\n[5] 引入前的风险清单 — 只要有一项是'否'，就回炉重新设计")
    for i, item in enumerate(RISK_CHECKLIST, 1):
        print(f"    {i}. {item}")

    print("\n总结: 从'初稿生成器 + 分类器'开始引入 LLM 是最稳妥的。")
    print("      一定要把'由人负最终责任的审核环节'写进流程设计。")

    # ------------------------------------------------------------------
    # [参考] 有真实 API 密钥时这样写 (Claude API 示例)
    # import anthropic
    # client = anthropic.Anthropic()          # ANTHROPIC_API_KEY 环境变量
    # response = client.messages.create(
    #     model="claude-opus-5",
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # print(response.content[0].text)
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
