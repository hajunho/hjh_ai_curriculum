"""
让程序能直接吃下 LLM 输出的办法: 结构化输出 (JSON) + 校验 + 重试。
1) 要求模型从合同文本里把核心信息抽成 JSON
2) 模拟 LLM 的第一次回答复现了实战中经常遇到的失败 (夹带闲聊、类型错误)
3) 实现"模式校验 -> 把失败原因写进提示词再问一次 -> 成功"的循环
这套"请求-校验-重试"模式，换成真实 API 也可以原样复用。
"""

import json
import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # mock_llm 路径 (沿用惯例)

CONTRACT = (
    "服务合同。韩辉物产股份有限公司(以下简称'甲方')与青岚软件股份有限公司(以下简称'乙方')，"
    "就库存管理系统建设服务事项，订立合同如下。"
    "合同金额共计120,000,000韩元(不含增值税)。"
    "合同期限自2026年11月1日起至2027年4月30日止。"
    "乙方逾期交付时，每逾期1日按合同金额的0.1%支付违约金。"
)

# 程序期望的模式 (schema): 字段名 -> (类型, 说明)
SCHEMA = {
    "party_a": (str, "甲方公司名"),
    "party_b": (str, "乙方公司名"),
    "amount_krw": (int, "合同金额(韩元，仅数字)"),
    "start_date": (str, "开始日 YYYY-MM-DD"),
    "end_date": (str, "结束日 YYYY-MM-DD"),
    "penalty_rate_per_day": (float, "每逾期1日的违约金比例(%)"),
}


class MockExtractorLLM:
    """负责回应抽取请求的模拟 LLM。
    第1次: 闲聊 + markdown 代码围栏 + 金额是字符串 (实战常见失败)
    第2次: 收到带错误反馈的强化提示词后，返回正确的 JSON"""

    def complete(self, prompt: str) -> str:
        strict_retry = "错误" in prompt and "数字" in prompt
        if not strict_retry:
            return (
                "好的！我来帮您从合同里抽取信息。\n"
                "```json\n"
                "{\n"
                '  "party_a": "韩辉物产股份有限公司",\n'
                '  "party_b": "青岚软件股份有限公司",\n'
                '  "amount_krw": "1亿2千万韩元",\n'
                '  "start_date": "2026-11-01",\n'
                '  "end_date": "2027-04-30"\n'
                "}\n"
                "```\n"
                "希望对您有帮助！"
            )
        return (
            "{\n"
            '  "party_a": "韩辉物产股份有限公司",\n'
            '  "party_b": "青岚软件股份有限公司",\n'
            '  "amount_krw": 120000000,\n'
            '  "start_date": "2026-11-01",\n'
            '  "end_date": "2027-04-30",\n'
            '  "penalty_rate_per_day": 0.1\n'
            "}"
        )


def extract_json_block(text: str) -> str:
    """防御式地只截出回答里的 JSON 部分 (去掉闲聊和代码围栏)。"""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


def validate(data: dict) -> list[str]:
    """模式校验: 把缺失字段和类型错误一次性全收集起来返回。"""
    errors = []
    for field, (ftype, desc) in SCHEMA.items():
        if field not in data:
            errors.append(f"字段缺失: {field} ({desc})")
        elif not isinstance(data[field], ftype):
            errors.append(f"类型错误: {field} 必须是 {ftype.__name__} "
                          f"(当前值: {data[field]!r})")
    for d in ("start_date", "end_date"):
        if isinstance(data.get(d), str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data[d]):
            errors.append(f"格式错误: {d} 必须是 YYYY-MM-DD")
    return errors


def extract_with_retry(llm: MockExtractorLLM, max_attempts: int = 3) -> dict | None:
    """请求 -> 解析 -> 校验 -> (失败时) 带上错误原因再问一次的循环。"""
    base_prompt = (
        "请从下面的合同中抽取信息，只输出 JSON。\n"
        f"字段: {', '.join(f'{k}({v[1]})' for k, v in SCHEMA.items())}\n"
        "正文: " + CONTRACT
    )
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + feedback
        print(f"\n  --- 第 {attempt} 次尝试 ---")
        raw = llm.complete(prompt)
        preview = raw.replace("\n", " ")[:56]
        print(f"  回答(原文片段): {preview}...")

        try:
            data = json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            print(f"  [解析失败] 无法解析为 JSON: {e}")
            feedback = "\n[错误] 请只输出 JSON，不要写说明文字。"
            continue

        errors = validate(data)
        if not errors:
            print("  [校验通过] 与模式完全一致。")
            return data
        print(f"  [校验失败] 共 {len(errors)} 条:")
        for err in errors:
            print(f"    - {err}")
        # 把失败原因原样写进下一轮提示词 (诱导模型自我修正)
        feedback = ("\n[错误] 上一次回答的问题: " + " / ".join(errors) +
                    "\n金额只写数字，所有字段一个都不能少，只输出 JSON。")
    return None


def main() -> None:
    print("=" * 62)
    print("Level 03 | 结构化输出 — JSON 抽取·校验·重试")
    print("=" * 62)

    print("\n[1] 任务: 合同文本 -> 可以灌进系统的结构化数据")
    print(f"    原文({len(CONTRACT)}字): {CONTRACT[:40]}...")
    print("    期望的模式:")
    for field, (ftype, desc) in SCHEMA.items():
        print(f"      {field:<22}{ftype.__name__:<7}{desc}")

    print("\n[2] 运行请求-校验-重试循环")
    llm = MockExtractorLLM()
    result = extract_with_retry(llm)

    print("\n[3] 最终结果")
    if result is None:
        print("    3次尝试全部失败 -> 转入人工复核队列 (实务兜底)。")
    else:
        for k, v in result.items():
            print(f"    {k:<22}= {v!r}")
        amount = result["amount_krw"]
        penalty_per_day = amount * result["penalty_rate_per_day"] / 100
        print(f"\n    用法示例: 每逾期1日违约金 = {amount:,} x 0.1% = {penalty_per_day:,.0f} 韩元")
        print("    -> 类型有了保证，就能直接拿去计算、写数据库。")

    print("\n[4] 模式总结")
    print("    (1) 把格式钉死: 明确要求只输出 JSON (并把模式写进提示词)")
    print("    (2) 即便如此也可能跑偏，所以解析要防御式 (去围栏、去闲聊)")
    print("    (3) 把校验失败的原因写进提示词再问一次 (通常 1~2 次就好了)")
    print("    (4) 超过最大次数就交给人 (禁止悄无声息地失败)")

    # ------------------------------------------------------------------
    # [参考] 真实 Claude API 的话: 有结构化输出功能可以在服务端强制模式，
    # 于是重试循环基本就不需要了。
    # import anthropic
    # client = anthropic.Anthropic()
    # response = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user", "content": "请从合同中抽取: " + CONTRACT}],
    #     output_config={"format": {"type": "json_schema", "schema": {...}}},
    # )
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
