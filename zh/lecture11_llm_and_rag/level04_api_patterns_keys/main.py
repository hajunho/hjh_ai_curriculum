"""
用一个假的 API 服务器，安全地练习 LLM API 调用的运维模式。
- 请求/响应的结构 (模型名、消息、token 用量)
- 用环境变量管理 API 密钥的方法 (禁止硬编码进代码)
- 实现带重试 + 指数退避 + 超时 + 费用追踪的客户端
假服务器用固定种子的随机数，复现真实运维里会碰到的
429 (速率限制)、500 (服务器错误) 和延迟。
"""

import os
import random
import time

# 价目表: 每百万 token 的费用 (教学用虚拟单价，美元)
PRICE_PER_MTOK = {"input": 3.0, "output": 15.0}
USD_KRW = 1400


# ---------------------------------------------------------------------------
# 假 API 服务器 — 请想象它在网络的另一头
# ---------------------------------------------------------------------------

def fake_llm_api(request: dict, rng: random.Random) -> dict:
    """模仿真实 LLM API 的行为 (成功/超限/服务器错误/延迟)。"""
    if request.get("api_key") != "sk-demo-1234":
        return {"status": 401, "error": "authentication_error: API 密钥无效"}

    roll = rng.random()
    latency = rng.uniform(0.3, 1.2)          # 平时的响应延迟 (秒，虚拟)
    if roll < 0.25:
        return {"status": 429, "error": "rate_limit_error: 超过每分钟请求上限",
                "retry_after": 1.0}
    if roll < 0.40:
        return {"status": 500, "error": "internal_server_error: 临时服务器错误"}
    if roll < 0.50:
        latency = 45.0                       # 偶尔来一个极端慢的响应

    prompt = request["messages"][0]["content"]
    in_tok = max(1, len(prompt) // 2)        # 中文这里按大约2字1个token估算
    out_text = f"(模拟回答) 已处理 '{prompt[:14]}...' 这个请求。"
    out_tok = max(1, len(out_text) // 2)
    return {"status": 200, "latency": latency,
            "content": out_text,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}


# ---------------------------------------------------------------------------
# 运维级客户端 — 重试·退避·超时·费用追踪
# ---------------------------------------------------------------------------

class RobustClient:
    def __init__(self, api_key: str, timeout: float = 10.0,
                 max_retries: int = 4, seed: int = 42, speedup: float = 100.0):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.rng = random.Random(seed)       # 固定种子: 保证演示可复现
        self.speedup = speedup               # 仅为演示: 把等待时间压缩到 1/100
        self.total = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}

    def _wait(self, seconds: float, reason: str) -> None:
        print(f"      … 等待 {seconds:.1f} 秒 ({reason})")
        time.sleep(seconds / self.speedup)   # 真实代码里就直接 sleep(seconds)

    def chat(self, prompt: str) -> str | None:
        request = {"model": "mock-llm-v1", "api_key": self.api_key,
                   "messages": [{"role": "user", "content": prompt}]}
        for attempt in range(self.max_retries + 1):
            resp = fake_llm_api(request, self.rng)
            status = resp["status"]

            if status == 200 and resp["latency"] > self.timeout:
                status = 408                 # 超时: 来得太晚的成功也算失败
                print(f"    第 {attempt + 1} 次: 响应 {resp['latency']:.0f} 秒 "
                      f"> 超时 {self.timeout:.0f} 秒 -> 断开重试")
            elif status != 200:
                print(f"    第 {attempt + 1} 次: HTTP {status} — {resp['error']}")

            if status == 200:
                self.total["calls"] += 1
                self.total["input_tokens"] += resp["usage"]["input_tokens"]
                self.total["output_tokens"] += resp["usage"]["output_tokens"]
                print(f"    第 {attempt + 1} 次: 成功 ({resp['latency']:.1f} 秒, "
                      f"输入 {resp['usage']['input_tokens']}tok / "
                      f"输出 {resp['usage']['output_tokens']}tok)")
                return resp["content"]
            if status == 401:
                print("    -> 密钥错误重试也没用: 立刻中断 (这是配置问题)")
                return None

            if attempt < self.max_retries:
                self.total["retries"] += 1
                # 指数退避: 1、2、4、8秒... + 随机抖动 (把并发重试打散)
                backoff = min(2 ** attempt, 30) + self.rng.uniform(0, 0.5)
                if status == 429 and "retry_after" in resp:
                    backoff = max(backoff, resp["retry_after"])
                self._wait(backoff, "指数退避")
        print("    -> 超过重试上限: 按失败处理 (入队列/通知人工)")
        return None

    def cost_report(self) -> str:
        cin = self.total["input_tokens"] / 1e6 * PRICE_PER_MTOK["input"]
        cout = self.total["output_tokens"] / 1e6 * PRICE_PER_MTOK["output"]
        return (f"成功调用 {self.total['calls']} 次，重试 {self.total['retries']} 次 | "
                f"输入 {self.total['input_tokens']:,}tok + 输出 {self.total['output_tokens']:,}tok"
                f" = ${cin + cout:.6f} (约 {(cin + cout) * USD_KRW:.2f} 韩元)")


def main() -> None:
    print("=" * 62)
    print("Level 04 | LLM API 调用模式与密钥管理")
    print("=" * 62)

    # [1] API 密钥放环境变量 — 绝不留在代码和仓库里
    print("\n[1] API 密钥管理: 从环境变量读取")
    key = os.environ.get("MOCK_LLM_API_KEY", "sk-demo-1234")
    masked = key[:7] + "*" * (len(key) - 7)
    src = "环境变量" if "MOCK_LLM_API_KEY" in os.environ else "默认值(演示用)"
    print(f"    MOCK_LLM_API_KEY -> {masked} ({src})")
    print("    实务: 像 export ANTHROPIC_API_KEY=... 这样存进 shell / 密钥管理器，")
    print("          代码和 git 仓库里连一个字符都不能出现。")

    # [2] 观察请求/响应的结构
    print("\n[2] 请求/响应结构 (调用一次来观察)")
    rng = random.Random(7)
    demo_req = {"model": "mock-llm-v1", "api_key": key,
                "messages": [{"role": "user", "content": "帮我总结一下远程办公制度"}]}
    demo_resp = fake_llm_api(demo_req, rng)
    while demo_resp["status"] != 200:        # 仅为观察: 一直调到出现成功响应
        demo_resp = fake_llm_api(demo_req, rng)
    print(f"    请求  : model={demo_req['model']!r}, messages=[{{role, content}}]")
    print(f"    响应  : {demo_resp}")
    print("    -> usage 里的 token 数就是费用。每次收到响应都要记录下来。")

    # [3] 错误的密钥: 区分出"重试也没意义"的错误
    print("\n[3] 用错误的密钥调用 (重试无意义的 4xx 错误)")
    bad = RobustClient(api_key="sk-wrong-key", seed=1)
    bad.chat("这个请求失败才是正常的")

    # [4] 用带重试·退避·超时的客户端处理 5 条
    print("\n[4] 用运维级客户端调用 5 次 (混有 429/500/延迟的环境)")
    client = RobustClient(api_key=key, timeout=10.0, seed=7)
    prompts = ["总结休假制度", "报销期限通知文案初稿", "客诉分类: 快递延迟",
               "会议纪要3行总结", "保修期说明文案"]
    for i, p in enumerate(prompts, 1):
        print(f"  ({i}) 请求: {p}")
        answer = client.chat(p)
        if answer:
            print(f"      回答: {answer}")

    # [5] 费用报告
    print("\n[5] 费用追踪报告")
    print(f"    {client.cost_report()}")
    print("    -> 月度预算 = 预计调用数 x 平均 token x 单价。请用看板常态监控。")

    print("\n总结: 只做成功路径叫演示，把失败路径 (429/500/超时/密钥错误) 也做出来")
    print("      才叫运维。退避、上限、费用追踪是 LLM 调用的基本功。")

    # ------------------------------------------------------------------
    # [参考] 真实 Claude API 的话 (SDK 会替你做重试和超时)
    # import anthropic
    # client = anthropic.Anthropic(          # 使用 ANTHROPIC_API_KEY 环境变量
    #     timeout=20.0, max_retries=3)
    # resp = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user", "content": "总结休假制度"}])
    # print(resp.usage.input_tokens, resp.usage.output_tokens)  # 确认计费
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
