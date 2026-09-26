"""
Practice the operational patterns of LLM API calls safely against a fake API server.
- The request/response structure (model name, messages, token usage)
- Managing the API key via environment variables (never hardcode it)
- Building a client with retry + exponential backoff + timeout + cost tracking
The fake server reproduces what you hit in real operations — 429 (rate
limit), 500 (server error), and latency — with seed-fixed randomness.
"""

import os
import random
import time

# Price sheet: cost per million tokens (fictional teaching rates, USD)
PRICE_PER_MTOK = {"input": 3.0, "output": 15.0}
USD_KRW = 1400


# ---------------------------------------------------------------------------
# Fake API server — imagine it sitting across the network
# ---------------------------------------------------------------------------

def fake_llm_api(request: dict, rng: random.Random) -> dict:
    """Mimics a real LLM API's behavior (success / rate limit / server error / latency)."""
    if request.get("api_key") != "sk-demo-1234":
        return {"status": 401, "error": "authentication_error: invalid API key"}

    roll = rng.random()
    latency = rng.uniform(0.3, 1.2)          # normal response latency (seconds, fictional)
    if roll < 0.25:
        return {"status": 429, "error": "rate_limit_error: requests-per-minute limit exceeded",
                "retry_after": 1.0}
    if roll < 0.40:
        return {"status": 500, "error": "internal_server_error: transient server error"}
    if roll < 0.50:
        latency = 45.0                       # occasionally an extremely slow response

    prompt = request["messages"][0]["content"]
    in_tok = max(1, len(prompt) // 4)        # English: roughly 4 characters per token
    out_text = f"(mock response) Processed your request '{prompt[:14]}...'."
    out_tok = max(1, len(out_text) // 4)
    return {"status": 200, "latency": latency,
            "content": out_text,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}


# ---------------------------------------------------------------------------
# Production-grade client — retry, backoff, timeout, cost tracking
# ---------------------------------------------------------------------------

class RobustClient:
    def __init__(self, api_key: str, timeout: float = 10.0,
                 max_retries: int = 4, seed: int = 42, speedup: float = 100.0):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.rng = random.Random(seed)       # fixed seed: demo reproducibility
        self.speedup = speedup               # demo only: waits shrunk to 1/100
        self.total = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}

    def _wait(self, seconds: float, reason: str) -> None:
        print(f"      … waiting {seconds:.1f}s ({reason})")
        time.sleep(seconds / self.speedup)   # in real code: sleep(seconds) as-is

    def chat(self, prompt: str) -> str | None:
        request = {"model": "mock-llm-v1", "api_key": self.api_key,
                   "messages": [{"role": "user", "content": prompt}]}
        for attempt in range(self.max_retries + 1):
            resp = fake_llm_api(request, self.rng)
            status = resp["status"]

            if status == 200 and resp["latency"] > self.timeout:
                status = 408                 # timeout: a late success counts as failure
                print(f"    attempt {attempt + 1}: response {resp['latency']:.0f}s "
                      f"> timeout {self.timeout:.0f}s -> cut off and retry")
            elif status != 200:
                print(f"    attempt {attempt + 1}: HTTP {status} — {resp['error']}")

            if status == 200:
                self.total["calls"] += 1
                self.total["input_tokens"] += resp["usage"]["input_tokens"]
                self.total["output_tokens"] += resp["usage"]["output_tokens"]
                print(f"    attempt {attempt + 1}: success ({resp['latency']:.1f}s, "
                      f"input {resp['usage']['input_tokens']}tok / "
                      f"output {resp['usage']['output_tokens']}tok)")
                return resp["content"]
            if status == 401:
                print("    -> retrying a key error is pointless: stop now (a config problem)")
                return None

            if attempt < self.max_retries:
                self.total["retries"] += 1
                # Exponential backoff: 1, 2, 4, 8s... + random jitter (spreads out retries)
                backoff = min(2 ** attempt, 30) + self.rng.uniform(0, 0.5)
                if status == 429 and "retry_after" in resp:
                    backoff = max(backoff, resp["retry_after"])
                self._wait(backoff, "exponential backoff")
        print("    -> retry limit exceeded: mark as failed (queue it / alert a human)")
        return None

    def cost_report(self) -> str:
        cin = self.total["input_tokens"] / 1e6 * PRICE_PER_MTOK["input"]
        cout = self.total["output_tokens"] / 1e6 * PRICE_PER_MTOK["output"]
        return (f"{self.total['calls']} successful calls, {self.total['retries']} retries | "
                f"input {self.total['input_tokens']:,}tok + output {self.total['output_tokens']:,}tok"
                f" = ${cin + cout:.6f} (about KRW {(cin + cout) * USD_KRW:.2f})")


def main() -> None:
    print("=" * 62)
    print("Level 04 | API calling patterns and key management")
    print("=" * 62)

    # [1] API keys live in environment variables — never in code or the repo
    print("\n[1] API key management: read from an environment variable")
    key = os.environ.get("MOCK_LLM_API_KEY", "sk-demo-1234")
    masked = key[:7] + "*" * (len(key) - 7)
    src = "environment variable" if "MOCK_LLM_API_KEY" in os.environ else "default (for the demo)"
    print(f"    MOCK_LLM_API_KEY -> {masked} ({src})")
    print("    In practice: store it in the shell / a secrets manager, e.g.")
    print("          export ANTHROPIC_API_KEY=... — not one character of the key")
    print("          may appear in code or the git repository.")

    # [2] Inspecting the request/response structure
    print("\n[2] Request/response structure (one call, observed)")
    rng = random.Random(7)
    demo_req = {"model": "mock-llm-v1", "api_key": key,
                "messages": [{"role": "user", "content": "Summarize the remote-work policy"}]}
    demo_resp = fake_llm_api(demo_req, rng)
    while demo_resp["status"] != 200:        # for observation: loop until a success
        demo_resp = fake_llm_api(demo_req, rng)
    print(f"    request : model={demo_req['model']!r}, messages=[{{role, content}}]")
    print(f"    response: {demo_resp}")
    print("    -> The token counts in usage ARE the bill. Log them on every response.")

    # [3] Wrong key: recognizing errors where retrying is pointless
    print("\n[3] Calling with a wrong key (a 4xx error not worth retrying)")
    bad = RobustClient(api_key="sk-wrong-key", seed=1)
    bad.chat("this request is supposed to fail")

    # [4] Processing 5 requests with the retry/backoff/timeout client
    print("\n[4] 5 calls through the production-grade client (429/500/latency mixed in)")
    client = RobustClient(api_key=key, timeout=10.0, seed=7)
    prompts = ["summarize the vacation policy", "draft a notice about the expense deadline",
               "classify this complaint: delayed delivery",
               "summarize the minutes in 3 lines", "wording for the warranty-period notice"]
    for i, p in enumerate(prompts, 1):
        print(f"  ({i}) request: {p}")
        answer = client.chat(p)
        if answer:
            print(f"      answer: {answer}")

    # [5] Cost report
    print("\n[5] Cost-tracking report")
    print(f"    {client.cost_report()}")
    print("    -> Monthly budget = expected calls x average tokens x unit price.")
    print("       Watch it on a dashboard at all times.")

    print("\nSummary: build only the success path and you have a demo; build the failure")
    print("         paths too (429/500/timeout/key errors) and you have operations.")
    print("         Backoff, caps, and cost tracking are the basics of LLM calling.")

    # ------------------------------------------------------------------
    # [Reference] With the real Claude API (the SDK does retry/timeout for you)
    # import anthropic
    # client = anthropic.Anthropic(          # uses the ANTHROPIC_API_KEY env var
    #     timeout=20.0, max_retries=3)
    # resp = client.messages.create(
    #     model="claude-opus-5", max_tokens=1024,
    #     messages=[{"role": "user", "content": "summarize the vacation policy"}])
    # print(resp.usage.input_tokens, resp.usage.output_tokens)  # billing check
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
