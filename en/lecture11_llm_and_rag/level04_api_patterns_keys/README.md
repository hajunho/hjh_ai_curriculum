# Lecture 11 · Level 04 — API Calling Patterns and Key Management

> An LLM is a paid API across the network — practice the calling skills operations demands, from key security and token billing to retry, backoff, and timeouts, against a fake server.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level03 / **Estimated time** 50 min

## 1. Why Learn This — The Business View

The moment you wire an LLM into a business system, three realities hit. First, **every call costs money** — metered billing per token. Second, **calls sometimes fail** — rate-limit rejections (429), server errors (500), and slow responses are part of normal operations. Third, **a leaked API key means someone else spends on your company's card** — pushing a key to GitHub and waking up to a four-figure bill is an accident that keeps happening.

This is where demo and production part ways. Code with only a success path is a demo; code with the failure paths too — retry, timeout, spending caps, key security — is production. This level covers the four fundamentals that make the difference.

## 2. Grasping It Through an Analogy

An LLM API is like **an external translation bureau you hire over the phone**.

- **API key = the corporate card number**: they only take the job after you read out the card number. Write that number on a sticky note on your monitor (hardcode it) and someday it gets photographed. Keep it in the safe (environment variables, a secrets manager).
- **Token billing = per-volume translation fees**: the longer the source (input tokens) and the longer the translation (output tokens), the higher the fee. Keep sending work without an estimate and the month-end invoice will surprise you.
- **A 429 error = "we're swamped right now"**: when the bureau is busy they say "please call back shortly." Calling back every second (mindless retries) only worsens the jam. Calling back after 1 minute, then 2, then 4… is exponential backoff.
- **Timeout = "if no reply within 30 minutes, the deal's off"**: waiting forever stalls your whole workflow. Set a ceiling on waiting; past it, hang up and try again.

## 3. Core Concepts

### 3.1 The request/response structure

The request carries the model name and the list of chat messages; the response carries the generated text and **usage (token counts)**. Recording usage on every call is where cost management begins. In English, a token is roughly 4 characters (it varies by model).

### 3.2 Two families of errors — retry or don't

| Family | Examples | Response |
|---|---|---|
| Transient (worth retrying) | 429 rate limit, 500 server error, timeout | back off, then retry |
| Configuration (retrying is pointless) | 401 key error, 400 malformed request | stop immediately; fix code/config |

Retrying a 401 with backoff is like redialing with the wrong card number — telling the two apart matters.

### 3.3 Exponential backoff + jitter

Stretch the retry interval 1s → 2s → 4s → 8s (exponential) and add a dash of random time (jitter). Without jitter, the hundreds of clients that failed together all retry together and knock the server over again. Always set a cap (say, 4 retries); past it, queue the task or alert a human.

### 3.4 Three rules of key management

1. No keys in code or the repository — read them with `os.environ.get("ANTHROPIC_API_KEY")`.
2. Never print the full key to screens or logs — mask all but the first few characters.
3. Suspect a leak? Revoke and reissue immediately — design so keys are swappable at any time.

### 3.5 Bulk work goes through batch

For jobs that don't need real-time answers (classifying 10,000 documents overnight), don't call one by one — look at batch processing. Major APIs offer a batch lane, usually at half price. Even without batch, when calling in bulk it is better to space the calls so you never hit the per-minute limit than to survive on retries. "Pacing yourself in advance" and "backing off after exceeding" are complementary — a production-grade client has both.

## 4. Hands-On — main.py

Run it:

```bash
cd lecture11_llm_and_rag/level04_api_patterns_keys
python3 main.py
```

Reading the output:

- **[1]** The key is read from an environment variable and displayed masked. Set `export MOCK_LLM_API_KEY=...` and the reported source switches to "environment variable".
- **[2]** Observe the request/response JSON structure. The input/output tokens in `usage` are the basis of billing.
- **[3]** Calling with a wrong key returns 401 — confirm the branch that stops immediately without retrying.
- **[4]** While processing 5 requests, all four failure paths appear: 429 (backoff and retry), 500 (retry then success), a 45-second response (timeout handling), and retry-limit exceeded (mark failed, alert a human — request 1 in our run exhausts all 5 attempts). Waits are shrunk to 1/100 for the demo, so it finishes instantly.
- **[5]** The cumulative token and cost report (fictional rates): in our run, 4 successful calls, 7 retries, 37 input + 56 output tokens = $0.000951 (about KRW 1.33).

Code heart: `fake_llm_api()` is a fake server that reproduces a real API's success/429/500/latency with seed-fixed randomness. `RobustClient.chat()` is the core of this level, embodying the operational pattern "call → branch on status (success / permanent error / transient error) → backoff wait → retry → give up past the cap." As the comment at the bottom shows, the real Claude SDK builds in retry and timeout as options (`timeout=`, `max_retries=`) — but you have to know the principle to set the options well.

## 5. Try It Yourself

1. **(Easy)** Run `export MOCK_LLM_API_KEY=sk-demo-1234` in your terminal and rerun to see the source in [1] change. What happens to [4] if you set a wrong value (`sk-oops`)?
2. **(Medium)** Add a "budget cap" feature to `RobustClient`: once cumulative cost exceeds $0.001, refuse further calls and print a warning. (Hint: at the top of `chat()`, compute the current cost with the same formula `cost_report` uses.)
3. **(Challenge)** Raise the 429 probability in `fake_llm_api` from 0.25 to 0.5. What happens to the success rate and retry counts? Experiment: is raising `max_retries` or spacing the calls (waiting before each call) more effective?

## 6. Common Mistakes

- **Hardcoding the key** — `api_key = "sk-..."` in code becomes an incident the moment it's pushed. A key left in commit history is recoverable even after deletion, so reissue immediately.
- **Retrying every error the same way** — 401/400 will fail no matter how often you resend. Branch on the error code.
- **Retrying without backoff** — a retry-on-failure-immediately loop worsens the rate limiting and just burns money.
- **No timeout configured** — when the default is very long, one slow response stalls the whole batch for minutes.
- **No cost logging** — throw away `usage` and the month-end invoice becomes your first cost report. Record per call.

## Next Level Preview

That was "the art of talking to the LLM"; next comes "the art of finding your company's documents for the LLM." Level 05 covers embeddings — turning text into coordinates (vectors) — and semantic search: the first component of RAG.
