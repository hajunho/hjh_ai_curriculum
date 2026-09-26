"""
Hands-on LLM service monitoring and governance.
We simulate 60 days of operational logs (request counts, error rate, latency,
satisfaction, token usage) and build a text dashboard that detects quality
drift with moving averages and z-scores.
Cost reporting, PII masking, and hardcoded-API-key scanning too —
one screen showing what 'protecting' a model means once 'building' it is done.
"""

import re
import statistics

import numpy as np

rng = np.random.default_rng(42)
DAYS = 60
DRIFT_DAY = 45          # quality quietly starts degrading on this day (e.g., a catalog overhaul)
BASELINE_DAYS = 30      # the first 30 days serve as the 'normal baseline'
PRICE_PER_MTOK = 3.0    # assume $3 per million tokens


def simulate_logs() -> dict:
    """Generate daily operational metrics. After DRIFT_DAY, inject satisfaction↓ and error-rate↑ drift."""
    day = np.arange(DAYS)
    requests = (1200 + 8 * day + rng.normal(0, 60, DAYS)).astype(int)  # gentle growth
    error_rate = np.clip(rng.normal(0.010, 0.003, DAYS), 0, None)
    latency_p95 = rng.normal(820, 40, DAYS)                            # ms
    feedback = rng.normal(4.35, 0.08, DAYS)                            # out of 5
    tokens = requests * rng.normal(900, 40, DAYS)                      # tokens per day
    drifted = day >= DRIFT_DAY
    feedback[drifted] -= 0.012 * (day[drifted] - DRIFT_DAY + 1)        # gradual decline
    error_rate[drifted] += 0.0022 * (day[drifted] - DRIFT_DAY + 1)     # gradual rise
    return {"requests": requests, "error_rate": error_rate,
            "latency_p95": latency_p95, "feedback": feedback, "tokens": tokens}


def moving_avg(x: np.ndarray, w: int = 7) -> np.ndarray:
    """Mean of the last w days. Early days average over whatever exists."""
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def zscore_today(series: np.ndarray, baseline: np.ndarray) -> float:
    """Today's z-score (of the 7-day moving average) against the baseline distribution."""
    return (series[-1] - baseline.mean()) / (baseline.std() + 1e-12)


def bar(v: float, vmax: float, width: int = 24) -> str:
    return "#" * max(1, int(width * v / vmax))


def mask_pii(text: str) -> str:
    """Mask emails and phone numbers before the log is stored."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<EMAIL>", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "<PHONE>", text)
    return text


def scan_secrets(code: str) -> list[str]:
    """Find traces of hardcoded API keys in source code."""
    patterns = [r"sk-[A-Za-z0-9]{16,}", r"(api_key|API_KEY)\s*=\s*['\"][^'\"]{8,}"]
    return [m.group(0) for p in patterns for m in re.finditer(p, code)]


if __name__ == "__main__":
    logs = simulate_logs()

    print("[1] Operational log simulation — 60 days of daily metrics")
    print(f"    Metrics: requests, error rate, p95 latency, user satisfaction (out of 5), token usage")
    print(f"    (We secretly injected drift starting on day {DRIFT_DAY}. Let's detect it.)")

    print("\n[2] Dashboard — the last 10 days")
    print(f"    {'day':>4s} {'reqs':>6s} {'err%':>7s} {'p95(ms)':>8s} {'sat.':>6s}  satisfaction trend")
    fb_ma = moving_avg(logs["feedback"])
    for d in range(DAYS - 10, DAYS):
        print(f"    {d + 1:>4d} {logs['requests'][d]:>6d} "
              f"{logs['error_rate'][d]:>6.2%} {logs['latency_p95'][d]:>8.0f} "
              f"{logs['feedback'][d]:>6.2f}  {bar(fb_ma[d] - 3.5, 1.0)}")

    print("\n[3] Drift detection — z-score of the 7-day moving average vs. the baseline (first 30 days)")
    alerts = 0
    for name, series, direction in [("satisfaction", logs["feedback"], -1),
                                    ("error rate", logs["error_rate"], +1),
                                    ("p95 latency", logs["latency_p95"], +1)]:
        ma = moving_avg(series)
        z = zscore_today(ma, series[:BASELINE_DAYS])
        status = "normal"
        if direction * z > 2.0:  # alert when more than 2 sigma off in the bad direction
            status = "!! ALERT: drift suspected"
            alerts += 1
        print(f"    {name:12s} baseline {series[:BASELINE_DAYS].mean():8.3f} -> "
              f"now (7-day avg) {ma[-1]:8.3f}  z={z:+5.1f}  {status}")
    print(f"    -> {alerts} alerts. The model is unchanged — when the world (the input distribution) shifts, this is how it shows.")

    print("\n[4] Cost report — converting tokens to money")
    month_tokens = logs["tokens"][-30:].sum()
    month_cost = month_tokens / 1e6 * PRICE_PER_MTOK
    per_req = month_cost / logs["requests"][-30:].sum()
    print(f"    Last 30 days: {month_tokens / 1e6:,.1f}M tokens -> ${month_cost:,.2f}"
          f" (assuming ${PRICE_PER_MTOK:.2f} per million tokens)")
    print(f"    Average cost per request: ${per_req:.4f}")
    print(f"    -> If this exceeds the revenue contribution, look into caching, smaller models, and prompt diets.")

    print("\n[5] Privacy — masking before anything lands in the logs")
    raw = "Inquiry: please refund my order. You can reach me at kim.cs@example.com / 555-123-4567."
    print(f"    raw log    : {raw}")
    print(f"    stored log : {mask_pii(raw)}")
    print("    -> Store the raw text and every employee with log access can see the customer's PII.")

    print("\n[6] Security check — detecting hardcoded API keys (a mini pre-commit scan)")
    bad_code = 'client = Client(api_key="sk-live-abcdef1234567890XYZ")  # TODO remove'
    found = scan_secrets(bad_code)
    print(f"    code under scan: {bad_code[:52]}...")
    print(f"    secrets found: {len(found)}: {[f[:14] + '...' for f in found]}")
    print("    -> Keys belong in environment variables or a secrets manager, not in code.")

    print("\n[7] Wrap-up — the operations checklist")
    for item in ["Are quality metrics (satisfaction, error rate) collected automatically every day?",
                 "Is there a path that delivers drift alerts to a human?",
                 "Is PII masked in the logs?",
                 "Are API keys outside the code (in environment variables)?",
                 "Are model and data versions recorded so you can roll back anytime?"]:
        print(f"    [ ] {item}")
    print("    These five lines are the difference between 'someone who builds models'")
    print("    and 'someone who is accountable for a service'.")
