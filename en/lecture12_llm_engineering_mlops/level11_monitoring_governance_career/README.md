# Lecture 12 · Level 11 — Monitoring, Cost, Governance — and Your Career Roadmap

> The craft of protecting a deployed model — drift detection, cost control, privacy and security — and the growth roadmap that closes this curriculum's journey.

**Difficulty** ⭐⭐⭐ / **Prerequisites** level10_serving_deployment / **Estimated time** 45 min

## 1. Why Learn This — The Business View

The real story begins the day after deployment. The model is the same as yesterday, but the world changes. A new product launches, slang is born, the customer base shifts. The distribution of questions the model receives drifts away from what it was trained on, and quality slides — silently. This is drift. The frightening part: the server keeps happily returning 200 OK. No outage alarm rings; only customer satisfaction quietly sinks.

That is why a live AI service needs three gauges on its dashboard: quality (satisfaction, error rate), cost (tokens are money), and risk (privacy, security). Tending these three is the operations half of MLOps, and it is the difference between a company that "adopted AI" and one that "runs AI responsibly." For people from non-CS backgrounds this is actually an opening: this territory is won with diligence, cost sense, and communication rather than cutting-edge math.

## 2. Grasping It Through an Analogy

Opening a new store (deployment) and running the store are different jobs. A packed opening day is not the end of it. A good store manager watches three things daily. First, customer response — if repeat visits creep downward, they hunt the cause: has the menu gone stale, or did a competitor open next door? (quality drift). Second, the books — if ingredient costs grow faster than sales, they adjust the recipes (cost control). Third, the locks — if an employee has the safe combination on a sticky note taped to the register, it comes down immediately (no hardcoded API keys).

Drift detection works like an annual physical. Only because you know your usual weight can you say "I've dropped 3kg — better see a doctor." You build a baseline from metrics in the normal period, and when today's metric strays statistically far from that baseline (the z-score), you ring the alarm.

## 3. Core Concepts

### 3.1 Drift — the Model Stands Still While the World Moves

- **Data drift**: the input distribution changes. Example: questions about a new product line pour in that the model never learned.
- **Performance drift**: the resulting decline in answer quality and satisfaction.

The detection procedure is simple. (1) Record metrics daily, (2) smooth the noise with a 7-day moving average, (3) compute the z-score against the normal period's mean and standard deviation, and (4) alert when it strays more than 2 sigma in the bad direction. A z-score of +13 means "far too far to be coincidence." The response after an alert: investigate the cause → augment the data → re-fine-tune (levels 06–07) → redeploy (level10) — the pipeline you learned in this lecture goes around one more time.

### 3.2 The Cost Dashboard — Tokens Are Money

LLM cost is essentially `token count × unit price`. Record daily token usage and you get "cost per request" and "projected monthly cost" — and only then do decisions become possible. If cost bites, review in this order: prompt diet (trim needless instructions) → caching (reuse identical questions) → routing to smaller models (send easy questions to a cheap model) → self-hosting (levels 09–10). None of these needs a coding genius. They need someone who measures.

### 3.3 Privacy and Security — the Governance Floor

- **Log masking**: raw customer inquiries arrive laced with emails, phone numbers, and addresses. Masking with regular expressions before storage is the minimum defense. It prevents the incident where every employee with log access can read customers' PII.
- **No hardcoded API keys**: the moment `api_key="sk-..."` lands in a repository, treat that key as leaked. Keys live in environment variables or a secrets manager, and secret scanning filters them out before every commit.
- **Version records**: log which model, data, and prompt version served when — otherwise you cannot roll back when something breaks.

### 3.4 The Career Roadmap — a Realistic Path from Non-CS to AI Professional

Having finished this curriculum, you are no longer "the person who only used spreadsheets." A realistic growth path looks like this:

1. **Stage 1 — Automate your own work (right now)**: use the Python, SQL, and LLM skills you learned to automate one repetitive task for your team. The in-house reputation of "just ask that person" is your first asset.
2. **Stage 2 — Domain × AI specialist (6 months–1 year)**: do not compete with pure developers; attach AI to *your* domain knowledge (sales, finance, logistics, HR). "An AI-fluent logistics person" is rarer than "a developer who only knows AI." Internal chatbot rollouts, data quality management, and prompt/eval design are good entry points.
3. **Stage 3 — AI product/operations owner (1–3 years)**: the monitoring, cost, and governance you learned in this level are that job's daily routine. Vanishingly few companies build models from scratch, but tens of thousands must run them responsibly.
4. **Keep going**: a portfolio (reproduce this curriculum's exercises on your own data), a steady execution habit (run something, anything, once a week), and change tracking (models keep changing, but this lecture's principles — tokens, training, alignment, quantization, serving, monitoring — endure).

## 4. Hands-On — main.py

How to run:

```bash
cd lecture12_llm_engineering_mlops/level11_monitoring_governance_career
python3 main.py
```

Stage-by-stage guide:

- **[1]** Generates 60 days of operational logs (requests, error rate, p95 latency, satisfaction, tokens) with seed-fixed random numbers. Drift — falling satisfaction, rising error rate — is quietly injected from day 45.
- **[2]** A mini dashboard showing the last 10 days as a table with text bars. Can you see the satisfaction bar shortening? To the naked eye it still reads as "probably nothing."
- **[3]** The key stage. Using the first 30 days as the baseline, it computes 7-day-moving-average z-scores for the three metrics. Satisfaction (z=-2.4) and error rate (z=+13.1) get alerts, while p95 latency — where no drift was injected — reads normal (z=-0.0). This is the moment statistics separates "probably nothing" from "an incident."
- **[4]** Converts the last 30 days of tokens to dollars (42.1M tokens → $126.25) down to the average cost per request ($0.0027).
- **[5]** Masks emails and phone numbers out of a raw customer inquiry with regular expressions to produce the storable log.
- **[6]** Detects hardcoded API-key patterns in a source-code string — a scaled-down version of a real pre-commit scan (1 secret found).
- **[7]** Closes with the five-line operations checklist.

The core of the code is `zscore_today()`. Note that one line — `(current moving average − baseline mean) / baseline standard deviation` — is the heart of a monitoring system, and that the alert direction (`direction`) differs per metric (lower is bad for satisfaction; higher is bad for error rate).

## 5. Try It Yourself

1. **(Basic)** Change `DRIFT_DAY` to 55 so the drift starts late — do the alerts disappear? Observe how many days the 7-day moving average needs to register the change. Hint: the crux is the point where the accumulated drift exceeds twice the baseline standard deviation.
2. **(Intermediate)** What happens if you lower the alert threshold from 2.0 to 1.0? Experiment with whether the healthy p95 latency starts throwing alerts on some days, and write one paragraph on the false-alarm vs. missed-alarm trade-off.
3. **(Challenge)** Add masking for US Social Security numbers (`###-##-####`) to `mask_pii()` and verify with a test string. Hint: `\d{3}-\d{2}-\d{4}` is the starting point, but add word boundaries (`\b`) so you do not misfire on other digit runs like dates or phone fragments.

## 6. Common Mistakes

- **Trusting only outage alarms.** An alarm that rings only when the server dies cannot see quality decline. Returning 200 OK with wrong answers is *the* signature failure mode of LLM services.
- **Judging by "feel" without a baseline.** If you never recorded the normal period's numbers, there is nothing to compare against when things worsen. Start accumulating metrics on deployment day one.
- **Overreacting to a single day's number.** Daily metrics naturally bounce. React to one bad day without a moving average and false alarms will wear you down until you switch the alerts off entirely.
- **Storing raw text in logs.** "We need it for debugging" is the classic first line of a privacy-breach postmortem. Make mask-then-store the default.
- **Putting a key in code "just temporarily."** Temporary always gets committed. Starting with environment variables from the first line is the only safe policy.

## Next Level Preview

This is the curriculum's final level, so instead of a next level, here is your next step. From the first terminal command in lecture01, you have come through Python, data, SQL, statistics, machine learning, deep learning, and the full process of building an LLM. Now reproduce these exercises on your own data to build a portfolio, pick one real problem at your company, and ship something small. Models will keep changing — but the principles you practiced across these twelve lectures will last. Congratulations on finishing.
