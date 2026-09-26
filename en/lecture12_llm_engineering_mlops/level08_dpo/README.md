# Lecture 12 · Level 08 — Preference Alignment (DPO)

> We implement the DPO loss exactly as written — raising the probability of "the answer a human picked" of two drafts — and watch the model's taste change through its log-probabilities.

**Difficulty** ⭐⭐⭐⭐⭐ / **Prerequisites** level07 / **Estimated time** 70 min

## 1. Why Learn This — The Business View

A model that has finished SFT answers in proper form. But "being able to answer" and "answering well" are different things. To a customer complaint, "that's the courier's problem" and "we're so sorry for the inconvenience" are both grammatically fine answers. Which is better is a matter not of correctness but of **preference** — and the stage that teaches it is preference alignment.

RLHF (reinforcement learning from human feedback), the step that made ChatGPT famous, was this stage. But RLHF requires training a separate reward model and running PPO reinforcement learning — a heavy, unstable process with four models in memory at once. DPO (Direct Preference Optimization), which appeared in 2023, achieves the same goal with **a single classification loss**, dramatically lowering the barrier to alignment, and has become the de facto standard for aligning open-source models. When the quality complaint is "our chatbot keeps answering brusquely," this stage is the remedy — and once you know the required data is just triples of (question, good answer, bad answer), you can design the data-collection plan yourself.

## 2. Grasping It Through an Analogy

A new hire (the SFT'd model) brings two draft replies to a customer. A senior colleague only marks in red "this one is better." No score, no rubric. Just **better of the two**.

- **The RLHF way** = first build a "grading robot" (reward model) that learns the senior's taste, then have the robot score every draft the new hire writes, correcting them via reinforcement learning. If the grading robot is flawed, the hire starts gaming its blind spots (reward hacking).
- **The DPO way** = no grading robot. Directly adjust the hire's habits: "write more like the drafts the senior picked, less like the ones that lost." With one safety rail: **don't drift too far from your original self (the reference model)**. Overhaul your whole way of speaking to win praise, and abilities that were fine start collapsing.

## 3. Core Concepts

### 3-1. Preference Data: (x, y_w, y_l)

One DPO example is a triple: prompt x, the selected response y_w (chosen/winner), and the discarded response y_l (rejected/loser). In practice you generate two model responses and have a human (or a stronger model) compare them. The starting insight of this data format: relative comparison is far easier and more consistent for humans than absolute scoring.

### 3-2. The DPO Loss — One Line of Math

```
L = -log σ( β·[ (log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x)) ] )
```

- log π(y|x): the current policy model's log-probability of generating that response after the prompt (the sum of the response tokens' logp).
- log π_ref: the same value from the reference model — a frozen copy of the pre-alignment self. It relativizes each response's probability as "how much did I raise/lower it versus my starting point."
- The bracketed part (the margin) is "how much chosen was raised minus how much rejected was raised." Push it through the sigmoid σ and take -log, and you get a **binary classification loss** that approaches 0 as the margin grows. At the start of training policy=ref, so the margin is 0 and the loss is exactly -log 0.5 = 0.693 — you can verify this value in the exercise output.
- **β (beta)**: sensitivity to straying from the reference model. Usually 0.1–0.5. Smaller β moves cautiously; larger β moves boldly.

The equation comes from a derivation: solve RLHF's "maximize reward + KL constraint" problem mathematically and the reward model cancels out, leaving exactly this form. DPO is a shortcut through RLHF, not a different goal. That is why β(log π − log π_ref) is called the **implicit reward**.

### 3-3. Why the Reference Model Is Needed

Without the reference terms, just "chosen ↑, rejected ↓" lets the model take the easiest path to suppressing rejected — for instance fleeing into a nonsensical output distribution. Relativizing against the reference makes the training "keep what you were already good at; only tilt in the preferred direction." Even so, with little data or long training, over-optimization sets in and the chosen probability starts falling too (see section 6).

### 3-4. RLHF vs. DPO at a Glance

| | RLHF (PPO) | DPO |
|---|---|---|
| Models needed | 4: policy, ref, reward model, value model | 2: policy, ref |
| Training style | reinforcement learning (unstable, hard to tune) | supervised-style classification loss (stable) |
| Reward | explicit reward model | implicit reward built into the loss |
| Cost | large | comparatively small |

## 4. Hands-On — main.py

Run:

```bash
cd lecture12_llm_engineering_mlops/level08_dpo
python3 main.py
```

- **[1]** Six customer-service preference pairs (question / polite chosen / blame-shifting rejected) are encoded (vocab 42).
- **[2]** Warm-up SFT: chosen and rejected are trained **with equal weight**, making both plausible. Real pipelines also start DPO from an SFT'd model.
- **[3]** The policy is deepcopy'd and frozen as the reference model. Look at the before-DPO table — every margin is exactly 0.00 (policy=ref), and with this seed the rejected side actually has the *higher* log-probabilities: chosen wins 0/6. Sure enough, greedy generation produces the rude 'Ask the courier company, not us.'
- **[4]** DPO trains for 30 epochs. `-F.logsigmoid(BETA * ((pw - rw) - (pl - rl)))` — the loss is the equation, in one line. Confirm the first epoch's loss is the theoretical 0.6931. Also note that the reference model's log-probs are computed once, outside the loop, because they never change.
- **[5]** The after-training table: chosen log-probs stay in the -3 to -7 range while rejected crashes to -25 to -45, opening margins of 22–40, and the win rate becomes 6/6. The greedy answer flips to 'So sorry. Let me check it right away.' **We taught no new answers — we tilted the preference between two answers it already knew.**

## 5. Try It Yourself

1. **β experiment**: change BETA to 0.5 and 0.02. How do the speed at which margins open and the final chosen log-probs differ? (Hint: bigger β aligns faster but strays boldly from the reference.)
2. **Watch over-optimization**: raise the DPO epochs from 30 to 200 and the lr to 5e-4. Rejected keeps crashing, but the chosen log-probs start falling too — find the point where greedy generation begins to break. (Hint: a famous post-DPO-paper phenomenon: the region where "the margin keeps growing while both get worse.")
3. **Flip the preferences**: swap chosen and rejected in PAIRS and retrain. Does the model really align into a "rude chatbot"? The alignment technique itself is direction-neutral; humans decide which way to align — which is exactly why governance (level11) is needed.

## 6. Common Mistakes

- **Training the reference model too**: fail to freeze ref, or use an alias (=) instead of deepcopy, and policy and ref move together — margins hover near 0 and nothing is learned.
- **Computing log-probability as a mean**: the standard DPO equation uses the **sum** of response-token logp. Mean-based variants exist, but mixing them changes what β means.
- **DPO without SFT first**: tilting preferences while both responses have essentially zero probability is unstable. "First make it able to write both, then tilt the taste" is the right order.
- **Judging success by the margin alone**: the margin is relative. Check the chosen side's absolute log-prob and real generation quality too, or you will miss over-optimization.

## Next Level Preview

The model is now smart and polite. The remaining problem: it is "heavy." In level09, quantization and compression, we shrink an fp16 model to int8/int4 — the technique that lets an LLM run on a laptop — implemented by hand in numpy.
