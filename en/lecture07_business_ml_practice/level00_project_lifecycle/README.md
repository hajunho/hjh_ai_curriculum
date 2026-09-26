# Lecture 07 · Level 00 — The Full Life Cycle of a Data Project

> Problem framing → data → model → deployment → monitoring: put the five-stage map in your head and learn the failure patterns of doomed projects before they happen to you.
**Difficulty** ⭐ / **Prerequisites** lecture06 level00–02 / **Estimated time** 25 min

## 1. Why this matters — the business view

The number-one cause of machine learning project failure is not the model. In the post-mortems the industry keeps repeating, the story is almost always one of these: "we never agreed on what to predict in the first place," "the data couldn't answer that question," or "we built it and nobody used it." In other words, failure happens not in the coding stage but in the stages before and after it.

If you're not an engineer, here is exactly why you need this map: your job on a real project is not to write model code, but to judge where the project is on the five-stage map right now and at which gate it needs to stop. Whether you're the sponsor, the business owner, or the reviewer, this one map changes the questions you ask in the meeting.

## 2. An analogy to hold onto

A data project is surprisingly similar to **opening a restaurant**.

1. **Problem framing = choosing the location and the menu**: a restaurant that starts with "we'll just make everything tasty" goes under. You first decide who you're selling to (target customer), what (the menu), and at what bar for success (the success criterion).
2. **Data = sourcing ingredients**: even a brilliant chef can't cook with spoiled ingredients. Skip the incoming inspection (data quality checks) and the accident happens in the kitchen (the model).
3. **Model = cooking**: roughly 20% of the total work. Just as most of opening a restaurant isn't the cooking itself, most of a project's time goes to the stages before and after.
4. **Deployment = opening day**: even a finished dish feeds nobody if there's no path to the customer's table (the connection to existing business systems).
5. **Monitoring = keeping the regulars**: opening-day quality doesn't hold forever. When ingredient prices shift (the data distribution changes), you adjust the recipe.

"We just need a good model" is dangerous for the same reason "good cooking is all a restaurant needs" is.

## 3. Core concepts

### 3.1 The five stages and each stage's deliverable

| Stage | Key question | Typical deliverable |
|---|---|---|
| ① Problem framing | What, why, and what counts as success? | Problem spec (prediction target, how results will be used, target metric) |
| ② Data | Do we have data that can answer that question? | Data audit report (row counts, missing values, target ratio) |
| ③ Model | Is it better than the baseline? | Validated performance report (with cross-validation) |
| ④ Deployment | Can the business actually use it? | Hook into the business process (e.g., an at-risk customer list auto-sent every week) |
| ⑤ Monitoring | Is performance holding up? | Performance dashboard, retraining criteria |

The crucial point is that there is a **gate** between every pair of stages. If you fail a gate, you don't move on — you stop or go back. Example: if the data audit finds only 50 churned customers in the sample, the right move is to collect more data, not to proceed to the model stage.

### 3.2 What failing projects have in common

- **No target metric**: a project that starts as "let's do something with AI" can never be judged a success or a failure, no matter what comes out.
- **Sloppy ground-truth labels**: if every department defines "churn" differently (cancellation request? no login for 3 months?), the model ends up learning a different problem than the one you meant.
- **No baseline**: models that perform worse than the naive rule "predict the same as last month" really do get deployed.
- **No deployment plan**: development starts before anyone decides which person, system, or business procedure will actually receive the probability numbers the model spits out.
- **No monitoring**: an environmental shock like COVID shifts the data distribution (data drift) and nobody notices.

### 3.3 The reality of time allocation

Textbooks spend most of their pages on models, but real-world time allocation looks more like "problem framing and data 60–70%, model 10–20%, deployment and monitoring 20–30%." The structure of this lecture (8 building-block levels + 4 case studies) mirrors that reality.

## 4. Hands-on — main.py

```bash
python3 main.py
```

main.py is a simulation that pushes a fictional project — "subscription churn defense" — through the five stage gates.

- [1] Problem framing: checks whether the spec items (prediction target, how results will be used, target metric) are filled in, and prints the gate verdict.
- [2] Data: actually loads `hjh_data.churn_table()` to audit row counts, missing values, and the target ratio, and passes the "do we have enough minority-class samples?" gate.
- [3] Model: since we haven't built models yet, it computes the accuracy of the "majority-vote baseline" — the bar any future model must clear.
- [4]–[5] Deployment and monitoring: some checklist items are deliberately left blank so you can watch the gate return a "STOP" verdict.

In the summary table at the end of the output, check which stages PASS and which say STOP. The heart of the code is the `gate()` function: it takes a stage's checklist and refuses to pass if even one item is blank — extremely simple, and exactly the logic most often skipped in practice.

## 5. Try it yourself

1. **(Easy)** In stage [4] deployment, fill the blank items (`""`) with real plan sentences and rerun. Confirm the verdict flips to PASS.
2. **(Medium)** Raise `min_minority` to 500. The data gate turns to STOP. Using README section 3.1, write down which options a practitioner should weigh at that point (extend the collection window, relax the definition, shelve the project).
3. **(Challenge)** Pick one problem from your company (real or imagined) and write the [1] problem-framing dictionary yourself. The key test: can you state the "target metric" as a number? Hint: "reduce churn" (x) → "monthly churn rate from 18% to 15% within 6 months" (o).

## 6. Common mistakes

- **Picking the model first and bending the problem to fit**: "let's try that trendy technique" is like buying the oven before deciding the menu.
- **Adopting accuracy as the success criterion by default**: on data with an 18% churn rate, saying "nobody churns" already scores 82% accuracy. Accuracy without a baseline is a decorative number.
- **Confusing a one-off analysis with an operational system**: an analysis that ends with one report and a system that must run every week have completely different requirements. Agree on which one it is before you start.
- **Deciding gates by emotion**: "we've come this far, it'd be a shame to stop" is the voice of sunk cost. Gates run on checklists, mechanically.

## Next level preview

You have the map — now you learn the language. In level01 we convert each cell of the confusion matrix into an **amount in KRW**, and build the calculator that answers "how much money is 1 percentage point of recall worth?"
