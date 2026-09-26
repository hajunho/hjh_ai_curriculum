# Lecture 06 · Level 02 — Turning a Business Question into an ML Problem

> You learn the skill of translating a wish like "we want to reduce churn" into a specification a machine can solve (what to predict, at what unit, as of when, judged by which metric).
**Difficulty** ⭐⭐ / **Prerequisites** level01 / **Estimated time** 35 min

## 1. Why Learn This — The Business View

The most common reason machine-learning projects fail is not a bad model. **It's starting before the problem was ever defined.** "We want to reduce churn" is a fine executive goal, but you can't hand it to a machine as-is. What exactly is churn? Three months of inactivity? Clicking the cancel button? At what unit do you predict — a customer? An account? A whole family plan? As of when do you predict — the 1st of each month? And did you know you may only use information actually known at that moment?

This translation work isn't coding, which is exactly why a business owner can do it better than a data scientist. And if it goes wrong here, no amount of modeling afterward can save you. This is the only level in the lecture where we don't build a model — yet in practical value it may be the biggest one.

## 2. Understanding by Analogy

**Defining an ML problem is like writing a statement of work for a contractor.** Tell an interior contractor "make my house pretty" and the result is a coin flip. A good statement of work is different: it says which rooms (unit), by when (timing), in what style (target), what the budget and acceptance criteria are (metric), and what happens after completion (action).

A machine-learning model is a contractor you cannot reason with. If the statement of work (the problem spec) is vague, it will diligently build the wrong thing. The truly frightening part: **the model never asks about the gaps in your spec.** Omit the condition "don't feed it information unavailable at prediction time," and the model will quietly use that information to deliver a counterfeit that only aces the test (this is called leakage).

## 3. Core Concepts

### 3.1 The 5 Elements of an ML Spec

Five questions you must answer when converting a business question into an ML problem.

1. **Target**: what exactly are you predicting? Use a measurable definition. — "churn" → "as of end of this month, cancels the subscription within the next 30 days (1/0)"
2. **Unit**: what is one row? — One customer? One account? One order? One store-day?
3. **Timing**: when is the prediction made, and are you using **only information actually known at that moment**? — Violate the timing and you get leakage.
4. **Metric**: how do you measure success, and does that metric connect to business profit and loss? — For churn prediction, recall and precision beat accuracy (level06).
5. **Action**: what happens once the prediction is out? — "Call the top 10% highest-risk customers." **A prediction with no action is a decoration.** Only once the action is fixed does the model's output format (like "top 10%") get fixed too.

In practice you add a **baseline** — "what score does the current approach (gut feel, rules) get without any model?" A model that can't beat the baseline has no reason to be deployed.

### 3.2 Solvable Problems vs Unsolvable Problems

Even written in the same format, some problems are ML-solvable and some are not.

**Conditions for a solvable problem**: (a) similar events happened repeatedly in the past and left data behind, (b) a pattern plausibly exists between inputs and outcomes, (c) answers (labels) can be obtained, (d) the action still works even when some predictions are wrong.

**Signals of an unsolvable one**: only a few dozen labeled cases exist ("predict the success of our company's M&A deals over 10 years"), the rule is already 100% clear (sales-tax calculation — level00), there's no action you could take even with a prediction, past patterns are unlikely to hold in the future (right after a total regime change), or lives/legal liability are at stake and error tolerance is zero.

### 3.3 Leakage — Ingredients That Peek into the Future

The item to watch hardest when reviewing a spec. Example: a "cancellation penalty charged" column predicts churn almost perfectly — because it's only ever charged to people who canceled. But at prediction time (start of the month) that information does not exist. Be suspicious of any column that exists in the training data but not at prediction time, or that only gets filled in after the outcome is settled. The one test question: **"At the moment you press the predict button, can you know this value?"**

### 3.4 A Spec Is a Document You Keep Refining

The first spec is usually wrong. Define churn as "cancels within 30 days" and you may discover, once you look at the data, that every customer on a 45-day billing cycle gets flagged as churned. Going back and forth — spec → check the data → revise the spec — is the normal process.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

This program is a **checklist engine**. Write an ML spec as a Python data structure, and it automatically checks whether the 5 elements are filled in, whether there's leakage risk, and whether the problem suits ML — then scores it.

- **[1]** Feed in "we want to reduce churn" as a bad spec (mostly blank) → get called out on what's missing.
- **[2]** Complete the same wish as a good spec → watch it pass and absorb the 5 elements.
- **[3]** Leakage check: submit a spec whose feature list includes "cancellation penalty charged" and see whether the engine catches it (detecting columns created after prediction time).
- **[4]** Automatically classify 4 candidate problems (demand forecasting, sales-tax calculation, M&A success prediction, fraud detection) as "solvable/unsolvable" and print the reasons.

The heart of the code is the `MLSpec` dataclass and the `validate()` function. Each check rule corresponds 1:1 with a concept from section 3, so read the output and the code side by side. You can copy this checklist straight onto a whiteboard in a real meeting.

## 5. Try It Yourself

1. **(Easy)** In [2]'s good spec, change `action` to an empty string and rerun. What does the engine say? Find how the rule "a prediction with no action is a decoration" is expressed in code.
2. **(Medium)** Write one real concern from your own job as an `MLSpec` and run it through `validate()`. Which of the 5 elements is hardest to fill in? (In our experience: timing and action.)
3. **(Challenge)** Add a new rule to `check_leakage()`: warn when a feature name contains "satisfaction survey" and the survey happens later than prediction time. (Hint: compare the `available_at` field.)

## 6. Common Mistakes

- **Equating the goal with the target**: the goal is "reduce churn" (business); the target is "cancels within 30 days, yes/no" (ML). The model only does the latter. Reducing churn is produced by prediction + action (outreach, incentives) together.
- **Writing "accuracy" as the metric by default**: on data with a 15% churn rate, insisting "everyone stays" already gets 85% accuracy. Metric selection is covered in depth in level06.
- **Mistaking a leaky column for a performance breakthrough**: if the test score suddenly gets suspiciously good, suspect leakage before you celebrate.
- **Trying to write the perfect spec in one pass**: a spec is a living document you revise as you look at the data.

## Next Level Preview

Now that you can write the spec, it's time to build your first real model. In level03 we answer "if we raise ad spend, how much does revenue grow?" with linear regression, and see how a single straight line becomes a prediction tool.
