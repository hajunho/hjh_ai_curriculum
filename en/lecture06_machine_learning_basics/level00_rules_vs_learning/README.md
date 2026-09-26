# Lecture 06 · Level 00 — What Is Machine Learning? — Rules vs Learning

> We pit two approaches against each other on the very same problem: a human writing out rules one by one, versus a machine discovering the rules from data on its own.
**Difficulty** ⭐ / **Prerequisites** none / **Estimated time** 25 min

## 1. Why Learn This — The Business View

In a meeting where someone asks "shouldn't we be adopting AI too?", the first thing you need is not technology — it's the ability to tell problems apart. Some problems are perfectly handled by a spreadsheet formula or a few IF statements; others simply cannot be solved without machine learning (ML). If you can't make that distinction, you'll either commission a project worth tens of thousands of dollars for something an IF statement could do, or try to patch a genuinely ML-shaped problem with hundreds of hand-written rules and sink into maintenance hell.

By the end of this level, you will have your own criteria for answering "should we go rule-based or learning-based?" — and those criteria are the launching point for all eleven levels that follow.

## 2. Understanding by Analogy

Imagine **two ways to build a spam filter**.

Approach A — hard-coded rules: this is like handing a new hire a procedures manual. "If the subject line contains 'loan', it's spam." "If the sender isn't in the address book and there are 3 or more links, it's spam."… It works at first. But then the spammers change 'loan' to 'l-o-a-n'? You add another rule. By the time you have 500 rules, they start contradicting each other, and nobody understands the whole thing anymore.

Approach B — learning: instead of a manual, you show the new hire **100,000 emails that senior staff have already sorted into spam/legitimate**. "Look through these and get a feel for it." Even though no human wrote any rules, with enough data the new hire learns to catch 'l-o-a-n' on their own. When a new trick appears, you just retrain on new data.

Machine learning is exactly Approach B. Traditional programming is **"rules + data → answers"**; machine learning is **"data + answers → rules"**. The arrow has been flipped — that is the single most important sentence in this entire lecture.

## 3. Core Concepts

### 3.1 Traditional Programming vs Machine Learning

| | Traditional programming | Machine learning |
|---|---|---|
| What the human provides | Rules (code) + inputs | Inputs + labeled examples |
| What the computer produces | Answers | Rules (a model) |
| When the rules change | Edit the code | Retrain on data |
| Well-suited problems | Rules are clear and few (tax calculation, payroll) | Rules are fuzzy and countless (image recognition, churn prediction) |

### 3.2 What "Learning" Really Is — A Search for Good Values

The word *learning* sounds mysterious, but the reality is surprisingly humble: **"trying out different numbers (parameters) and searching for the value that best matches the known answers."** Today's exercise uses the simplest possible form. We fix a template — "predict fraud if the transaction amount is at least T" — then try every candidate T and pick the one that fits the data best. Instead of a human deciding T = 500,000 by gut feel, we let the data choose T. That is the seed of learning. Linear regression and neural networks, which you'll meet later, are the same thing at heart — they just have millions of numbers and a cleverer search method.

### 3.3 Rules Often Win, Too

Machine learning is not always superior.

- **When the rule is fixed by law or policy**: never use ML to compute a 10% sales tax. A 100%-accurate rule already exists.
- **When you have almost no data**: 20 labeled examples won't support learning. An expert's rules are better here.
- **When you must explain every mistake completely**: rules are easy to trace; a learned model needs separate interpretation work (level11).

The standard playbook in practice: "start with rules, and switch to learning at the point where rules can no longer keep up."

### 3.4 Terminology

- **Feature**: an input column the decision is based on. Example: transaction amount, transaction time
- **Label**: the answer column to predict. Example: fraud or not (0/1)
- **Model**: the "bundle of rules" produced by learning. Today we build an ultra-mini model with a single threshold.

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

This program takes card-fraud transaction data (synthetic, 5,000 rows, fraud rate about 1.4%) and solves the same classification problem three ways. It has exactly the same structure as the spam filter — picking out a rare bad minority from an overwhelmingly normal majority. (Amounts in this dataset are in Korean won, KRW.)

- **[1]** Load the data and inspect its structure. Because fraud is so rare, plain accuracy is an illusion, so we grade with "the average of the fraud hit rate and the normal hit rate (the balanced score)."
- **[2] Hand-made rule**: we grade the kind of gut-feel rule you'd hear in a meeting room — "500,000 KRW or more, in the small hours → fraud." Everything it flags is real, but it misses 70% of the fraud.
- **[3] Learning (threshold search)**: pick one feature, try every threshold, and predict with the optimal threshold the data chose. Be sure to check the output for which feature and which value got picked.
- **[4] Learning (two features combined)**: keep the exact shape of the human rule ("amount at least T AND small hours") but let the data pick the number T. See how far the human's 500,000 is from the data's answer.
- **[5]** Compare the three approaches in a table and recap the key idea: "the arrow has been flipped."

The heart of the code is the `learn_threshold()` function — ten lines that loop over every candidate threshold with a for statement and score each one. This is the first learning algorithm you will ever build. We deliberately avoided libraries like sklearn, so you can see with your own eyes that learning is search, not magic.

What to notice in the output: the human rule was biased toward "only flag the sure things and miss most of the rest," while learning finds a number with a better balance between misses and false alarms. The human decides only the *shape* of the rule; the data decides the *number* — that division of labor is where machine learning begins.

## 5. Try It Yourself

1. **(Easy)** In `[2]`, change the amount cutoff in the hand-made rule from 500,000 to 100,000 and rerun. How does the score change? Feel for yourself why picking numbers by gut is so precarious. (Hint: edit just one number in the `hand_rule` function.)
2. **(Medium)** Add `tx_count_1h` (number of payments in the last hour) to the list of features searched in `[3]`. You'll get a startling result. This synthetic dataset hides a near-perfect feature — with real data, the moment you see a feature like that, your first suspicion should be "isn't this leakage?" (covered in level02).
3. **(Challenge)** Right now `learn_threshold()` only searches the "fraud if value ≥ T" direction. Extend it to also search "fraud if value < T". It should help on features like `hour`, where *small* values are the risky ones. (Hint: loop twice with a direction flag set to True/False.)

## 6. Common Mistakes

- **Believing "AI = machine learning = a cure-all"**: using learning on a problem with clear rules is like rolling dice when you own a calculator.
- **Mystifying learning**: it's not "the model thinks for itself" — it's "searching over numbers to fit the data." Mystify it and you'll forget to verify the results.
- **Grading on the same data you trained on and calling it a day**: today we did exactly that, on purpose. Why that is dangerous is the topic of level04. For now, just remember today's report card is "an optimistically inflated number."
- **Trying to learn before you have labeled data**: Approach B only works because "100,000 emails sorted by senior staff" exist. Preparing the data is more than half the project.

## Next Level Preview

What we built today was supervised learning — learning from data with the answers attached. But what if there are no answers? What if there are only rewards instead? In level01 we sort out machine learning's three learning paradigms — supervised, unsupervised, and reinforcement learning — in one sitting.
