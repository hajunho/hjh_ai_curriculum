# Lecture 06 — Introduction to Machine Learning

> Instead of hard-coding rules, you will learn — from start to finish — how to have a machine learn patterns from data.
> You will run the flagship models for regression, classification, and clustering yourself, and tackle the question "how do we know when to trust a model?" (evaluation, overfitting, interpretation).

## What You Will Learn in This Lecture

- How machine learning (ML) differs from traditional programming, and when to use it — and when not to
- How to turn a business question ("we want to reduce churn") into an ML problem specification
- How linear regression, logistic regression, decision trees, random forests, and gradient boosting work and how to use them
- Train/validation/test splits, evaluation metrics, overfitting and regularization — the craft of trusting a model *properly*
- How to find structure in data with no answer key, using clustering and dimensionality reduction
- How to explain a model to non-specialists with feature importance and per-prediction explanations

## Prerequisites

- **lecture02 — Python Programming Basics** (you should be able to read functions, lists, and dictionaries)
- **lecture03 — Working with Data** (NumPy arrays, Pandas DataFrames, handling missing values)

High-school-level intuition (the slope of a line, basic probability) is all the math you need. We explain with analogies and experiments rather than equations.

## Level Index

| Level | Title | Difficulty |
|---|---|---|
| [level00](level00_rules_vs_learning/README.md) | What Is Machine Learning? — Rules vs Learning | ⭐ |
| [level01](level01_learning_paradigms/README.md) | Supervised, Unsupervised, and Reinforcement Learning | ⭐ |
| [level02](level02_framing_business_problems/README.md) | Turning a Business Question into an ML Problem | ⭐⭐ |
| [level03](level03_linear_regression/README.md) | Linear Regression | ⭐⭐ |
| [level04](level04_train_valid_test/README.md) | Train / Validation / Test Splits | ⭐⭐ |
| [level05](level05_logistic_regression/README.md) | Logistic Regression — Where Classification Begins | ⭐⭐⭐ |
| [level06](level06_evaluation_metrics/README.md) | Evaluation Metrics — The Accuracy Trap | ⭐⭐⭐ |
| [level07](level07_decision_trees/README.md) | Decision Trees | ⭐⭐⭐ |
| [level08](level08_random_forest_ensembles/README.md) | Random Forests and Ensembles | ⭐⭐⭐ |
| [level09](level09_overfitting_regularization/README.md) | Overfitting, Regularization, and Hyperparameters | ⭐⭐⭐⭐ |
| [level10](level10_clustering_dimreduction/README.md) | Clustering and Dimensionality Reduction | ⭐⭐⭐⭐ |
| [level11](level11_boosting_interpretation/README.md) | Gradient Boosting and Model Interpretation | ⭐⭐⭐⭐⭐ |

## Fast Track (Short on Time? Do These 5)

1. **level00** — Rules vs learning: understand in your bones what machine learning is
2. **level03** — Linear regression: your first predictive model
3. **level04** — Splitting data: if you don't know the self-grading trap, every result becomes a lie
4. **level05** — Logistic regression: the starting point for real-world classification problems
5. **level06** — Evaluation metrics: how not to be fooled by "98.5% accuracy"

## Where This Lecture Shows Up at Work

- **Subscription/membership operations**: find the customers likely to cancel next month, and concentrate coupons and outreach on them (level05, 07, 11)
- **Marketing budget meetings**: answer "if we raise ad spend by 10 million KRW, how much will revenue grow?" with a defensible number (level03)
- **Fraud and abnormal-transaction monitoring**: on data where fraud is 1.5% of cases, speak in recall and precision instead of accuracy (level06)
- **Customer segmentation**: with no answer key, split customers into groups from purchase patterns alone and build a strategy per group (level10)
- **Working with vendors and data teams**: ask the questions that let you verify a report like "the AUC is 0.85" yourself (level04, 06, 09)
- **Executive reporting**: convey "why the model flagged this customer as risky" using feature importance and per-prediction explanations (level11)

## Tips for Working Through It

- Every level is designed to be read while running `main.py` yourself. Reading the notes alone gets you only half the learning.
- To run: use the virtual environment at the repository root, `python3 main.py` (see the root `SETUP.md` for environment setup)
- All data is synthesized inside the repository, so no internet connection is required.
