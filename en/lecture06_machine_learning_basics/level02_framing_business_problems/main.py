"""
level02 — Business question -> ML problem spec: a checklist engine

Write a wish like "we want to reduce churn" as a 5-element spec
(target/unit/timing/metric/action) and this tool automatically checks for
blanks, leakage risk, and ML fitness.
The only level where we build no model — yet the skill you'll use most at work.
"""

from dataclasses import dataclass, field


@dataclass
class Feature:
    """One prediction ingredient (feature). available_at: when this value becomes known."""
    name: str
    available_at: str  # "before prediction time" or "after the outcome is settled"


@dataclass
class MLSpec:
    """ML problem spec — the 5 whiteboard elements plus a baseline."""
    business_goal: str = ""    # the business goal (the wish)
    target: str = ""           # 1. target: a measurable definition
    unit: str = ""             # 2. unit: what one row is
    timing: str = ""           # 3. prediction timing
    metric: str = ""           # 4. evaluation metric
    action: str = ""           # 5. action after the prediction
    baseline: str = ""         # the current approach's score, without a model
    features: list[Feature] = field(default_factory=list)


def check_completeness(spec: MLSpec) -> list[str]:
    """Check that the 5 elements + baseline are filled in."""
    problems = []
    checks = [
        (spec.target, "Target is blank. Not 'churn' but something measurable, like 'cancels within the next 30 days (1/0)'."),
        (spec.unit, "Unit is blank. Decide whether one row is a customer, an account, or an order."),
        (spec.timing, "Timing is blank. Decide when the predict button gets pressed."),
        (spec.metric, "Metric is blank. Pick a metric tied to business profit and loss."),
        (spec.action, "Action is blank. A prediction with no action is a decoration."),
        (spec.baseline, "Baseline is blank. First measure how the current, model-free approach scores."),
    ]
    for value, msg in checks:
        if not value.strip():
            problems.append(msg)
    return problems


def check_leakage(spec: MLSpec) -> list[str]:
    """Leakage check: 'At the moment you press the predict button, can you know this value?'"""
    return [
        f"Suspected leakage: '{f.name}' is created {f.available_at}. "
        f"It does not exist at prediction time ({spec.timing}), so drop it from the ingredients."
        for f in spec.features if f.available_at != "before prediction time"
    ]


def validate(title: str, spec: MLSpec) -> None:
    """Check one spec and print the results."""
    print(f"  ■ Spec: {title}")
    print(f"    Goal: {spec.business_goal}")
    issues = check_completeness(spec) + check_leakage(spec)
    if not issues:
        print(f"    [PASS] All 5 elements present — target: {spec.target} / unit: {spec.unit}")
        print(f"           timing: {spec.timing} / metric: {spec.metric}")
        print(f"           action: {spec.action}")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"    [Issue {i}] {msg}")
    print(f"    Score: {6 + len(spec.features) - len(issues)} / {6 + len(spec.features)}\n")


def judge_ml_fitness() -> None:
    """[4] Automatic 'solvable / unsolvable' classification.
    Criteria: recurring events, enough labeled data, plausible pattern, error tolerance."""
    candidates = [
        ("Tomorrow's sandwich demand per store", dict(repeats=True, labels=3000, pattern=True, error_ok=True)),
        ("Automatic 10% sales-tax calculation", dict(repeats=True, labels=100000, pattern=True, error_ok=False)),
        ("Predicting our company's M&A success", dict(repeats=False, labels=12, pattern=True, error_ok=True)),
        ("Real-time card-fraud detection", dict(repeats=True, labels=50000, pattern=True, error_ok=True)),
    ]
    for name, c in candidates:
        reasons = []
        if not c["repeats"]:
            reasons.append("Not a recurring event (cases never accumulate)")
        if c["labels"] < 500:
            reasons.append(f"Only {c['labels']} labeled cases (not enough to learn from)")
        if not c["error_ok"]:
            reasons.append("Zero error tolerance -> if a clear rule already exists, use the rule (level00)")
        verdict = "worth solving with ML" if not reasons else "not a fit for ML"
        print(f"    {name:42s} -> {verdict}")
        for r in reasons:
            print(f"        Reason: {r}")


if __name__ == "__main__":
    # [1] Bad spec: just the wish, everything else blank ---------------------
    print("[1] Bad spec — submitting 'we want to reduce churn' as-is")
    validate("The wish, verbatim", MLSpec(business_goal="We want to reduce churn"))

    # [2] Good spec: the same wish, fully translated --------------------------
    print("[2] Good spec — the same wish translated into an ML problem")
    good = MLSpec(
        business_goal="We want to reduce subscription churn",
        target="As of end of this month, cancels the subscription within the next 30 days (1/0)",
        unit="One active subscriber (snapshot on the 1st of each month)",
        timing="Morning of the 1st each month, using only data finalized by end of prior month",
        metric="Precision/recall of the top-10%-risk list (not accuracy; see level06)",
        action="CS team gives the top-10%-risk customers a retention call + offer within 1 week",
        baseline="Today: 'text everyone inactive for 3 months' — 2% response rate",
        features=[
            Feature("usage days in last 30 days", "before prediction time"),
            Feature("support calls in last 30 days", "before prediction time"),
            Feature("months since sign-up", "before prediction time"),
        ],
    )
    validate("Churn prediction v1", good)

    # [3] Leakage trap: a column created after the outcome slips into the mix --
    print("[3] Leakage check — a spec with ingredients that peek into the future")
    leaky = MLSpec(
        business_goal="We want to reduce subscription churn",
        target=good.target, unit=good.unit, timing=good.timing,
        metric=good.metric, action=good.action, baseline=good.baseline,
        features=[
            Feature("usage days in last 30 days", "before prediction time"),
            Feature("cancellation penalty charged", "after the outcome is settled"),   # <- only exists once they cancel!
            Feature("exit-survey response", "after the outcome is settled"),
        ],
    )
    validate("Churn prediction v2 (trap)", leaky)
    print("    -> Leaky columns produce perfect test scores and are useless in production.\n")

    # [4] Solvable / unsolvable classification --------------------------------
    print("[4] ML-fitness verdict for 4 candidate problems")
    judge_ml_fitness()
    print()
    print("[5] Summary: translation order = wish -> (target/unit/timing/metric/action) -> leakage check -> baseline")
    print("    Only after passing this checklist is modeling (level03+) worth starting.")
