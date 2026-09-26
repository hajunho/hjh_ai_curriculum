"""
A simulation that pushes a data project through the five stages
(problem framing -> data -> model -> deployment -> monitoring)
using gate checklists.
With a fictional 'subscription churn defense' project as the example,
watch how a stage whose deliverables are blank gets a STOP verdict at the gate.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def gate(stage_name: str, checklist: dict) -> bool:
    """Fail the gate if even one checklist value is blank."""
    print(f"\n  [{stage_name}] gate check")
    ok = True
    for item, value in checklist.items():
        filled = value not in ("", None, [])
        mark = "OK     " if filled else "MISSING"
        shown = value if filled else "(blank)"
        print(f"    - {mark} | {item}: {shown}")
        if not filled:
            ok = False
    print(f"    => Verdict: {'PASS - proceed to next stage' if ok else 'STOP - do not proceed until filled in'}")
    return ok


def main() -> None:
    print("=" * 62)
    print(" Data project life-cycle simulation: the 'subscription churn defense' project")
    print("=" * 62)
    results = {}

    # ------------------------------------------------------------------
    print("\n[1] Problem framing — what, why, and what counts as success?")
    problem_spec = {
        "Prediction target (label definition)": "customer who does not renew next month's payment = churned (1)",
        "How predictions will be used": "every Monday, offer outreach/coupons to the top 200 at-risk customers",
        "Target metric (a number)": "monthly churn rate 18% -> 15% within 6 months",
        "Baseline (current practice)": "send coupons to 100 customers chosen at random",
    }
    results["1.Problem framing"] = gate("Problem framing", problem_spec)

    # ------------------------------------------------------------------
    print("\n[2] Data — do we have data that can answer the question? (running a real audit)")
    rows = hjh_data.churn_table(n=2000, seed=7)
    n_total = len(rows)
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    n_churn = sum(r["churned"] for r in rows)
    churn_rate = n_churn / n_total
    min_minority = 200  # gate threshold: minimum minority-class (churn) samples

    print(f"    rows: {n_total} / rows with missing values: {n_missing}")
    print(f"    churned customers: {n_churn} (churn rate {churn_rate:.1%})")
    data_audit = {
        "Sample size check": f"{n_total} rows",
        "Missing-value check": f"{n_missing} rows (within tolerance)",
        "Minority-class sample size": f"{n_churn} customers" if n_churn >= min_minority else "",
        "Label definition alignment": "based on payment logs, agreed with the CS team",
    }
    results["2.Data"] = gate("Data", data_audit)

    # ------------------------------------------------------------------
    print("\n[3] Model — first compute the baseline the model must beat")
    # Accuracy of a baseline that predicts only the majority class (no churn), with no model at all
    majority_acc = 1 - churn_rate
    print(f"    majority-vote baseline accuracy: {majority_acc:.1%}  <- you get this much just by saying 'nobody churns'")
    print("    => Any model we build later must beat this baseline not on 'accuracy'")
    print("       but on recall/precision — actually finding the churners (continued in level01, level07).")
    model_report = {
        "Baseline performance recorded": f"majority-vote accuracy {majority_acc:.1%}",
        "Validation method agreed": "5-fold cross-validation (learned in level06)",
        "Model performance report": "(to be written in later levels)",  # treated as filled for this demo
    }
    results["3.Model"] = gate("Model", model_report)

    # ------------------------------------------------------------------
    print("\n[4] Deployment — can the business actually use it?")
    deploy_plan = {
        "Delivery channel for results": "upload the at-risk list to the CRM system every Monday",
        "Recipients and business procedure": "",  # left blank on purpose: to observe the STOP verdict
        "Failure response (if the model stops working)": "",
    }
    results["4.Deployment"] = gate("Deployment", deploy_plan)

    # ------------------------------------------------------------------
    print("\n[5] Monitoring — is performance holding up?")
    monitor_plan = {
        "Performance tracking metrics": "weekly recall/precision, actual churn rate after each campaign",
        "Data drift surveillance": "",  # left blank on purpose
        "Retraining criterion": "retrain if recall drops 5+ pp for two consecutive weeks",
    }
    results["5.Monitoring"] = gate("Monitoring", monitor_plan)

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("[6] Final summary — where must this project stop right now?")
    print("=" * 62)
    first_stop = None
    for stage, ok in results.items():
        print(f"    {stage:<20} : {'PASS' if ok else 'STOP'}")
        if not ok and first_stop is None:
            first_stop = stage
    if first_stop:
        print(f"\n    => First STOP point: {first_stop}")
        print("       Filling in this gate's blanks matters more than squeezing out more model performance.")
    print("\n    Lesson: failing projects collapse not because of code, but because they 'march forward carrying blanks'.")


if __name__ == "__main__":
    main()
