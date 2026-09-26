"""Conditionals (if/elif/else) — automatic travel-expense approval judge.

We translate a company expense policy into an if/elif/else ladder and let it
judge automatically. We check the True/False of comparison operators
(==, <=, ...) and logical operators (and/or/not), and prove by experiment
that condition *order* is the priority order of the policy.
"""


def judge_expense(amount, has_receipt):
    """Judge one expense claim by company policy.

    Policy (reading from the top, only the first matching clause applies):
      1) 50,000 KRW or less             -> auto-approved
      2) 150,000 KRW or less + receipt  -> team lead sign-off
      3) 500,000 KRW or less            -> division head approval
      4) anything else                  -> rejected (justification requested)
    """
    if amount <= 50000:                       # clause 1
        return "Auto-approved"
    elif amount <= 150000 and has_receipt:    # clause 2 (and: both must be true)
        return "Team lead sign-off"
    elif amount <= 500000:                    # clause 3
        return "Division head approval"
    else:                                     # clause 4: none of the above
        return "Rejected (justification requested)"


def judge_wrong_order(amount, has_receipt):
    """A deliberately broken policy that checks the *broad* condition first.
    Used to compare why order matters."""
    if amount <= 500000:                      # when the broad condition comes first...
        return "Division head approval"
    elif amount <= 150000 and has_receipt:    # this clause never runs!
        return "Team lead sign-off"
    elif amount <= 50000:                     # neither does this one
        return "Auto-approved"
    else:
        return "Rejected (justification requested)"


def main():
    print("=" * 56)
    print(" Conditionals — automatic travel-expense approval judge")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] Single judgment: run one expense through the policy
    # ---------------------------------------------------------
    print("\n[1] Single judgment")
    amount = 120000          # claimed amount (KRW)
    has_receipt = True       # receipt submitted?

    print(f"  Claimed {amount:,} KRW / receipt {'yes' if has_receipt else 'no'}")
    print(f"  -> Verdict: {judge_expense(amount, has_receipt)}")
    print("  (Over 50k, so clause 1 fails -> 150k or less + receipt, so clause 2 decides; later clauses never checked)")

    # ---------------------------------------------------------
    # [2] Comparison & logical operator lab
    # ---------------------------------------------------------
    print("\n[2] True/False of comparison and logical operators")
    print(f"  amount <= 150000        -> {amount <= 150000}")
    print(f"  amount == 120000        -> {amount == 120000}")
    print(f"  amount != 120000        -> {amount != 120000}")
    print(f"  amount <= 150000 and has_receipt -> {amount <= 150000 and has_receipt}")
    print(f"  amount <= 50000 or has_receipt   -> {amount <= 50000 or has_receipt}")
    print(f"  not has_receipt         -> {not has_receipt}")
    print(f"  50000 < amount <= 150000 (range comparison) -> {50000 < amount <= 150000}")
    grade = "VIP"
    print(f"  grade in ('VIP','VVIP') -> {grade in ('VIP', 'VVIP')}")

    # ---------------------------------------------------------
    # [3] Why condition *order* matters: same clauses, different order
    # ---------------------------------------------------------
    print("\n[3] Change the order and the policy breaks")
    test_amount = 30000      # an amount that should be 'Auto-approved'
    ok = judge_expense(test_amount, True)
    bad = judge_wrong_order(test_amount, True)
    print(f"  30,000 KRW expense, correct order (narrow first) -> {ok}")
    print(f"  30,000 KRW expense, wrong order (broad first)    -> {bad}")
    print("  -> With the broad condition on top, the stricter clauses below never run.")

    # ---------------------------------------------------------
    # [4] Batch judgment: 6 claims through the same policy (a taste of loops)
    # ---------------------------------------------------------
    print("\n[4] Batch judgment of this week's 6 expense claims")

    # (claimant, amount, receipt?)
    requests = [
        ("Kim", 32000, True),
        ("Lee", 120000, True),
        ("Park", 120000, False),   # same amount, no receipt -> different verdict
        ("Choi", 480000, True),
        ("Jung", 750000, True),
        ("Han", 50000, False),
    ]

    approved = 0     # auto-approved count
    escalated = 0    # needs-approval count
    rejected = 0     # rejected count

    for name, amt, receipt in requests:
        decision = judge_expense(amt, receipt)
        print(f"  {name} | {amt:>8,} KRW | receipt {'O' if receipt else 'X'} -> {decision}")
        if decision == "Auto-approved":
            approved += 1
        elif decision == "Rejected (justification requested)":
            rejected += 1
        else:
            escalated += 1

    print(f"\n  Tally: auto-approved {approved} / needs approval {escalated} / rejected {rejected}")
    print("\n[End] Policy clauses = if/elif/else branches. Clause order = condition order.")


if __name__ == "__main__":
    main()
