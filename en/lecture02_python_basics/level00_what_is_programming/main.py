"""What is programming? — a sequence / branch / loop demo.

We translate a cafe's "americano order handling" work manual into a Python
procedure.
[1] Sequence: runs top to bottom, in order
[2] Branch: takes a different path depending on a condition (rewards member?)
[3] Loop: processes several orders with the same procedure
You will see with your own eyes that every program is a combination of these three.
"""


def main():
    print("=" * 52)
    print(" What is programming? — americano order handling demo")
    print("=" * 52)

    # ---------------------------------------------------------
    # [1] Sequence: code runs top to bottom, in order
    # ---------------------------------------------------------
    print("\n[1] Sequence — processing 1 order step by step")

    customer = "Kim"           # data: customer name
    cup_price = 4500           # data: price of one americano (KRW)
    quantity = 2               # data: order quantity

    print(f"  Step 1) Order taken: {customer}, {quantity} americano(s)")

    total = cup_price * quantity          # procedure: calculate the amount
    print(f"  Step 2) Amount: {cup_price} KRW x {quantity} cups = {total} KRW")

    print(f"  Step 3) Receipt printed: {total} KRW paid in full")
    print("  -> These three lines always run in this order. That is 'sequence'.")

    # ---------------------------------------------------------
    # [2] Branch: take a different path depending on a condition
    # ---------------------------------------------------------
    print("\n[2] Branch — the procedure forks on rewards membership")

    is_member = True           # data: rewards member? (try switching True/False)

    if is_member:              # <-- this is the fork in the road
        point = int(total * 0.05)   # members earn 5% in points
        print(f"  Member check O -> {point} points added")
    else:
        print("  Member check X -> no points, straight to the next step")
    print("  -> Splitting on 'if ... then' is a 'branch'.")

    # ---------------------------------------------------------
    # [3] Loop: apply the same procedure to many pieces of data
    # ---------------------------------------------------------
    print("\n[3] Loop — processing 3 waiting orders with the same procedure")

    # A waiting list of 3 orders, each a (name, quantity) pair
    waiting_orders = [
        ("Lee", 1),
        ("Park", 3),
        ("Choi", 2),
    ]

    order_no = 0
    day_total = 0
    for name, qty in waiting_orders:      # <-- repeat the steps below for each order
        order_no = order_no + 1
        amount = cup_price * qty
        day_total = day_total + amount    # accumulate revenue
        print(f"  Order {order_no}) {name}, {qty} cup(s) -> {amount} KRW (running total {day_total} KRW)")

    print(f"  -> Even with 300 orders the code stays the same. That is the power of a 'loop'.")

    # ---------------------------------------------------------
    # [4] Today's takeaway
    # ---------------------------------------------------------
    print("\n[4] Summary")
    print("  Program = data (prices, quantities, names...) + procedure (calculate, decide, print...)")
    print("  And every procedure is a combination of these three:")
    print("    1. Sequence — top to bottom")
    print("    2. Branch — a different path depending on a condition")
    print("    3. Loop — the same job, many times")
    print("  From the next level on, we learn each of the three properly.")


if __name__ == "__main__":
    main()
