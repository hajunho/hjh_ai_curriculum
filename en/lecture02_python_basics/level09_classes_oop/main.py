"""Classes and OOP — payroll with Employee -> Manager inheritance.

We stamp objects (cookies) from a class (the cookie cutter) and compute
salaries via attributes (self.xxx) and methods. Manager inherits from
Employee and only adds (overrides with) the role and team allowances,
and polymorphism prints the payroll ledger with a single loop.
"""


# =============================================================
# Cookie cutter 1: the regular employee (parent class)
# =============================================================
class Employee:
    """An employee: has a name, base salary, and years of service, and can compute salary and leave."""

    def __init__(self, name, base_salary, years):
        # Constructor: runs the moment the object is created, filling in the 'filling'
        self.name = name                  # attribute: name
        self.base_salary = base_salary    # attribute: monthly base salary (KRW)
        self.years = years                # attribute: years of service

    def monthly_pay(self):
        """Monthly pay = base salary + seniority allowance (50,000 KRW per year served)."""
        seniority_bonus = self.years * 50000
        return self.base_salary + seniority_bonus

    def annual_leave(self):
        """Annual leave days = base 15 + 1 per 3 years served (capped at 25)."""
        return min(15 + self.years // 3, 25)

    def __str__(self):
        # The self-introduction sentence produced by print(object)
        return f"[Employee] {self.name} ({self.years} yrs of service)"


# =============================================================
# Cookie cutter 2: the manager (an upgraded cutter inheriting Employee)
# =============================================================
class Manager(Employee):
    """A manager: all employee rules + 500,000 KRW role allowance + 30,000 KRW per team member."""

    ROLE_BONUS = 500000               # role allowance (class constant)
    PER_MEMBER = 30000                # allowance per team member

    def __init__(self, name, base_salary, years, team_size):
        super().__init__(name, base_salary, years)   # run the parent's initialization first
        self.team_size = team_size                   # manager-only attribute

    def monthly_pay(self):
        """Override: take the parent's result and add the allowances on top."""
        base = super().monthly_pay()                 # delegate the common rule to the parent
        return base + self.ROLE_BONUS + self.team_size * self.PER_MEMBER

    def __str__(self):
        return f"[Manager] {self.name} ({self.years} yrs of service, team of {self.team_size})"


def main():
    print("=" * 56)
    print(" Classes and OOP — a payroll system")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] Stamping cookies: same cutter, different fillings
    # ---------------------------------------------------------
    print("\n[1] Creating objects — same class, different attributes")

    kim = Employee("Kim", 3000000, 3)
    park = Employee("Park", 3400000, 6)

    print(f"  kim.name  = {kim.name} / kim.base_salary  = {kim.base_salary:,}")
    print(f"  park.name = {park.name} / park.base_salary = {park.base_salary:,}")
    print(f"  From the same cutter? type(kim) is type(park) -> {type(kim) is type(park)}")

    # ---------------------------------------------------------
    # [2] Calling methods: self = 'compute with your own data'
    # ---------------------------------------------------------
    print("\n[2] Methods — each calculates with its own filling")
    print(f"  {kim.name}'s monthly pay  : {kim.monthly_pay():,} KRW / leave {kim.annual_leave()} days")
    print(f"  {park.name}'s monthly pay : {park.monthly_pay():,} KRW / leave {park.annual_leave()} days")
    print("  -> Same monthly_pay(), different results: because self is each object itself.")

    # ---------------------------------------------------------
    # [3] Inheritance: common rules in the parent, differences in the child
    # ---------------------------------------------------------
    print("\n[3] Inheritance — Manager is Employee's upgraded cutter")

    lee = Manager("Lee", 4200000, 9, team_size=5)
    base_part = Employee.monthly_pay(lee)          # applying only the parent's rule?
    full_pay = lee.monthly_pay()                   # applying the manager rule (override)

    print(f"  {lee.name}: parent rule (base + seniority)      = {base_part:,} KRW")
    print(f"  {lee.name}: + role allowance 500,000 + team 5x30,000")
    print(f"  {lee.name}: final pay under the manager rule    = {full_pay:,} KRW")
    print(f"  Inherited methods still work: leave {lee.annual_leave()} days (reusing the parent's)")

    # ---------------------------------------------------------
    # [4] Polymorphism: one loop over a mixed employee/manager roster
    # ---------------------------------------------------------
    print("\n[4] Polymorphism — the full payroll ledger")

    staff = [
        kim,
        park,
        lee,
        Manager("Choi", 5000000, 15, team_size=12),
        Employee("Jung", 2800000, 1),
    ]

    payroll_total = 0
    for person in staff:
        pay = person.monthly_pay()        # never ask who they are. Each follows their own policy!
        payroll_total += pay
        role = "Manager" if isinstance(person, Manager) else "Employee"
        print(f"  {person.name:4s} ({role:8s}) -> {pay:>9,} KRW")

    print(f"  Total payroll this month: {payroll_total:,} KRW")
    print("  -> No 'if title == ...' branching, and a new role never touches this loop.")

    # ---------------------------------------------------------
    # [5] __str__: the object's self-introduction
    # ---------------------------------------------------------
    print("\n[5] __str__ — why print(object) reads nicely")
    for person in staff[:3]:
        print(f"  {person}")               # __str__ is called automatically
    print("  -> Without __str__ you would see '<...object at 0x...>'.")

    print("\n[End] Class = data (attributes) + procedure (methods) in one body.")
    print("      Inheritance = common rules in one parent, only the differences in the child.")


if __name__ == "__main__":
    main()
