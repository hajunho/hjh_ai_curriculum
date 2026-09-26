# Lecture 02 · Level 09 — Classes and Object-Oriented Programming

> A class is the cookie cutter (the blueprint); an object is a cookie stamped out with it (the real thing). Classes bind data (attributes) and procedure (methods) into one body, and inheritance keeps "almost the same, slightly different" rules tidy.
**Difficulty** ⭐⭐⭐ / **Prerequisites** level08 / **Estimated time** 60 min

## 1. Why learn this — the business angle

Say you are building a payroll program. Each employee has a name, base
salary, and years of service (data), plus salary calculation and leave
accrual (procedure). With what we have learned so far, the data goes in
dictionaries and the procedures in separate functions. Grow to five kinds of
employees (regular, manager, contractor...) with slightly different rules,
and the matching mistakes begin: "which function goes with this dictionary?"

A class binds data and procedure into one unit, turning "the very concept of
an employee" into code. And inheritance lets you write **the common rules in
one place and only the differences separately** — "a manager follows all the
employee rules, but adds a role allowance to the salary." The pandas
DataFrame you will use daily and the torch models you will meet later are all
classes, so this concept is also the syntactic foundation of the whole AI
curriculum.

## 2. Understanding through analogies

**The cookie-cutter analogy.** To start a cookie business you first make a
cutter. The cutter itself is not edible. Only when you press it into dough do
you get cookies you can actually eat.

- **Class = the cookie cutter**: the blueprint saying "an employee has a name, base salary, and years of service, and can compute a salary."
- **Object / instance = a cookie**: the real thing stamped from the cutter. Every `Employee("Kim", ...)` produces one more.
- **Attributes = each cookie's own filling**: just as the same cutter yields a chocolate cookie and a jam cookie, each object carries its own name and base salary.
- **Methods = what a cookie can do**: procedures engraved in the cutter. Every cookie has the same procedures, but each runs them **on its own filling**.

**Inheritance is making an upgraded cutter.** You don't melt down the
original — you model a new cutter on it that "adds cheese to the tail." When
the base cutter improves, the upgraded one benefits automatically.

## 3. Core concepts

### 3.1 Defining a class and __init__

```python
class Employee:
    def __init__(self, name, base_salary):   # constructor: runs the moment a cookie is stamped
        self.name = name                      # attribute: this object's filling
        self.base_salary = base_salary

    def monthly_pay(self):                    # method: what this object does
        return self.base_salary

emp = Employee("Kim", 3000000)                # create an object (one from the cutter)
emp.monthly_pay()                             # call a method
```

- `__init__` is the initialization procedure (constructor) that runs
  automatically the moment an object is created.
- `self` is "the very cookie currently performing this action." It is always
  the first parameter of a method, and Python fills it in for you at call
  time. `emp.monthly_pay()` is really shorthand for
  `Employee.monthly_pay(emp)`.
- Convention: class names start uppercase (`Employee`); objects and functions
  are lowercase (`emp`).

### 3.2 Inheritance — common rules in the parent, differences in the child

```python
class Manager(Employee):                      # inherits from Employee
    def __init__(self, name, base_salary, team_size):
        super().__init__(name, base_salary)   # run the parent's initialization first
        self.team_size = team_size            # add a manager-only attribute

    def monthly_pay(self):                    # overriding: same name, different content
        return super().monthly_pay() + 500000 + self.team_size * 30000
```

- `class Manager(Employee)` — the parentheses hold the parent class. A
  manager automatically has every attribute and method of an employee.
- `super()` points to the parent. "Run the parent's procedure first, then add
  mine" is the standard shape.
- Redefine a method with the same name and the child's version wins — this is
  **overriding**.

### 3.3 Polymorphism — one instruction, each their own way

Loop over a mixed list of employees and managers calling
`person.monthly_pay()`, and each object calculates **its own class's way**.
The caller never needs to ask whether it is dealing with an employee or a
manager. That is polymorphism. "Announce 'please compute this month's
salary' to all staff, and each submits a figure per their own policy" —
vastly easier to extend than code that branches on job title with if/elif.

### 3.4 __str__ — an object's self-introduction

To make `print(emp)` produce a human-readable sentence instead of
`<Employee object at 0x...>`, define the `__str__` method. These
double-underscore special methods (dunder methods) are "the contact points
between Python's syntax and your class" — `__init__` is one of them too.

### 3.5 When to use a class

Not all code needs to be a class. The criteria:

- Data and the functions that handle it **always travel together** → class candidate
- **Many** instances of the same structure exist (100 employees, 500 products) → class candidate
- There are "almost the same, slightly different" variants → inheritance candidate
- A procedure that computes once and is done → a function is plenty

## 4. Hands-on — main.py

How to run:

```bash
cd lecture02_python_basics/level09_classes_oop
python3 main.py
```

We build a payroll system with Employee → Manager inheritance.

- **[1] Stamping cookies**: creates two objects from the Employee class and confirms same cutter, different fillings (each their own name and base salary).
- **[2] Calling methods**: calls `monthly_pay()` (base + seniority allowance) and `annual_leave()` (leave accrual). See how `self` makes "each calculates with its own data" possible.
- **[3] Inheritance**: Manager inherits Employee and only adds the role and team allowances. Follow the `super()` call flow in the output.
- **[4] A polymorphic payroll ledger**: one loop over a mixed employee/manager list prints the full payroll. We also check who is a manager with isinstance.
- **[5] __str__**: confirms that the self-introduction method makes `print(employee)` a readable one-liner.

Read the class-definition comments as "rules engraved into the cutter" and
the main-side comments as "the shop floor where cookies are baked and sold."

## 5. Try it yourself

1. **Add an attribute** — add a `department` attribute to Employee and
   include it in the payroll output. (Hint: change three places — the
   `__init__` parameter, the `self.department` assignment, and `__str__` —
   and every object is updated at once.)
2. **Inherit a new role** — create an hourly `PartTimer(Employee)` and
   override `monthly_pay` as "hourly wage x hours worked." Confirm the
   payroll loop needs not a single character changed. (Hint: that is the
   practical payoff of polymorphism.)
3. **Raise simulation** — if you add a `raise_salary(rate)` method to
   Employee that raises base salary by 5%, can Manager use it automatically?
   Experiment. (Hint: think about the direction of inheritance and the answer
   appears.)

## 6. Common mistakes

- **Forgetting self** — define `def monthly_pay():` and the call errors with
  `takes 0 positional arguments but 1 was given`. A method's first parameter
  is always self.
- **Confusing the class with the object** — there is no `Employee.name`. The
  name lives on each cookie (object), so it is `emp.name`. Don't try to eat
  the cutter.
- **Skipping super().__init__** — omit the parent's initialization in the
  child's `__init__` and attributes the parent would have created are
  missing, causing an `AttributeError` later.
- **Making everything a class** — wrapping a one-function job in a class just
  hurts readability. Judge by the criteria in 3.5.
- **Shared attributes in the class body** — putting a mutable value like
  `members = []` directly under `class Employee:` makes every object share
  one list. Per-object data must be created with `self.` inside `__init__`.

## Next level preview

The skeleton of Python syntax is now complete. In level10 we sample the
expressions that say the same things shorter and more Pythonically —
comprehensions, generators, lambda, decorators. This is the level where you
gain the eye to read other people's polished code.
