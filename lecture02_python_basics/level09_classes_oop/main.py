"""클래스와 객체지향 — Employee -> Manager 상속 급여 계산.

클래스(붕어빵틀)에서 객체(붕어빵)를 만들고, 속성(self.xxx)과
메서드로 급여를 계산합니다. Manager 는 Employee 를 상속받아
직책·팀원 수당만 추가(재정의)하며, 다형성으로 급여 대장을
한 반복문으로 출력합니다.
"""


# =============================================================
# 붕어빵틀 1: 일반 직원 (부모 클래스)
# =============================================================
class Employee:
    """직원: 이름·기본급·근속연수를 갖고, 급여와 연차를 계산할 수 있다."""

    def __init__(self, name, base_salary, years):
        # 생성자: 객체가 만들어지는 순간 실행되어 '속재료'를 채운다
        self.name = name                  # 속성: 이름
        self.base_salary = base_salary    # 속성: 월 기본급(원)
        self.years = years                # 속성: 근속연수

    def monthly_pay(self):
        """월 급여 = 기본급 + 근속수당(연차당 5만 원)."""
        seniority_bonus = self.years * 50000
        return self.base_salary + seniority_bonus

    def annual_leave(self):
        """연차 일수 = 기본 15일 + 3년마다 1일 (상한 25일)."""
        return min(15 + self.years // 3, 25)

    def __str__(self):
        # print(객체) 할 때 나오는 자기소개 문장
        return f"[직원] {self.name} (근속 {self.years}년)"


# =============================================================
# 붕어빵틀 2: 매니저 (Employee 를 물려받은 개량 틀)
# =============================================================
class Manager(Employee):
    """매니저: 직원의 모든 규칙 + 직책수당 50만 원 + 팀원당 3만 원."""

    ROLE_BONUS = 500000               # 직책수당(클래스 상수)
    PER_MEMBER = 30000                # 팀원 1인당 수당

    def __init__(self, name, base_salary, years, team_size):
        super().__init__(name, base_salary, years)   # 부모의 초기화를 먼저 수행
        self.team_size = team_size                   # 매니저만의 속성 추가

    def monthly_pay(self):
        """재정의(오버라이딩): 부모의 계산 결과에 수당만 얹는다."""
        base = super().monthly_pay()                 # 공통 규칙은 부모에게 위임
        return base + self.ROLE_BONUS + self.team_size * self.PER_MEMBER

    def __str__(self):
        return f"[매니저] {self.name} (근속 {self.years}년, 팀원 {self.team_size}명)"


def main():
    print("=" * 56)
    print(" 클래스와 객체지향 — 급여 계산 시스템")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] 붕어빵 찍기: 같은 틀, 다른 속재료
    # ---------------------------------------------------------
    print("\n[1] 객체 생성 — 같은 클래스, 서로 다른 속성")

    kim = Employee("김주임", 3000000, 3)
    park = Employee("박대리", 3400000, 6)

    print(f"  kim.name  = {kim.name} / kim.base_salary  = {kim.base_salary:,}")
    print(f"  park.name = {park.name} / park.base_salary = {park.base_salary:,}")
    print(f"  같은 틀에서 나왔나? type(kim) is type(park) -> {type(kim) is type(park)}")

    # ---------------------------------------------------------
    # [2] 메서드 호출: self = '자기 데이터로 계산'
    # ---------------------------------------------------------
    print("\n[2] 메서드 — 각자 자기 속재료로 계산한다")
    print(f"  {kim.name} 월 급여  : {kim.monthly_pay():,}원 / 연차 {kim.annual_leave()}일")
    print(f"  {park.name} 월 급여 : {park.monthly_pay():,}원 / 연차 {park.annual_leave()}일")
    print("  -> 같은 monthly_pay() 인데 결과가 다른 이유: self 가 각자 자신이기 때문.")

    # ---------------------------------------------------------
    # [3] 상속: 공통은 부모에, 차이만 자식에
    # ---------------------------------------------------------
    print("\n[3] 상속 — Manager 는 Employee 의 개량 틀")

    lee = Manager("이과장", 4200000, 9, team_size=5)
    base_part = Employee.monthly_pay(lee)          # 부모 규칙만 적용하면?
    full_pay = lee.monthly_pay()                   # 매니저 규칙(재정의) 적용

    print(f"  {lee.name}: 부모 규칙(기본급+근속수당)     = {base_part:,}원")
    print(f"  {lee.name}: + 직책수당 500,000 + 팀원 5x30,000")
    print(f"  {lee.name}: 매니저 규칙 최종 급여          = {full_pay:,}원")
    print(f"  상속받은 메서드도 그대로 사용: 연차 {lee.annual_leave()}일 (부모 것 재사용)")

    # ---------------------------------------------------------
    # [4] 다형성: 직원·매니저 섞인 명단을 한 반복문으로
    # ---------------------------------------------------------
    print("\n[4] 다형성 — 전 직원 급여 대장")

    staff = [
        kim,
        park,
        lee,
        Manager("최부장", 5000000, 15, team_size=12),
        Employee("정사원", 2800000, 1),
    ]

    payroll_total = 0
    for person in staff:
        pay = person.monthly_pay()        # 누구인지 묻지 않는다. 각자 자기 규정대로!
        payroll_total += pay
        role = "매니저" if isinstance(person, Manager) else "직원"
        print(f"  {person.name:4s} ({role:3s}) -> {pay:>9,}원")

    print(f"  이번 달 인건비 총액: {payroll_total:,}원")
    print("  -> if 직급 == ... 분기 없이, 새 직군이 생겨도 이 반복문은 그대로입니다.")

    # ---------------------------------------------------------
    # [5] __str__: 객체의 자기소개
    # ---------------------------------------------------------
    print("\n[5] __str__ — print(객체) 가 읽기 좋은 이유")
    for person in staff[:3]:
        print(f"  {person}")               # __str__ 이 자동 호출된다
    print("  -> __str__ 을 정의하지 않으면 '<...object at 0x...>' 가 나옵니다.")

    print("\n[끝] 클래스 = 데이터(속성) + 절차(메서드)를 한 몸으로.")
    print("     상속 = 공통 규칙은 부모 한 곳에, 차이만 자식에.")


if __name__ == "__main__":
    main()
