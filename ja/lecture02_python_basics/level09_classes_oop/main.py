"""クラスとオブジェクト指向 — Employee -> Manager 継承の給与計算。

クラス (たい焼きの型) からオブジェクト (たい焼き) を作り、属性 (self.xxx) と
メソッドで給与を計算します。Manager は Employee を継承して
役職・チーム人数の手当だけを追加 (再定義) し、ポリモーフィズムで給与台帳を
1つの繰り返しで出力します。
"""


# =============================================================
# たい焼きの型 1: 一般社員 (親クラス)
# =============================================================
class Employee:
    """社員: 名前・基本給・勤続年数を持ち、給与と年休を計算できる。"""

    def __init__(self, name, base_salary, years):
        # コンストラクタ: オブジェクトが作られる瞬間に実行され、「中身」を詰める
        self.name = name                  # 属性: 名前
        self.base_salary = base_salary    # 属性: 月の基本給 (ウォン)
        self.years = years                # 属性: 勤続年数

    def monthly_pay(self):
        """月給 = 基本給 + 勤続手当 (勤続1年あたり5万ウォン)。"""
        seniority_bonus = self.years * 50000
        return self.base_salary + seniority_bonus

    def annual_leave(self):
        """年休日数 = 基本15日 + 3年ごとに1日 (上限25日)。"""
        return min(15 + self.years // 3, 25)

    def __str__(self):
        # print(オブジェクト) したときに出る自己紹介の文章
        return f"[社員] {self.name} (勤続 {self.years}年)"


# =============================================================
# たい焼きの型 2: マネージャー (Employee を受け継いだ改良型)
# =============================================================
class Manager(Employee):
    """マネージャー: 社員のすべてのルール + 役職手当50万ウォン + チーム1人あたり3万ウォン。"""

    ROLE_BONUS = 500000               # 役職手当 (クラス定数)
    PER_MEMBER = 30000                # チームメンバー1人あたりの手当

    def __init__(self, name, base_salary, years, team_size):
        super().__init__(name, base_salary, years)   # 親の初期化を先に実行
        self.team_size = team_size                   # マネージャーだけの属性を追加

    def monthly_pay(self):
        """再定義 (オーバーライド): 親の計算結果に手当を上乗せするだけ。"""
        base = super().monthly_pay()                 # 共通ルールは親に委任
        return base + self.ROLE_BONUS + self.team_size * self.PER_MEMBER

    def __str__(self):
        return f"[マネージャー] {self.name} (勤続 {self.years}年, チーム {self.team_size}人)"


def main():
    print("=" * 56)
    print(" クラスとオブジェクト指向 — 給与計算システム")
    print("=" * 56)

    # ---------------------------------------------------------
    # [1] たい焼きを焼く: 同じ型、違う中身
    # ---------------------------------------------------------
    print("\n[1] オブジェクト生成 — 同じクラス、互いに異なる属性")

    kim = Employee("田中主任", 3000000, 3)
    park = Employee("鈴木係長", 3400000, 6)

    print(f"  kim.name  = {kim.name} / kim.base_salary  = {kim.base_salary:,}")
    print(f"  park.name = {park.name} / park.base_salary = {park.base_salary:,}")
    print(f"  同じ型から出てきた? type(kim) is type(park) -> {type(kim) is type(park)}")

    # ---------------------------------------------------------
    # [2] メソッド呼び出し: self = 「自分のデータで計算」
    # ---------------------------------------------------------
    print("\n[2] メソッド — 各自が自分の中身で計算する")
    print(f"  {kim.name} の月給  : {kim.monthly_pay():,}ウォン / 年休 {kim.annual_leave()}日")
    print(f"  {park.name} の月給 : {park.monthly_pay():,}ウォン / 年休 {park.annual_leave()}日")
    print("  -> 同じ monthly_pay() なのに結果が違う理由: self がそれぞれ自分自身だから。")

    # ---------------------------------------------------------
    # [3] 継承: 共通は親に、差分だけ子に
    # ---------------------------------------------------------
    print("\n[3] 継承 — Manager は Employee の改良型")

    lee = Manager("佐藤課長", 4200000, 9, team_size=5)
    base_part = Employee.monthly_pay(lee)          # 親のルールだけを適用すると?
    full_pay = lee.monthly_pay()                   # マネージャーのルール (再定義) を適用

    print(f"  {lee.name}: 親のルール (基本給+勤続手当)     = {base_part:,}ウォン")
    print(f"  {lee.name}: + 役職手当 500,000 + チーム 5x30,000")
    print(f"  {lee.name}: マネージャールールの最終給与     = {full_pay:,}ウォン")
    print(f"  継承したメソッドもそのまま使える: 年休 {lee.annual_leave()}日 (親のものを再利用)")

    # ---------------------------------------------------------
    # [4] ポリモーフィズム: 社員・マネージャー混在の名簿を1つの繰り返しで
    # ---------------------------------------------------------
    print("\n[4] ポリモーフィズム — 全社員の給与台帳")

    staff = [
        kim,
        park,
        lee,
        Manager("高橋部長", 5000000, 15, team_size=12),
        Employee("伊藤社員", 2800000, 1),
    ]

    payroll_total = 0
    for person in staff:
        pay = person.monthly_pay()        # 相手が誰かは尋ねない。各自が自分の規定どおりに!
        payroll_total += pay
        role = "マネージャー" if isinstance(person, Manager) else "社員"
        print(f"  {person.name:6s} ({role}) -> {pay:>9,}ウォン")

    print(f"  今月の人件費総額: {payroll_total:,}ウォン")
    print("  -> if 役職 == ... の分岐なしで、新しい職種が増えてもこの繰り返しはそのままです。")

    # ---------------------------------------------------------
    # [5] __str__: オブジェクトの自己紹介
    # ---------------------------------------------------------
    print("\n[5] __str__ — print(オブジェクト) が読みやすい理由")
    for person in staff[:3]:
        print(f"  {person}")               # __str__ が自動的に呼ばれる
    print("  -> __str__ を定義しないと '<...object at 0x...>' が出てきます。")

    print("\n[終] クラス = データ (属性) + 手順 (メソッド) を一体に。")
    print("     継承 = 共通ルールは親の1か所に、差分だけを子に。")


if __name__ == "__main__":
    main()
