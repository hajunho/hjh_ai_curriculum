"""变量与数据类型 — 咖啡店营业额计算示例。

我们亲手创建 int/float/str/bool 四种基本数据类型，
用变量计算营业额，复现一次数据类型转换的失误，
最后用 f-string 格式输出一份人读起来舒服的报表。
"""


def main():
    print("=" * 52)
    print(" 变量与数据类型 — 计算咖啡店一天的营业额")
    print("=" * 52)

    # ---------------------------------------------------------
    # [1] 创建变量: 把值放进贴了名签的盒子里
    # ---------------------------------------------------------
    print("\n[1] 创建变量并确认数据类型")

    menu_name = "美式咖啡"       # str  : 字符串
    unit_price = 4500            # int  : 整数 (以韩元为单位的价格)
    cups_sold = 120              # int  : 卖出的杯数
    vat_rate = 0.1               # float: 增值税率 10%
    is_open = True               # bool : 今天在营业吗

    print(f"  menu_name  = {menu_name!r:14} -> {type(menu_name).__name__}")
    print(f"  unit_price = {unit_price!r:14} -> {type(unit_price).__name__}")
    print(f"  cups_sold  = {cups_sold!r:14} -> {type(cups_sold).__name__}")
    print(f"  vat_rate   = {vat_rate!r:14} -> {type(vat_rate).__name__}")
    print(f"  is_open    = {is_open!r:14} -> {type(is_open).__name__}")

    # ---------------------------------------------------------
    # [2] 计算: '=' 的意思是"把右边的值放进左边的盒子"
    # ---------------------------------------------------------
    print("\n[2] 计算营业额 (顺序执行)")

    revenue = unit_price * cups_sold        # 营业额 = 单价 x 数量
    vat = int(revenue * vat_rate)           # 增值税 (取整到韩元)
    total_with_vat = revenue + vat          # 含税合计

    print(f"  营业额       = {unit_price} x {cups_sold} = {revenue}韩元")
    print(f"  增值税(10%)  = {vat}韩元")
    print(f"  合计         = {total_with_vat}韩元")

    # count = count + 1 的形式: 在当前值上相加后再放回去
    cups_sold = cups_sold + 5               # 打烊前又多卖了 5 杯
    print(f"  追加销售后 cups_sold = {cups_sold} (在原值上加 5 再放回盒子)")

    # ---------------------------------------------------------
    # [3] 数据类型转换: 实际工作中的事故第一名"字符串数字"
    # ---------------------------------------------------------
    print("\n[3] 数据类型转换 — 字符串数字的陷阱")

    typed_price = "4500"        # 从 CSV / 输入里读到的值就是这样的字符串
    typed_qty = "2"

    wrong = typed_price * 2                  # 字符串 * 2 = 拼接!
    print(f"  '4500' * 2          = {wrong!r}  <- 不是 9000，而是字符串重复")

    right = int(typed_price) * int(typed_qty)  # 转换之后再计算
    print(f"  int('4500')*int('2') = {right}   <- 转换后就能正常计算")

    # ---------------------------------------------------------
    # [4] 用 f-string 格式做报表
    # ---------------------------------------------------------
    print("\n[4] f-string 营业额报表")

    target = 600000                              # 今天的目标营业额
    achieve_rate = total_with_vat / target       # 达成率

    print(f"  菜品      : {menu_name}")
    print(f"  总营业额  : {total_with_vat:,}韩元 (含增值税)")   # 千位分隔逗号
    print(f"  目标      : {target:,}韩元")
    print(f"  达成率    : {achieve_rate:.1%}")                  # 百分比，保留 1 位小数
    print(f"  每杯单价  : {unit_price:,.0f}韩元")                # 不带小数的逗号格式

    # ---------------------------------------------------------
    # [5] float 的微小误差
    # ---------------------------------------------------------
    print("\n[5] 确认 float 误差")

    result = 0.1 + 0.2
    print(f"  0.1 + 0.2            = {result}  <- 并不正好是 0.3(二进制存储的局限)")
    print(f"  round(0.1 + 0.2, 2)  = {round(result, 2)}  <- 用四舍五入解决")
    print("  金额计算尽量用整数(以韩元为单位)才更安全。")

    print("\n[完] 变量 = 贴了名签的盒子，数据类型 = 盒子里装的东西的种类。")


if __name__ == "__main__":
    main()
