"""
level02 — 正则表达式

用 re 模块从虚构的公司内部文档里抽取手机号·固话·邮箱·身份证号·金额，
再实践个人信息脱敏和贪婪 (greedy) 匹配的陷阱。
只使用标准库。
"""

import re

# ---------------------------------------------------------------------------
# 练习用的公司文档 (为本课程创作的虚构文本，人名·号码全部是编的)
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "采购确认_邮件.txt": (
        "您好，我是行政部王志明。确认一下 9 月的采购单。"
        "办公椅 20 把，总额 1,200,000 元，发票请开好后发到 "
        "wang.zm@example.com.cn。急事请拨 010-63124567，"
        "或者打我手机 138-1234-5678。"
    ),
    "供应商_通讯录.txt": (
        "韩光物流 对接人 李秀英 187-8765-4321 / lixy@hanguang-logi.cn，"
        "弘文印务 总机 021-58889912，报价咨询发 quote@hongwen-print.com。"
        "夜间紧急调车给 15955551234 发短信即可，"
        "常驻司机张浩然（身份证 110105199003072316）已备案。"
    ),
    "结算_通知.txt": (
        "三季度活动费结算说明：展位租金 350万 元，宣传品制作是 9900元 一只的"
        "环保袋 300 个（合计 2,970,000 元）。凭证缺失请咨询财务部 "
        "fin.help@example.com.cn（内线 010-63129999）。"
    ),
}

# 带名字的模式集合 — 为了好维护，全部集中放在一处。
PATTERNS = {
    "手机号": r"1[3-9]\d(?:-?\d{4}){2}",
    "固定电话": r"0\d{2,3}-?\d{7,8}",
    "邮箱": r"[\w.]+@[\w.-]+\.[a-z]{2,}",
    "身份证": r"\d{6}(?:19|20)\d{6}\d{3}[\dXx]",
    "金额": r"\d{1,3}(?:,\d{3})+\s?元|\d+万\s?元|\d+\s?元",
}


def demo_basics() -> None:
    """[1] 基本零件热身 — 找什么 / 找几个 / 在哪找"""
    print("[1] 语法热身: 每个模式零件到底抓到了什么")
    sample = "订单号 A-2093，数量 15个，负责人 王志明，备注：9月26日出库"
    drills = [
        (r"\d+", "数字块"),
        (r"[一-鿿]+", "汉字块"),
        (r"[A-Z]-\d{4}", "大写字母-4位数字 (订单号格式)"),
        (r"\d+个", "数字+'个' (数量表达)"),
    ]
    print(f"    对象: {sample}")
    for pattern, meaning in drills:
        found = re.findall(pattern, sample)
        print(f"    {pattern:22s} ({meaning:24s}) -> {found}")
    # 中文环境特有的坑: Python 的 \d 连全角数字也一起抓
    fullwidth = "库存１２３件，售价 456 元"
    print(f"    全角陷阱: {fullwidth}")
    print(f"      \\d+     -> {re.findall(r'\d+', fullwidth)}   <- 全角'１２３'也被抓了")
    print(f"      [0-9]+  -> {re.findall(r'[0-9]+', fullwidth)}       <- 只要半角就用 [0-9]")
    print()


def demo_extraction() -> None:
    """[2] 从 3 份文档里一次性抽取联系方式·身份证·金额"""
    print("[2] 信息抽取: 从文档堆里只把联系方式和金额捞出来")
    for name, text in DOCUMENTS.items():
        print(f"    -- {name}")
        for label, pattern in PATTERNS.items():
            found = re.findall(pattern, text)
            print(f"       {label:5s}: {found}")
    print("    -> 以前拿荧光笔一行行划的活，三五次 findall 就干完了。\n")


def mask_phone(text: str) -> str:
    """把手机号中间 4 位换成 ****。用到分组引用 \\1, \\3。"""
    return re.sub(r"(1[3-9]\d-?)(\d{4})(-?\d{4})", r"\1****\3", text)


def mask_id(text: str) -> str:
    """身份证只留前 6 位和后 4 位，中间 8 位打码。"""
    return re.sub(r"(\d{6})(?:19|20)\d{6}(\d{3}[\dXx])", r"\1********\2", text)


def demo_masking() -> None:
    """[3] 个人信息脱敏 — re.sub 与分组引用"""
    print("[3] 个人信息脱敏: 做对外材料时绕不开的一步")
    original = DOCUMENTS["供应商_通讯录.txt"]
    masked = mask_id(mask_phone(original))
    print(f"    原文  : {original[:56]}...")
    print(f"    脱敏后: {masked[:56]}...")
    print(f"    身份证: {re.findall(r'\d{6}\*{8}\d{3}[\dXx]', masked)}")
    n = len(re.findall(r"\*{4}", masked))
    print(f"    -> 共有 {n} 处号码片段被换成了 ****。\n")


def demo_greedy() -> None:
    """[4] 重现贪婪匹配事故 — .* vs .*?"""
    print("[4] 贪婪匹配的陷阱: 星号默认'能抓多长就抓多长'")
    text = "参会人：<王志明> <李秀英> <张浩然>"
    greedy = re.findall(r"<.*>", text)
    lazy = re.findall(r"<.*?>", text)
    print(f"    对象          : {text}")
    print(f"    <.*>  (贪婪)  : {greedy}   <- 整段被当成一个!")
    print(f"    <.*?> (懒惰)  : {lazy}")
    # 替换事故: 本想只删尖括号，结果把名字也一起抹掉
    broken = re.sub(r"<.*>", "", text)
    fixed = re.sub(r"<.*?>", "", text)
    print(f"    删标记(贪婪)  : {broken!r}  <- 数据整块蒸发")
    print(f"    删标记(懒惰)  : {fixed!r}")
    print("    -> 替换后内容离奇消失，十次有九次是贪婪匹配干的。\n")


def demo_summary_table() -> None:
    """[5] 把抽取结果做成 CSV 风格汇总表 — 实务里的最终交付形态"""
    print("[5] 最终整理: 各文档抽取结果汇总表 (可直接粘进 Excel)")
    print("    文档," + ",".join(f"{k}数" for k in PATTERNS))
    for name, text in DOCUMENTS.items():
        counts = [len(re.findall(p, text)) for p in PATTERNS.values()]
        print(f"    {name}," + ",".join(str(c) for c in counts))
    print("\n结论: 正则是'按长相找东西'的技术。按意思找东西，从下一关开始。")


if __name__ == "__main__":
    print("=" * 70)
    print("正则表达式 — 从业务文档中自动抽取信息")
    print("=" * 70 + "\n")
    demo_basics()
    demo_extraction()
    demo_masking()
    demo_greedy()
    demo_summary_table()
