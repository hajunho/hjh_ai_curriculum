"""
hjh_data.py — 课程专用合成数据生成器

本模块为 hjh_ai_curriculum 全部课程"亲手生成"练习用数据。
由于不下载任何外部数据集，完全不存在版权和许可证问题，
即使在没有网络的教室里也能得到完全相同的运行结果。

所有函数都接收 seed 参数，因此每次运行都能复现相同的结果。
这是为了保证讲义中写的数字与实际运行结果始终一致。

作者: 河俊镐 (hajunho) · MIT License
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# numpy 有则用，没有也能以纯 Python 方式运行。
# (lecture01~02 还没学到 numpy，所以这里必须不依赖它。)
try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 0. 通用工具
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    """独立的随机数生成器，不会影响全局 random 的状态。"""
    return random.Random(seed)


def to_csv(rows: list[dict], path: str) -> str:
    """把字典列表保存为 CSV 文件，并返回文件路径。"""
    import csv
    if not rows:
        raise ValueError("不能保存空数据。")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def head(rows: list[dict], n: int = 5) -> None:
    """以表格形式漂亮地打印前几行 (不需要 pandas 也能运行)。"""
    if not rows:
        print("(暂无数据)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows[:n])) for c in cols}
    line = " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows[:n]:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"... 共 {len(rows)} 行")


# ---------------------------------------------------------------------------
# 1. 销售数据 — lecture03, 05, 07
# ---------------------------------------------------------------------------

STORES = ["朝阳店", "海淀店", "浦东店", "天河店", "南山店"]
CATEGORIES = ["咖啡", "烘焙", "三明治", "甜品", "饮品"]


def sales_table(n_days: int = 365, seed: int = 42) -> list[dict]:
    """
    一家虚构连锁咖啡店的每日销售数据。

    我们故意在数据中埋下了以下规律 (课程中会一个个把它们找出来):
      - 周末销售额高于工作日            -> groupby / 星期效应
      - 夏天饮品销售额猛增              -> 季节性 / 时间序列
      - 混入了缺失值和负数异常值        -> 数据清洗
      - 广告费与销售额存在相关关系      -> 回归 / 相关 vs 因果
    """
    rng = _rng(seed)
    rows: list[dict] = []
    for day in range(n_days):
        weekday = day % 7                      # 0=周一 ... 6=周日
        is_weekend = weekday >= 5
        season = math.sin(2 * math.pi * day / 365.0)   # -1(冬) ~ +1(夏)

        for store in STORES:
            store_power = 1.0 + 0.15 * STORES.index(store)
            ad_cost = round(rng.uniform(50_000, 400_000), -3)

            for cat in CATEGORIES:
                base = 300_000 * store_power
                if cat == "饮品":
                    base *= 1.0 + 0.45 * season          # 夏天猛增
                if cat == "咖啡":
                    base *= 1.3
                if is_weekend:
                    base *= 1.25
                noise = rng.gauss(1.0, 0.18)
                revenue = base * noise + ad_cost * 0.35

                # 为数据清洗练习准备的"污染": 1% 缺失、0.5% 负数
                u = rng.random()
                if u < 0.010:
                    revenue_out = None
                elif u < 0.015:
                    revenue_out = -abs(round(revenue))
                else:
                    revenue_out = round(revenue)

                rows.append({
                    "date": f"2025-{day // 31 + 1:02d}-{day % 31 + 1:02d}",
                    "day_index": day,
                    "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][weekday],
                    "store": store,
                    "category": cat,
                    "ad_cost": int(ad_cost),
                    "revenue": revenue_out,
                })
    return rows


# ---------------------------------------------------------------------------
# 2. 客户流失数据 — lecture06, 07
# ---------------------------------------------------------------------------

def churn_table(n: int = 2000, seed: int = 7) -> list[dict]:
    """
    订阅服务的客户流失数据 (用于二分类)。

    真实信号:  使用天数↓、客服来电↑、套餐变更次数↑  -> 流失概率↑
    虚假信号:  customer_id 没有任何含义 (用来练习识别泄漏变量)
    流失率约 18%，很适合做不平衡数据的实战练习。
    """
    rng = _rng(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(1, 60)                       # 入网月数
        monthly_fee = rng.choice([9900, 14900, 19900, 29900])
        usage_days = max(0, min(30, int(rng.gauss(18, 7))))
        support_calls = max(0, int(rng.expovariate(1 / 1.3)))
        plan_changes = max(0, int(rng.expovariate(1 / 0.6)))
        is_auto_pay = rng.random() < 0.7

        # 直接设计对数几率 -> 这份数据是"有标准答案"的
        z = (-1.2
             - 0.05 * usage_days
             + 0.45 * support_calls
             + 0.40 * plan_changes
             - 0.020 * tenure
             + 0.00004 * monthly_fee
             - (0.7 if is_auto_pay else 0.0))
        p = 1 / (1 + math.exp(-z))
        churned = 1 if rng.random() < p else 0

        rows.append({
            "customer_id": f"C{100000 + i}",
            "tenure_months": tenure,
            "monthly_fee": monthly_fee,
            "usage_days_30d": usage_days,
            "support_calls_30d": support_calls,
            "plan_changes": plan_changes,
            "auto_pay": int(is_auto_pay),
            "churned": churned,
        })
    return rows


# ---------------------------------------------------------------------------
# 3. 异常交易数据 — lecture07
# ---------------------------------------------------------------------------

def fraud_table(n: int = 5000, seed: int = 11) -> list[dict]:
    """银行卡异常交易数据。欺诈比例约 1.5%，构造出严重的类别不平衡。"""
    rng = _rng(seed)
    rows = []
    for i in range(n):
        is_fraud = rng.random() < 0.015
        if is_fraud:
            amount = rng.lognormvariate(12.5, 1.1)     # 异常巨大的金额
            hour = rng.choice([0, 1, 2, 3, 4, 23])     # 凌晨时段
            foreign = rng.random() < 0.55
            n_recent = rng.randint(5, 20)              # 短时间内连续刷卡
        else:
            amount = rng.lognormvariate(10.2, 0.9)
            hour = int(max(0, min(23, rng.gauss(14, 4))))
            foreign = rng.random() < 0.05
            n_recent = rng.randint(0, 4)
        rows.append({
            "tx_id": f"T{i:06d}",
            "amount": round(amount),
            "hour": hour,
            "is_foreign": int(foreign),
            "tx_count_1h": n_recent,
            "is_fraud": int(is_fraud),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. 中文文本语料 — lecture10, 11, 12
# ---------------------------------------------------------------------------

REVIEW_POSITIVE = [
    "物流特别快，很满意", "这个价格能买到这种品质，太值了",
    "百分之百会回购", "包装非常用心，好评",
    "比想象中结实很多，很喜欢", "店员态度热情又周到",
    "和图片完全一致，很满意", "店里环境非常舒适",
    "性能远超预期，惊喜", "安装很简单，很省心",
]
REVIEW_NEGATIVE = [
    "物流整整走了一个星期", "这个价格买到这种品质，太亏了",
    "应该不会再买第二次了", "包装破损着就送到了",
    "比想象中脆弱，很快就坏了", "留言咨询了却一直没人回复",
    "颜色和图片完全不一样", "店里又小又吵",
    "性能和描述不符，很失望", "说明书写得不清楚，折腾了半天",
]


def review_corpus(n: int = 600, seed: int = 3) -> list[dict]:
    """用于情感分类的中文评论数据。label 1=好评，0=差评。"""
    rng = _rng(seed)
    fillers = ["", " 不过也有一点小遗憾。", " 下次还会再来。",
               " 供大家参考。", " 评分是我的真实感受。"]
    rows = []
    for i in range(n):
        label = i % 2
        pool = REVIEW_POSITIVE if label == 1 else REVIEW_NEGATIVE
        text = rng.choice(pool) + rng.choice(fillers)
        rows.append({"id": i, "text": text, "label": label})
    rng.shuffle(rows)
    return rows


SAMPLE_DOCS = {
    "公司制度_休假.txt": (
        "第一条 年假以入职日为基准，工作满1年授予15天。"
        "工作满3年以上的员工每满2年增加1天，累计不超过25天。"
        "使用年假须至少提前3个工作日通过审批系统提交申请。"
        "未使用的年假在会计年度结束后折算为补贴发放。"
    ),
    "公司制度_报销.txt": (
        "第二条 出差费用仅限事先获得审批的事项方可报销。"
        "出差补贴标准为国内出差每日3万韩元，海外出差每日8万韩元。"
        "发票须在出差结束后7日内提交，"
        "超过提交期限的费用原则上不予报销。"
    ),
    "公司制度_远程办公.txt": (
        "第三条 远程办公每周最多允许2次，且需团队负责人批准。"
        "远程办公日也须在核心工作时间即上午10点至下午4点保持可联系状态。"
        "将公司资产带出办公场所时，须提前向信息安全部门报备。"
    ),
    "产品手册_安装.txt": (
        "安装本产品之前，请务必先切断电源。"
        "产品与墙面之间至少保留10厘米以上的间距，以保证通风。"
        "初始设置在开机后按照屏幕提示操作，5分钟以内即可完成。"
        "安装后如听到异常噪音，请立即停止使用并联系客服中心。"
    ),
    "产品手册_保修.txt": (
        "产品保修期为自购买之日起2年。"
        "因用户过失造成的损坏、进水、跌落不在保修范围内。"
        "申请保修维修时，需要提供购买发票或订单编号。"
        "耗材不论是否在保修期内，均需付费更换。"
    ),
}


def tiny_corpus(seed: int = 5) -> str:
    """
    用于迷你语言模型预训练的中文文本 (lecture12)。
    语法模式高度重复，让很小的模型也能捕捉到学习信号。
    """
    rng = _rng(seed)
    subjects = ["学生", "上班族", "厨师", "开发者", "老师"]
    objects = ["把报告", "把红烧肉", "把程序", "把信", "把计划书"]
    verbs = ["做好了", "改好了", "检查了一遍", "整理好了", "准备好了"]
    times = ["昨天", "今天", "早上", "晚上", "周末"]
    lines = []
    for _ in range(2000):
        lines.append(f"{rng.choice(times)} {rng.choice(subjects)} "
                     f"{rng.choice(objects)} {rng.choice(verbs)}。")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# 5. 图像数据 — lecture09
# ---------------------------------------------------------------------------

def shape_images(n: int = 800, size: int = 16, seed: int = 13):
    """
    用于图形分类的黑白图像。0=正方形，1=圆形，2=三角形。
    不需要下载 MNIST，我们自己画图就能做 CNN 实战练习。
    需要 numpy。返回值: (X, y) — X 是 (n, size, size) 的 float32 数组。
    """
    if _np is None:
        raise ImportError("shape_images() 需要 numpy。请运行 pip install numpy")
    rng = _np.random.default_rng(seed)
    X = _np.zeros((n, size, size), dtype="float32")
    y = _np.zeros(n, dtype="int64")
    for i in range(n):
        label = i % 3
        y[i] = label
        img = _np.zeros((size, size), dtype="float32")
        cy, cx = rng.integers(5, size - 5, size=2)
        r = int(rng.integers(3, 5))
        yy, xx = _np.mgrid[0:size, 0:size]
        if label == 0:                                    # 正方形
            img[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 1.0
        elif label == 1:                                  # 圆形
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = 1.0
        else:                                             # 三角形
            mask = (yy >= cy - r) & (yy <= cy + r) & (_np.abs(xx - cx) <= (yy - cy + r) / 2)
            img[mask] = 1.0
        img += rng.normal(0, 0.08, img.shape).astype("float32")   # 噪声
        X[i] = _np.clip(img, 0.0, 1.0)
    idx = rng.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# 6. 关系型数据库结构 — lecture04
# ---------------------------------------------------------------------------

def build_sqlite(path: str = "hjh_shop.db", seed: int = 21) -> str:
    """
    创建用于 SQL 实战练习的 SQLite 数据库。
    表: customers, products, orders, order_items, employees
    只使用 Python 标准库，无需安装任何数据库软件。
    """
    import os
    import sqlite3

    if os.path.exists(path):
        os.remove(path)
    rng = _rng(seed)
    con = sqlite3.connect(path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            city TEXT, grade TEXT, joined_at TEXT);
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            category TEXT, price INTEGER, cost INTEGER);
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            dept TEXT, salary INTEGER, manager_id INTEGER);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY, customer_id INTEGER,
            employee_id INTEGER, ordered_at TEXT, status TEXT);
        CREATE TABLE order_items (
            order_id INTEGER, product_id INTEGER, quantity INTEGER,
            PRIMARY KEY (order_id, product_id));
    """)

    cities = ["北京", "上海", "广州", "深圳", "成都", "杭州"]
    grades = ["VIP", "GOLD", "SILVER", "BASIC"]
    surnames = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴"]
    givens = ["志明", "秀英", "晓芳", "子豪", "雨桐", "浩然", "欣怡", "俊杰", "思远", "佳琪"]

    customers = [(i, rng.choice(surnames) + rng.choice(givens), rng.choice(cities),
                  rng.choice(grades), f"202{rng.randint(0, 5)}-{rng.randint(1, 12):02d}-15")
                 for i in range(1, 201)]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    pnames = [("笔记本电脑", "电子"), ("无线鼠标", "电子"), ("机械键盘", "电子"),
              ("显示器", "电子"), ("咖啡豆", "食品"), ("巧克力", "食品"),
              ("保温杯", "生活"), ("记事本", "文具"), ("钢笔", "文具"), ("办公椅", "家具")]
    products = []
    for pid, (nm, cat) in enumerate(pnames, start=1):
        price = rng.choice([4900, 12900, 29000, 89000, 350000, 1200000])
        products.append((pid, nm, cat, price, int(price * rng.uniform(0.45, 0.75))))
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    depts = ["销售", "市场", "研发", "客服"]
    employees = [(1, "林晓敏", "销售", 9000, None)]
    for eid in range(2, 21):
        employees.append((eid, rng.choice(surnames) + rng.choice(givens),
                          rng.choice(depts), rng.randint(3200, 8500),
                          1 if eid <= 5 else rng.randint(2, 5)))
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    orders, items = [], []
    for oid in range(1, 1001):
        orders.append((oid, rng.randint(1, 200), rng.randint(1, 20),
                       f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       rng.choices(["已完成", "已取消", "配送中"], weights=[8, 1, 2])[0]))
        for pid in rng.sample(range(1, 11), rng.randint(1, 3)):
            items.append((oid, pid, rng.randint(1, 5)))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?)", items)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print("=== hjh_data 自检 ===\n")
    print("[1] 销售数据"); head(sales_table(n_days=10), 3)
    print("\n[2] 客户流失数据"); head(churn_table(200), 3)
    print("\n[3] 异常交易数据"); head(fraud_table(500), 3)
    print("\n[4] 评论语料"); head(review_corpus(20), 3)
    print(f"\n[5] 文档 {len(SAMPLE_DOCS)} 份，迷你语料 {len(tiny_corpus()):,} 字")
    print(f"\n[6] 生成 SQLite -> {build_sqlite('/tmp/hjh_check.db')}")
    print("\n所有生成器均正常工作。")
