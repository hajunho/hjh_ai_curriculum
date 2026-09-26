"""
level00 — 规则 vs 学习

用两种方式解同一个分类问题 (银行卡异常交易检测) 并进行比较。
  A. 手工规则: 人凭感觉定下的 IF 语句 ("50 万韩元以上 + 凌晨就是欺诈")
  B. 学习: 把候选阈值全部试一遍，让数据自己选出最优值的搜索 (亲手实现)
核心信息: 传统编程是 "规则+数据->答案"，机器学习是 "数据+答案->规则"。
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

NIGHT_HOURS = {0, 1, 2, 3, 4, 23}          # 凌晨/深夜时段


def score(y_true: list[int], y_pred: list[int]) -> dict:
    """规则的成绩单。欺诈只占 1.4%，单纯的准确率会造成错觉，
    因此以'欺诈命中率(召回率)'与'正常命中率'的平均值(均衡分)为准。"""
    n_fraud = sum(y_true)
    hit_fraud = sum(p == 1 for t, p in zip(y_true, y_pred) if t == 1) / n_fraud
    hit_normal = sum(p == 0 for t, p in zip(y_true, y_pred) if t == 0) / (len(y_true) - n_fraud)
    n_alarm = sum(y_pred)
    precision = (sum(t == 1 for t, p in zip(y_true, y_pred) if p == 1) / n_alarm) if n_alarm else 0.0
    return {"balanced": (hit_fraud + hit_normal) / 2, "recall": hit_fraud,
            "precision": precision, "alarms": n_alarm}


def hand_rule(row: dict) -> int:
    """[手工规则] 会议室里常见的凭感觉的规则。
    '一大笔超过 50 万韩元的钱在凌晨被刷走，多半是欺诈吧。'"""
    return 1 if (row["amount"] >= 500_000 and row["hour"] in NIGHT_HOURS) else 0


def learn_threshold(rows: list[dict], feature: str) -> tuple[float, float]:
    """[学习] 把 '值 >= T 就是欺诈' 这条规则的阈值 T 全部试一遍。
    人只定规则的'形状'，'数字' T 由数据来选。"""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({r[feature] for r in rows}):
        y_pred = [1 if r[feature] >= t else 0 for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def learn_combo(rows: list[dict]) -> tuple[float, float]:
    """[学习扩展] 搜索 '金额 >= T 且是凌晨时段' 中的 T。
    形状和人工规则一样，数字却由数据来定。"""
    y_true = [r["is_fraud"] for r in rows]
    best_t, best_s = None, -1.0
    for t in sorted({round(r["amount"], -4) for r in rows}):    # 以万为单位的候选
        y_pred = [1 if (r["amount"] >= t and r["hour"] in NIGHT_HOURS) else 0
                  for r in rows]
        s = score(y_true, y_pred)["balanced"]
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


def report(name: str, s: dict) -> None:
    print(f"    {name}")
    print(f"      均衡分 {s['balanced']:.1%} / 欺诈命中率 {s['recall']:.1%} / "
          f"警报精确率 {s['precision']:.1%} (警报 {s['alarms']} 笔)")


if __name__ == "__main__":
    # [1] 数据准备 -----------------------------------------------------
    rows = hjh_data.fraud_table(n=5000, seed=11)   # seed 固定 -> 每次数据都相同
    y_true = [r["is_fraud"] for r in rows]
    print("[1] 数据: 银行卡交易", len(rows), "笔 (与垃圾邮件过滤器同构的分类问题)")
    hjh_data.head(rows, 3)
    print(f"    欺诈比例: {sum(y_true) / len(y_true):.1%}  (欺诈=1, 正常=0)\n")

    # [2] 方式 A — 手工规则 ---------------------------------------------
    pred_hand = [hand_rule(r) for r in rows]
    s_hand = score(y_true, pred_hand)
    print("[2] 方式 A — 手工规则 (数字由人凭感觉决定)")
    print("    规则: 金额 >= 500,000 韩元 且 是凌晨时段 就判为欺诈")
    report("成绩:", s_hand)
    print("      -> 抓到的全是真欺诈，却漏掉了 70% 的欺诈。标准定得太高了。\n")

    # [3] 方式 B — 学习: 最优阈值由数据来选 --------------------
    print("[3] 方式 B — 学习 (把阈值 T 的候选全部试一遍，由数据选择)")
    best_feat, best_t, best_s = None, None, -1.0
    for feat in ["amount", "hour", "is_foreign"]:
        t, s = learn_threshold(rows, feat)
        print(f"    特征 {feat:12s}: 最优 T={t:>10,} -> 均衡分 {s:.1%}")
        if s > best_s:
            best_feat, best_t, best_s = feat, t, s
    pred_learn = [1 if r[best_feat] >= best_t else 0 for r in rows]
    s_learn = score(y_true, pred_learn)
    print(f"    => 数据选出的规则: \"{best_feat} >= {best_t:,} 就是欺诈\"")
    report("成绩:", s_learn)
    print("      -> 数据找到了人不知道的数字，但误报较多 (精确率↓)。\n")

    # [4] 方式 B 扩展 — 组合两个特征 -------------------------------------
    t2, s2 = learn_combo(rows)
    pred_two = [1 if (r["amount"] >= t2 and r["hour"] in NIGHT_HOURS) else 0 for r in rows]
    s_combo = score(y_true, pred_two)
    print("[4] 方式 B 扩展 — 形状与人工规则相同，只有数字由数据决定")
    print(f"    数据选出的规则: \"金额 >= {t2:,} 韩元 且 是凌晨 就是欺诈\"")
    report("成绩:", s_combo)
    print("      -> 人的感觉(50 万韩元)和数据的答案(约 2 万韩元)差距就是这么大。\n")

    # [5] 综合比较 -------------------------------------------------------
    print("[5] 综合比较 (均衡分 = 欺诈命中率与正常命中率的平均)")
    print("    方式                          均衡分    欺诈命中率   警报精确率")
    print(f"    A. 手工规则                   {s_hand['balanced']:6.1%}     {s_hand['recall']:6.1%}      {s_hand['precision']:6.1%}")
    print(f"    B. 学习(1 个特征)             {s_learn['balanced']:6.1%}     {s_learn['recall']:6.1%}      {s_learn['precision']:6.1%}")
    print(f"    B. 学习(2 个特征组合)         {s_combo['balanced']:6.1%}     {s_combo['recall']:6.1%}      {s_combo['precision']:6.1%}")
    print()
    print("    核心: 传统编程    规则 + 数据 -> 答案")
    print("          机器学习    数据 + 答案 -> 规则")
    print("    今天的'学习'只是 for 循环的阈值搜索，但神经网络的本质也一样。")
    print("    (注意: 现在是用训练数据打的分，所以成绩偏乐观 -> level04)")
