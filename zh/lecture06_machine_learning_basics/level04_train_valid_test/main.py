"""
level04 — 训练/验证/测试划分: 用实验证明自己打分的乐观偏差

实验 A: 用训练数据打分，成绩被夸大多少 (按树深度看 gap)
实验 B: '偷看'测试集来挑设置，最终汇报成绩被系统性地
        夸大多少 (重复 30 次的统计)
教训: 训练=教科书, 验证=模拟考, 测试=高考(最后 1 次)。
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

K_CANDIDATES = list(range(1, 30, 2))     # 设置(超参数)候选 15 个


def make_data(seed: int, n: int = 1500):
    """混入噪声的二分类合成数据 (无需下载, seed 可复现)。"""
    return make_classification(
        n_samples=n, n_features=8, n_informative=4, n_redundant=2,
        flip_y=0.08,             # 8% 的标签本身是噪声 -> 产生死记硬背的诱惑
        class_sep=0.9, random_state=seed)


def experiment_a() -> None:
    """[2] 实验 A: 训练成绩 vs 测试成绩 — 越灵活差距越大。"""
    X, y = make_data(seed=0, n=1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    print("[2] 实验 A — 自己打分(用训练数据打分)的乐观偏差")
    print("    树深度     训练准确率   测试准确率    差距(gap)")
    for depth in [1, 2, 4, 8, 16, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        acc_tr = tree.score(X_tr, y_tr)
        acc_te = tree.score(X_te, y_te)
        label = "无限制" if depth is None else f"{depth:>4}"
        print(f"    {label:>7}      {acc_tr:6.1%}        {acc_te:6.1%}       {acc_tr - acc_te:+6.1%}")
    print("    -> 越深训练成绩越逼近 100%(背真题)，实战成绩却下滑。")
    print("       训练成绩不是实力的证明。\n")


def experiment_b(n_repeats: int = 30) -> None:
    """[3] 实验 B: 偷看测试集 vs 正确的三分法 — 重复 30 次的统计。
    把'新数据'集(完全没参与模型挑选的数据)视为真实力。"""
    inflate_peek, inflate_proper = [], []
    for rep in range(n_repeats):
        X, y = make_data(seed=100 + rep)
        # 60:20:20 划分 + 另行生成量真实力用的'新数据'
        X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=rep)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=rep)
        X_new, y_new = make_data(seed=9000 + rep, n=800)   # 上线后遇到的数据

        models = {k: KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
                  for k in K_CANDIDATES}

        # (a) 犯规: 看着测试成绩挑 k, 并把那个成绩原样汇报
        k_peek = max(K_CANDIDATES, key=lambda k: models[k].score(X_te, y_te))
        reported_peek = models[k_peek].score(X_te, y_te)
        real_peek = models[k_peek].score(X_new, y_new)
        inflate_peek.append(reported_peek - real_peek)

        # (b) 正规: 用验证集挑 k, 测试集只在最后用 1 次
        k_ok = max(K_CANDIDATES, key=lambda k: models[k].score(X_va, y_va))
        reported_ok = models[k_ok].score(X_te, y_te)
        real_ok = models[k_ok].score(X_new, y_new)
        inflate_proper.append(reported_ok - real_ok)

    peek = np.array(inflate_peek)
    proper = np.array(inflate_proper)
    print(f"[3] 实验 B — 挑选设置 k 候选 {len(K_CANDIDATES)} 个的两种方式, 重复 {n_repeats} 次")
    print("    夸大量 = (汇报的成绩) - (在新数据上的真实成绩)")
    print(f"    (a) 偷看测试集来挑: 平均夸大 {peek.mean():+.2%} (标准差 {peek.std():.2%})")
    print(f"    (b) 用验证集挑(正规): 平均夸大 {proper.mean():+.2%} (标准差 {proper.std():.2%})")
    print(f"    (a) 比 (b) 更夸大的次数: {int((peek > proper).sum())}/{n_repeats} 次")
    print("    -> 偷看造成的夸大不是偶然，是结构性的。")
    print("       候选越多，越容易挑中'在那张考卷上碰巧考得好的设置'。\n")


if __name__ == "__main__":
    np.random.seed(0)

    # [1] 三分法介绍 --------------------------------------------------------
    X, y = make_data(seed=0, n=1000)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=0)
    print("[1] 把 1000 条数据按 60:20:20 做三分")
    print(f"    训练(教科书) {len(X_tr)} 条 / 验证(模拟考) {len(X_va)} 条 / 测试(高考) {len(X_te)} 条\n")

    experiment_a()
    experiment_b()

    print("[4] 总结")
    print("    1. 用训练数据打分永远偏乐观 (实验 A 的 gap)。")
    print("    2. 用测试集挑设置，最终汇报也会被污染 (实验 B)。")
    print("    3. 看到报告里的性能数字，一定要问:")
    print("       \"这个数字是用哪份数据量的? 那份数据你们看过几次?\"")
