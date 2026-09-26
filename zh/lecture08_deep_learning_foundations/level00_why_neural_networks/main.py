"""
Lecture 08 · Level 00 — 神经网络为什么会出现
用最小二乘法和穷举搜索证明: 线性模型能完美解开 AND/OR,
但在 XOR 上绝对越不过 75% 的准确率。
最后手工塞进一个坐标变换(隐藏表示), 让同一个线性模型
把 XOR 做到 100%, 由此引出表示学习这个想法。
"""

import numpy as np

np.random.seed(42)  # 固定随机种子保证可复现 (这一关几乎不依赖随机数, 但按规矩固定)

# 输入: (x1, x2) 四种组合
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

# 标签表: 三种逻辑规则
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def fit_least_squares(features, y):
    """先补上偏置(bias)项, 再用最小二乘法求线性模型的最优系数。"""
    A = np.hstack([features, np.ones((len(features), 1))])  # 最后一列的 1 = 偏置的位置
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # (w1, w2, ..., b)


def linear_accuracy(features, y, coef):
    """按"分数 >= 0.5 判为 1"来分类, 返回准确率和预测结果。"""
    A = np.hstack([features, np.ones((len(features), 1))])
    pred = (A @ coef >= 0.5).astype(int)
    return float((pred == y).mean()), pred


def best_linear_accuracy(features, y, n_w=61, n_b=61, limit=3.0):
    """穷举权重·偏置的网格, 找出线性决策边界能达到的最高准确率。

    分数 = w1*x1 + w2*x2 + b, 分数 >= 0 就判为 1。
    用 numpy 广播一次性评估 (n_w * n_w * n_b) 种组合。
    """
    w_grid = np.linspace(-limit, limit, n_w)
    b_grid = np.linspace(-limit, limit, n_b)
    W1, W2, B = np.meshgrid(w_grid, w_grid, b_grid, indexing="ij")
    # scores 的最后一个轴是 4 条数据: shape = (n_w, n_w, n_b, 4)
    scores = (W1[..., None] * features[:, 0]
              + W2[..., None] * features[:, 1]
              + B[..., None])
    pred = (scores >= 0.0).astype(int)
    acc = (pred == y).mean(axis=-1)
    best_idx = np.unravel_index(np.argmax(acc), acc.shape)
    best = float(acc[best_idx])
    n_tried = acc.size
    return best, n_tried


def hidden_features(features):
    """手工设计两个隐藏表示(hidden representation)。

    h1 像 OR 那样工作(和 >= 0.5 就是 1), h2 像 AND 那样工作(和 >= 1.5 就是 1)。
    这和后面要学的两个隐藏层神经元干的活完全一样。
    """
    s = features[:, 0] + features[:, 1]
    h1 = (s >= 0.5).astype(float)  # 充当 OR
    h2 = (s >= 1.5).astype(float)  # 充当 AND
    return np.stack([h1, h2], axis=1)


def main():
    print("[1] 准备真值表数据 — 输入 (x1, x2) 四种组合与三种规则的标签")
    print("    x1 x2 | AND OR XOR")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} |  {TARGETS['AND'][i]}   {TARGETS['OR'][i]}   {TARGETS['XOR'][i]}")

    print()
    print("[2] 用最小二乘法为每个问题训练'最贴合'的线性模型")
    for name, y in TARGETS.items():
        coef = fit_least_squares(X, y)
        acc, pred = linear_accuracy(X, y, coef)
        print(f"    {name:>3}: w1={coef[0]:+.3f}, w2={coef[1]:+.3f}, b={coef[2]:+.3f}"
              f" -> 准确率 {acc * 100:5.1f}%  (预测 {pred.tolist()}, 标签 {y.tolist()})")
    print("    => AND/OR 是 100%, XOR 靠一条直线没法把四个点全部答对。")

    print()
    print("[3] '会不会藏着更好的直线?' — 穷举权重·偏置的网格")
    for name, y in TARGETS.items():
        best, n_tried = best_linear_accuracy(X, y)
        print(f"    {name:>3}: 试完全部 {n_tried:,} 种 (w1, w2, b) 组合 -> 最高准确率 {best * 100:5.1f}%")
    print("    => XOR 无论挑哪条直线, 75% 就是上限。这不是努力不够, 是结构性的极限。")

    print()
    print("[4] 手工塞进一个坐标变换(隐藏表示)会怎样?")
    H = hidden_features(X)
    print("    把数据搬到新坐标 h1=OR(x1,x2), h2=AND(x1,x2) 上。")
    print("    x1 x2 -> h1 h2 | XOR 标签")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} ->  {int(H[i, 0])}  {int(H[i, 1])} |    {TARGETS['XOR'][i]}")
    coef_h = fit_least_squares(H, TARGETS["XOR"])
    acc_h, pred_h = linear_accuracy(H, TARGETS["XOR"], coef_h)
    best_h, _ = best_linear_accuracy(H, TARGETS["XOR"])
    print(f"    隐藏表示之上的线性模型: w1={coef_h[0]:+.3f}, w2={coef_h[1]:+.3f}, b={coef_h[2]:+.3f}"
          f" -> 准确率 {acc_h * 100:.1f}%")
    print(f"    (穷举搜索也确认最高准确率 {best_h * 100:.1f}% — 实质上就是 y = h1 - h2 一行)")

    print()
    print("[5] 结论")
    print("    - 同一个线性模型, 只换了坐标(表示), XOR 准确率就从 75% -> 100%。")
    print("    - 也就是说, 不是模型太弱, 而是看数据的那套表示太糟。")
    print("    - 把这个坐标变换交给机器而不是人, 让它从数据里学出来 — 这就是")
    print("      表示学习(representation learning), 也就是神经网络与深度学习的出发点。")


if __name__ == "__main__":
    main()
