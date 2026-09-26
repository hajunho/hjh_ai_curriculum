"""
Lecture 08 · Level 01 — 亲手实现感知机
用大约 30 行 numpy 实现一个感知机(1 个人工神经元),
靠"错了就把权重改一点"这条学习规则来解开 AND/OR。
每个轮次输出权重·偏置·误分类个数的变化,
最后演示它在 XOR 上怎么也收敛不了的样子。
"""

import numpy as np

np.random.seed(42)  # 全局可复现 (类内部使用另有种子的 Generator)

# 真值表数据: 四种输入组合与各个问题的标签
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


class Perceptron:
    """感知机: 输入的加权投票(加权和)超过门槛就输出 1, 否则输出 0 的单个神经元。"""

    def __init__(self, n_inputs, lr=0.1, seed=7):
        rng = np.random.default_rng(seed)          # 固定种子 -> 每次运行初值都相同
        self.w = rng.normal(0.0, 0.1, size=n_inputs)  # 话语权(权重)从小随机数起步
        self.b = 0.0                                # 挪动基准线的偏置
        self.lr = lr                                # 学习率: 一次修正的步幅

    def predict(self, x):
        # 加权和 >= 0 就输出 1 (阶跃函数)
        return int(np.dot(self.w, x) + self.b >= 0.0)

    def fit(self, features, targets, max_epochs=20, verbose=True):
        """感知机学习规则。收敛则返回轮次数, 失败则返回 None, 并附上每个轮次的误分类记录。"""
        error_history = []
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for x, y in zip(features, targets):
                pred = self.predict(x)
                update = self.lr * (y - pred)       # 对了是 0, 错了是 +-lr
                if update != 0.0:
                    self.w = self.w + update * x    # 朝着错的反方向调整话语权
                    self.b = self.b + update
                    errors += 1
            error_history.append(errors)
            if verbose:
                print(f"    轮次 {epoch:2d}: w1={self.w[0]:+.3f}, w2={self.w[1]:+.3f}, "
                      f"b={self.b:+.3f}, 误分类 {errors}/4")
            if errors == 0:                          # 整整一圈都没错就算收敛
                return epoch, error_history
        return None, error_history


def show_truth_table(model, features, targets):
    """用整张真值表验证训练完的模型并输出结果。"""
    correct = 0
    for x, y in zip(features, targets):
        pred = model.predict(x)
        mark = "O" if pred == y else "X"
        correct += int(pred == y)
        print(f"      输入 ({int(x[0])}, {int(x[1])}) -> 预测 {pred}, 标签 {y}  [{mark}]")
    print(f"      准确率 {correct}/4 ({correct / 4 * 100:.0f}%)")


def train_and_report(step_no, name, max_epochs=20):
    print(f"[{step_no}] {name} 训练 — 每个轮次观察权重和误分类个数")
    model = Perceptron(n_inputs=2, lr=0.1, seed=7)
    print(f"    初值   : w1={model.w[0]:+.3f}, w2={model.w[1]:+.3f}, b={model.b:+.3f}")
    converged, history = model.fit(X, TARGETS[name], max_epochs=max_epochs)
    if converged is not None:
        print(f"    => 在第 {converged} 个轮次收敛(误分类 0)。最终真值表验证:")
    else:
        print(f"    => {max_epochs} 个轮次内没能收敛。各轮次误分类个数: {history}")
        print("       误分类掉不到 0。权重在一个轮次里被推来推去")
        print("       最后又回到原地, 陷在振荡(循环)里了。最终真值表验证:")
    show_truth_table(model, X, TARGETS[name])
    print()
    return converged


def main():
    print("[1] 感知机介绍")
    print("    1 个神经元 = 加权投票: z = w1*x1 + w2*x2 + b, z >= 0 就输出 1 (阶跃函数)")
    print("    学习规则 = 错了就改: w <- w + lr*(标签-预测)*x, b <- b + lr*(标签-预测)")
    print("    下面用同一个初值(种子 7)依次训练 AND, OR, XOR 三个问题。")
    print()

    train_and_report(2, "AND")
    train_and_report(3, "OR")
    converged = train_and_report(4, "XOR", max_epochs=25)

    print("[5] 结论")
    if converged is None:
        print("    - AND/OR 能用直线分开, 所以感知机在有限个轮次内收敛了。")
        print("    - XOR 用直线分不开(见 level00), 所以权重永远在振荡。")
        print("    - 解法不是训练得更久, 而是改结构, 也就是把神经元叠成层")
        print("      让它自己造表示。这件事的准备物就是下一关的激活函数。")


if __name__ == "__main__":
    main()
