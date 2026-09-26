"""
Lecture 08 · Level 01 — パーセプトロンを自作する
numpy 約 30 行でパーセプトロン(人工ニューロン 1 個)を実装し、
「間違えたら重みを少し直す」という学習規則で AND/OR を解いてみます。
エポックごとに重み・バイアス・誤分類数の変化を出力し、
最後に XOR ではどうしても収束できない様子をデモします。
"""

import numpy as np

np.random.seed(42)  # グローバルな再現性 (クラス内部は別シードの Generator を使用)

# 真理値表データ: 入力の 4 通りの組み合わせと問題ごとの正解
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
    """パーセプトロン: 入力の加重投票(加重和)がしきい値を超えれば 1、そうでなければ 0 を出すニューロン 1 個。"""

    def __init__(self, n_inputs, lr=0.1, seed=7):
        rng = np.random.default_rng(seed)          # シード固定 -> 実行するたびに同じ初期値
        self.w = rng.normal(0.0, 0.1, size=n_inputs)  # 発言権(重み)を小さな乱数で開始
        self.b = 0.0                                # 基準線を動かすバイアス
        self.lr = lr                                # 学習率: 一度に直す歩幅

    def predict(self, x):
        # 加重和が 0 以上なら 1 (ステップ関数)
        return int(np.dot(self.w, x) + self.b >= 0.0)

    def fit(self, features, targets, max_epochs=20, verbose=True):
        """パーセプトロンの学習規則。収束すればエポック数を、失敗すれば None とエポック別の誤分類記録を返す。"""
        error_history = []
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for x, y in zip(features, targets):
                pred = self.predict(x)
                update = self.lr * (y - pred)       # 正解なら 0、間違いなら +-lr
                if update != 0.0:
                    self.w = self.w + update * x    # 間違えた方向の逆へ発言権を調整
                    self.b = self.b + update
                    errors += 1
            error_history.append(errors)
            if verbose:
                print(f"    エポック {epoch:2d}: w1={self.w[0]:+.3f}, w2={self.w[1]:+.3f}, "
                      f"b={self.b:+.3f}, 誤分類 {errors}/4")
            if errors == 0:                          # 一周まるごと間違えなければ収束
                return epoch, error_history
        return None, error_history


def show_truth_table(model, features, targets):
    """学習の終わったモデルを真理値表全体で検証して出力します。"""
    correct = 0
    for x, y in zip(features, targets):
        pred = model.predict(x)
        mark = "O" if pred == y else "X"
        correct += int(pred == y)
        print(f"      入力 ({int(x[0])}, {int(x[1])}) -> 予測 {pred}, 正解 {y}  [{mark}]")
    print(f"      精度 {correct}/4 ({correct / 4 * 100:.0f}%)")


def train_and_report(step_no, name, max_epochs=20):
    print(f"[{step_no}] {name} の学習 — エポックごとに重みと誤分類数を観察します")
    model = Perceptron(n_inputs=2, lr=0.1, seed=7)
    print(f"    初期値 : w1={model.w[0]:+.3f}, w2={model.w[1]:+.3f}, b={model.b:+.3f}")
    converged, history = model.fit(X, TARGETS[name], max_epochs=max_epochs)
    if converged is not None:
        print(f"    => {converged}回目のエポックで収束(誤分類 0)。最終的な真理値表の検証:")
    else:
        print(f"    => {max_epochs} エポックの間、収束に失敗。エポック別の誤分類数: {history}")
        print("       誤分類が 0 に落ちません。エポックの中で重みがあちこちへ")
        print("       押されては元の場所へ戻る振動(循環)に囚われています。最終的な真理値表の検証:")
    show_truth_table(model, X, TARGETS[name])
    print()
    return converged


def main():
    print("[1] パーセプトロンの紹介")
    print("    ニューロン 1 個 = 加重投票: z = w1*x1 + w2*x2 + b, z >= 0 なら 1 (ステップ関数)")
    print("    学習規則 = 間違えたら直す: w <- w + lr*(正解-予測)*x, b <- b + lr*(正解-予測)")
    print("    以下では同じ初期値(シード 7)で AND, OR, XOR の 3 問題を学習させます。")
    print()

    train_and_report(2, "AND")
    train_and_report(3, "OR")
    converged = train_and_report(4, "XOR", max_epochs=25)

    print("[5] 結論")
    if converged is None:
        print("    - AND/OR は直線で分けられるので、パーセプトロンが有限のエポック内に収束しました。")
        print("    - XOR は直線で分けられないため(level00 参照)、重みが永遠に振動します。")
        print("    - 解決策はより長く学習することではなく、構造を変えること、つまりニューロンを層に")
        print("      積んで表現を作らせることです。その準備物が次のレベルの活性化関数です。")


if __name__ == "__main__":
    main()
