"""
Lecture 08 · Level 01 — 퍼셉트론 직접 만들기
numpy 약 30줄로 퍼셉트론(인공 뉴런 1개)을 구현하고,
"틀리면 가중치를 조금 고친다"는 학습 규칙으로 AND/OR 를 풀어 봅니다.
에폭마다 가중치·편향·오분류 수의 변화를 출력하고,
마지막으로 XOR 에서는 끝내 수렴하지 못하는 모습을 시연합니다.
"""

import numpy as np

np.random.seed(42)  # 전역 재현성 (클래스 내부는 별도 시드의 Generator 사용)

# 진리표 데이터: 입력 네 가지 조합과 문제별 정답
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
    """퍼셉트론: 입력의 가중 투표(가중합)가 문턱을 넘으면 1, 아니면 0을 내는 뉴런 1개."""

    def __init__(self, n_inputs, lr=0.1, seed=7):
        rng = np.random.default_rng(seed)          # 시드 고정 -> 실행할 때마다 같은 초기값
        self.w = rng.normal(0.0, 0.1, size=n_inputs)  # 발언권(가중치)을 작은 난수로 시작
        self.b = 0.0                                # 기준선을 움직이는 편향
        self.lr = lr                                # 학습률: 한 번에 고치는 보폭

    def predict(self, x):
        # 가중합이 0 이상이면 1 (계단 함수)
        return int(np.dot(self.w, x) + self.b >= 0.0)

    def fit(self, features, targets, max_epochs=20, verbose=True):
        """퍼셉트론 학습 규칙. 수렴하면 에폭 수, 실패하면 None 과 에폭별 오분류 기록을 반환."""
        error_history = []
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for x, y in zip(features, targets):
                pred = self.predict(x)
                update = self.lr * (y - pred)       # 맞으면 0, 틀리면 +-lr
                if update != 0.0:
                    self.w = self.w + update * x    # 틀린 방향의 반대로 발언권 조정
                    self.b = self.b + update
                    errors += 1
            error_history.append(errors)
            if verbose:
                print(f"    에폭 {epoch:2d}: w1={self.w[0]:+.3f}, w2={self.w[1]:+.3f}, "
                      f"b={self.b:+.3f}, 오분류 {errors}/4")
            if errors == 0:                          # 한 바퀴 내내 안 틀리면 수렴
                return epoch, error_history
        return None, error_history


def show_truth_table(model, features, targets):
    """학습이 끝난 모델을 진리표 전체로 검증해 출력합니다."""
    correct = 0
    for x, y in zip(features, targets):
        pred = model.predict(x)
        mark = "O" if pred == y else "X"
        correct += int(pred == y)
        print(f"      입력 ({int(x[0])}, {int(x[1])}) -> 예측 {pred}, 정답 {y}  [{mark}]")
    print(f"      정확도 {correct}/4 ({correct / 4 * 100:.0f}%)")


def train_and_report(step_no, name, max_epochs=20):
    print(f"[{step_no}] {name} 학습 — 에폭마다 가중치와 오분류 수를 관찰합니다")
    model = Perceptron(n_inputs=2, lr=0.1, seed=7)
    print(f"    초기값 : w1={model.w[0]:+.3f}, w2={model.w[1]:+.3f}, b={model.b:+.3f}")
    converged, history = model.fit(X, TARGETS[name], max_epochs=max_epochs)
    if converged is not None:
        print(f"    => {converged}번째 에폭에서 수렴(오분류 0). 최종 진리표 검증:")
    else:
        print(f"    => {max_epochs} 에폭 동안 수렴 실패. 에폭별 오분류 수: {history}")
        print("       오분류가 0으로 떨어지지 않습니다. 에폭 안에서 가중치가 이리저리")
        print("       밀리다가 제자리로 돌아오는 진동(순환)에 갇혔습니다. 최종 진리표 검증:")
    show_truth_table(model, X, TARGETS[name])
    print()
    return converged


def main():
    print("[1] 퍼셉트론 소개")
    print("    뉴런 1개 = 가중 투표: z = w1*x1 + w2*x2 + b, z >= 0 이면 1 (계단 함수)")
    print("    학습 규칙 = 틀리면 고친다: w <- w + lr*(정답-예측)*x, b <- b + lr*(정답-예측)")
    print("    아래에서 같은 초기값(시드 7)으로 AND, OR, XOR 세 문제를 학습시킵니다.")
    print()

    train_and_report(2, "AND")
    train_and_report(3, "OR")
    converged = train_and_report(4, "XOR", max_epochs=25)

    print("[5] 결론")
    if converged is None:
        print("    - AND/OR 는 직선으로 나눌 수 있어 퍼셉트론이 유한한 에폭 안에 수렴했습니다.")
        print("    - XOR 은 직선으로 나눌 수 없어(level00 참고) 가중치가 영원히 진동합니다.")
        print("    - 해결책은 더 오래 학습하는 것이 아니라 구조를 바꾸는 것, 즉 뉴런을 층으로")
        print("      쌓아 표현을 만들게 하는 것입니다. 그 준비물이 다음 레벨의 활성화 함수입니다.")


if __name__ == "__main__":
    main()
