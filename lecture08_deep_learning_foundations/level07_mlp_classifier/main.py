"""
nn.Module 로 나만의 MLP(다층 퍼셉트론) 분류 모델을 만듭니다.
구독 이탈 데이터(churn_table)를 학습/테스트로 나눠
(1) 로지스틱 회귀(= 은닉층 0개 신경망)와 (2) MLP(6-16-8-1)를
같은 조건에서 torch 로 학습해 정확도·정밀도·재현율을 비교합니다.
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


def load_data():
    """churn_table 을 표준화된 텐서로 변환합니다. customer_id 는 의미 없는 열이라 제외."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)

    # 학습/테스트 분리 (섞은 뒤 75:25)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.75)
    tr, te = idx[:cut], idx[cut:]

    # 표준화: 평균/표준편차는 반드시 '학습 데이터로만' 계산 (누설 방지)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    X = (X - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(y[tr]), t(X[te]), t(y[te])


class LogisticRegression(nn.Module):
    """은닉층 0개짜리 신경망 = 로지스틱 회귀 (lecture06 level05 의 그 모델)."""

    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class ChurnMLP(nn.Module):
    """은닉층 2개(16, 8) MLP. __init__ 에서 부품 선언, forward 에서 조립."""

    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train(model, X_tr, y_tr, n_epochs=800, lr=0.01):
    """full-batch Adam 학습. 소형 데이터라 배치 분할은 다음 레벨에서."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        if epoch in (1, 200, 800):
            print(f"      epoch {epoch:3d}: train loss = {loss.item():.4f}")
    return model


def evaluate(model, X_te, y_te, threshold=0.5):
    """테스트 성능: 정확도 + 이탈(1) 클래스의 정밀도/재현율."""
    model.eval()
    with torch.no_grad():                       # 평가 시엔 미분 장부 기록 OFF
        p = model(X_te)
    pred = (p > threshold).float()
    acc = (pred == y_te).float().mean().item()
    tp = ((pred == 1) & (y_te == 1)).sum().item()
    fp = ((pred == 1) & (y_te == 0)).sum().item()
    fn = ((pred == 0) & (y_te == 1)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return acc, prec, rec


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("[1] 데이터: 구독 이탈 churn_table (n=2000, 이탈률 약 18%)")
    X_tr, y_tr, X_te, y_te = load_data()
    print(f"    학습 {len(X_tr)}건 / 테스트 {len(X_te)}건, 특징 {X_tr.shape[1]}개 (customer_id 제외, 표준화 완료)")
    print(f"    테스트 이탈률: {y_te.mean().item():.1%} -> '전원 유지' 라고만 찍어도 정확도 ~{1-y_te.mean().item():.0%}\n")

    print("[2] 모델 A — 로지스틱 회귀 (은닉층 0개 신경망)")
    logreg = LogisticRegression(len(FEATURES))
    print(f"    파라미터 수: {count_params(logreg)}")
    train(logreg, X_tr, y_tr)

    print("\n[3] 모델 B — MLP (6-16-8-1, ReLU)")
    mlp = ChurnMLP(len(FEATURES))
    print(f"    파라미터 수: {count_params(mlp)}")
    train(mlp, X_tr, y_tr)

    print("\n[4] 테스트 성능 비교 (임계값 0.5, 이탈=양성)")
    print(f"    {'모델':16s} {'정확도':>8s} {'정밀도':>8s} {'재현율':>8s}")
    results = {}
    for name, model in [("LogisticReg", logreg), ("MLP", mlp)]:
        acc, prec, rec = evaluate(model, X_te, y_te)
        results[name] = acc
        print(f"    {name:16s} {acc:8.3f} {prec:8.3f} {rec:8.3f}")

    print("\n[5] 임계값을 0.5 -> 0.3 으로 낮추면? (MLP)")
    acc3, prec3, rec3 = evaluate(mlp, X_te, y_te, threshold=0.3)
    print(f"    {'MLP(th=0.3)':16s} {acc3:8.3f} {prec3:8.3f} {rec3:8.3f}")
    print("    정확도를 조금 내주는 대신 '놓치는 이탈 고객'(재현율)을 크게 줄입니다.")

    print("\n[6] 해석")
    print("    - 이 데이터는 이탈 규칙이 '거의 선형'으로 설계돼 있어 두 모델이 비슷합니다.")
    print("    - 교훈 1: 딥러닝이 항상 이기는 게 아닙니다. 표 형태 데이터는 단순 모델부터.")
    print("    - 교훈 2: 그래도 MLP 는 특징 간 상호작용(예: 요금제 변경 x 문의 급증)을")
    print("      자동으로 학습할 여지가 있어, 관계가 비선형일수록 격차가 벌어집니다.")
    print("    - 교훈 3: 재현율(놓친 이탈 고객)이 낮다면 임계값 0.5 를 내리는 것도 실무 선택지입니다.")


if __name__ == "__main__":
    main()
