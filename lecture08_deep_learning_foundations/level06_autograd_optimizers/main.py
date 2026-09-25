"""
PyTorch autograd(자동미분)와 옵티마이저를 배웁니다.
(1) 스칼라 예제로 backward() 가 계산한 기울기를 손계산과 대조하고,
(2) level04 에서 numpy 역전파 ~30줄로 풀었던 XOR 을
    loss.backward() 한 줄 + 옵티마이저로 다시 풀어 코드 축소를 체감하며,
(3) SGD 와 Adam 의 수렴 속도를 같은 조건에서 비교합니다.
"""

import torch
import torch.nn as nn


def make_xor():
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    return X, y


def build_model():
    """level04 와 같은 2-8-1 구조. 층 정의만 하면 미분은 autograd 몫입니다."""
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())


def train_xor(opt_name, lr, max_epochs=2000):
    """XOR 학습 루프. 반환: (loss<0.05 도달 에폭, 최종 loss)"""
    torch.manual_seed(0)                       # 두 옵티마이저에 같은 초기 가중치
    X, y = make_xor()
    model = build_model()
    loss_fn = nn.BCELoss()                     # level04 에서 손으로 짠 BCE 와 동일
    if opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    reached = None
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()                        # (1) 이전 기울기 장부를 백지로
        p = model(X)                           # (2) 순전파
        loss = loss_fn(p, y)                   # (3) 채점
        loss.backward()                        # (4) 역전파 — level04 의 30줄이 이 한 줄
        opt.step()                             # (5) 가중치 업데이트
        acc = ((p > 0.5) == y.bool()).float().mean().item()
        if loss.item() < 0.05 and reached is None:
            reached = epoch                    # '사실상 다 배운' 시점
        if epoch in (1, 100, 500, 1000, max_epochs):
            print(f"      epoch {epoch:4d}: loss={loss.item():.4f}, 정확도={acc:.0%}")
    return reached, loss.item()


def main():
    torch.manual_seed(0)                       # 재현성

    print("[1] autograd 맛보기 — y = x^2 + 3x 를 x=2 에서 미분")
    x = torch.tensor(2.0, requires_grad=True)  # '계산 내역을 기록하라' 스위치 ON
    y = x ** 2 + 3 * x
    y.backward()                               # 장부를 거꾸로 훑어 dy/dx 계산
    print(f"    autograd: dy/dx = {x.grad.item():.1f}")
    print(f"    손계산  : dy/dx = 2x + 3 = 2*2 + 3 = 7.0  -> 일치!\n")

    print("[2] zero_grad 가 필요한 이유 — 기울기는 '덮어쓰기'가 아니라 '누적'")
    x.grad.zero_()                             # 장부 초기화
    (x ** 2 + 3 * x).backward()
    (x ** 2 + 3 * x).backward()                # 초기화 없이 한 번 더
    print(f"    두 번 backward 후 grad = {x.grad.item():.1f} (7 이 아니라 14 = 7+7 누적)")
    print("    => 매 스텝 opt.zero_grad() 로 장부를 비워야 합니다.\n")

    print("[3] XOR 재도전 — level04 (numpy 수제) vs autograd")
    print("    level04: forward 4줄 + backward 15줄 + 검증 20줄 을 손으로 작성")
    print("    이번   : 모델 정의 1줄 + loss.backward() + opt.step() 이 전부\n")

    print("    [3-1] SGD (lr=0.5) — level04 와 같은 방식의 경사하강")
    sgd_epoch, sgd_loss = train_xor("SGD", lr=0.5)

    print("    [3-2] Adam (lr=0.05) — 방향(모멘텀)과 보폭(적응 학습률)을 자동 조절")
    adam_epoch, adam_loss = train_xor("Adam", lr=0.05)

    print("\n[4] 옵티마이저 비교 (같은 초기 가중치, 같은 데이터)")
    print(f"    {'옵티마이저':10s} {'loss<0.05 도달 에폭':>14s} {'최종 loss':>12s}")
    print(f"    {'SGD':10s} {str(sgd_epoch):>16s} {sgd_loss:>12.5f}")
    print(f"    {'Adam':10s} {str(adam_epoch):>15s} {adam_loss:>12.5f}")
    assert sgd_epoch is not None and adam_epoch is not None, "XOR 수렴 실패"
    print("    Adam 은 기울기의 이동평균(방향 관성)과 크기 보정(파라미터별 보폭)을 써서")
    print("    학습률에 덜 민감하게, 보통 더 빨리 수렴합니다. 실무 기본값이 Adam(류)인 이유입니다.\n")

    print("[5] 정리: '미분 자동 기록 장부(autograd)' + '하산 전략(optimizer)' 조합이")
    print("    모든 PyTorch 학습 코드의 5단계 루프(zero_grad -> forward -> loss -> backward -> step)입니다.")


if __name__ == "__main__":
    main()
