"""
학습률 스케줄과 학습 안정화 기법을 실험합니다.
파형 회귀(y=sin 2x) 문제에서 SGD+모멘텀으로 네 가지 조건을 비교합니다:
(A) 작은 고정 학습률 (B) 큰 고정 학습률(발산 사고 재현)
(C) 큰 학습률 + 그래디언트 클리핑 (D) warmup+cosine 스케줄 + 클리핑.
loss 곡선과 학습률 스케줄 곡선을 PNG 로 저장합니다.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
EPOCHS, BATCH = 60, 32


def make_data():
    """합성 회귀 데이터: y = sin(2x) + 잡음. 곡선을 따라 그리는 문제입니다."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, (512, 1)).astype(np.float32)
    y = (np.sin(2 * X) + 0.1 * rng.normal(size=(512, 1))).astype(np.float32)
    X_va = np.linspace(-3, 3, 200, dtype=np.float32).reshape(-1, 1)
    y_va = np.sin(2 * X_va)
    to = torch.from_numpy
    return to(X), to(y), to(X_va), to(y_va)


def build_model():
    return nn.Sequential(nn.Linear(1, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, 1))


def lr_at(step, total, base_lr, schedule):
    """스텝별 학습률. warmcos = 처음 10% 는 서서히 올리고(warmup),
    이후 cosine 곡선으로 부드럽게 내립니다(decay)."""
    if schedule == "fixed":
        return base_lr
    warm = int(0.1 * total)
    if step < warm:
        return base_lr * step / warm                      # 출근 직후 워밍업
    progress = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))  # 마감 전 미세 조정


def train(name, base_lr, schedule="fixed", clip=None):
    """미니배치 SGD+모멘텀 학습. 에폭별 valid MSE 와 스텝별 학습률 기록 반환."""
    torch.manual_seed(2)                                  # 모든 조건 같은 초기 가중치
    X, y, X_va, y_va = make_data()
    model = build_model()
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    loss_fn = nn.MSELoss()
    n, spe = len(X), len(X) // BATCH                      # spe = 에폭당 스텝 수
    total = EPOCHS * spe
    g = torch.Generator().manual_seed(0)

    va_hist, lr_hist, step = [], [], 0
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(spe):
            b = perm[i * BATCH:(i + 1) * BATCH]
            lr = lr_at(step, total, base_lr, schedule)
            for group in opt.param_groups:
                group["lr"] = lr
            lr_hist.append(lr)
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            if clip is not None:                          # 기울기 전체 크기(norm)를 제한
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            step += 1
        with torch.no_grad():
            va_hist.append(loss_fn(model(X_va), y_va).item())

    final = va_hist[-1]
    status = "발산(NaN)" if math.isnan(final) else f"{final:.4f}"
    print(f"    {name:34s}: 최종 valid MSE = {status}")
    return va_hist, lr_hist


def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 문제: y = sin(2x) 곡선 회귀, 1-64-64-1 MLP, SGD+모멘텀 0.9")
    print(f"    {EPOCHS}에폭 x 배치 {BATCH} (에폭당 16스텝). 조건 4개, 시드 동일.\n")

    print("[2] 조건별 학습")
    runs = {}
    runs["A"] = train("A. 고정 lr=0.05 (안전 운전)", 0.05)
    runs["B"] = train("B. 고정 lr=0.2  (과속, 무방비)", 0.2)
    runs["C"] = train("C. lr=0.2 + 클리핑 0.5", 0.2, clip=0.5)
    runs["D"] = train("D. warmup+cosine lr=0.2 + 클리핑", 0.2, schedule="warmcos", clip=0.5)

    a, b, c, d = (runs[k][0][-1] for k in "ABCD")
    assert math.isnan(b), "B 는 발산 시연용인데 발산하지 않았습니다"
    assert not any(math.isnan(v) for v in (a, c, d)), "A/C/D 가 발산했습니다"
    assert d < a, "스케줄 조합(D)이 기본(A)보다 좋아야 합니다"

    print("\n[3] 해석")
    print("    A: 작게 밟으면 안전하게 수렴 (0.007 수준).")
    print("    B: lr 4배 -> 몇 스텝 만에 loss 폭발, NaN. '과속 사고' 현장입니다.")
    print("    C: 같은 과속이라도 클리핑(핸들 각도 제한)이 있으면 살아남습니다.")
    print("    D: 여기에 warmup+cosine 까지 -> 전 조건 중 최저 MSE.")
    print("       초반엔 천천히(워밍업), 중반엔 과감히, 막판엔 세밀하게 — 스케줄의 힘입니다.\n")

    print("[4] 그림 저장 (loss 곡선 + 학습률 스케줄)")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    colors = {"A": "#1f77b4", "B": "#d62728", "C": "#ff7f0e", "D": "#2ca02c"}
    labels = {"A": "A fixed 0.05", "B": "B fixed 0.2 (diverged)",
              "C": "C 0.2 + clip", "D": "D warmup+cosine + clip"}
    for key, (va, _) in runs.items():
        vals = [min(v, 10.0) if not math.isnan(v) else 10.0 for v in va]  # NaN 은 위로 잘라 표시
        axes[0].plot(range(1, EPOCHS + 1), vals, label=labels[key], color=colors[key])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("valid MSE (log, capped at 10)")
    axes[0].set_title("Stability: same model, four training recipes")
    axes[0].legend(fontsize=8)
    for key in ("A", "D"):
        axes[1].plot(runs[key][1], label=labels[key], color=colors[key])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("LR schedule: fixed vs warmup+cosine")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "lr_schedule_compare.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"    저장: {png_path}\n")

    print("[5] 개념 메모 — 배치정규화(BatchNorm)")
    print("    층을 지날수록 값의 분포(눈금)가 흐트러져 학습이 불안정해질 수 있습니다.")
    print("    배치정규화는 각 층 입력을 배치 단위로 평균 0/분산 1 로 다시 맞춰")
    print("    큰 학습률도 견디게 만드는 '층 사이 눈금 교정기'입니다.")
    print("    코드로는 nn.BatchNorm1d(64) 를 Linear 뒤에 끼우면 됩니다 (과제 3).")


if __name__ == "__main__":
    main()
