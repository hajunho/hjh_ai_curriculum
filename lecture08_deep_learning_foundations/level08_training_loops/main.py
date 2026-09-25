"""
Dataset/DataLoader 와 표준 학습 루프를 배웁니다.
구독 이탈 데이터(churn_table)를 커스텀 Dataset 으로 감싸고
DataLoader 로 미니배치를 꺼내며 에폭 단위로 학습/검증하는
실무 표준 템플릿을 구현한 뒤, train/valid loss 곡선을 PNG 로 저장합니다.
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]


class ChurnDataset(Dataset):
    """Dataset = '창고 사서'. 두 가지 질문에만 답하면 됩니다:
    len(전체 몇 건?) 과 getitem(i번째 한 건 주세요)."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_datasets():
    """churn_table -> 표준화 -> train/valid Dataset 두 개."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8   # 학습셋으로만 통계 계산
    X = (X - mu) / sd
    return ChurnDataset(X[tr], y[tr]), ChurnDataset(X[va], y[va])


def build_model():
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(),
                         nn.Linear(16, 8), nn.ReLU(),
                         nn.Linear(8, 1))     # Sigmoid 없음: BCEWithLogitsLoss 사용


def run_epoch(model, loader, loss_fn, opt=None):
    """한 에폭 = 창고의 모든 접시(배치)를 한 바퀴. opt 가 있으면 학습, 없으면 평가."""
    training = opt is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for xb, yb in loader:                     # DataLoader 가 접시를 하나씩 배급
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                opt.zero_grad()                   # 표준 5단계 루프 (level06)
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(xb)   # 배치 크기로 가중 평균
            total_correct += ((logits > 0) == yb.bool()).sum().item()
            total_n += len(xb)
    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 데이터 준비 — Dataset 두 개 (train/valid)")
    train_ds, valid_ds = load_datasets()
    print(f"    train {len(train_ds)}건 / valid {len(valid_ds)}건")
    xb0, yb0 = train_ds[0]
    print(f"    train_ds[0] -> 특징 shape {tuple(xb0.shape)}, 라벨 {yb0.item():.0f}\n")

    print("[2] DataLoader — 배치 크기 64, 매 에폭 셔플")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(0))
    valid_loader = DataLoader(valid_ds, batch_size=256, shuffle=False)  # 평가는 셔플 불필요
    n_steps = len(train_loader)
    print(f"    1에폭 = {len(train_ds)}건 / 64 = {n_steps}스텝(배치)")
    print("    용어: 스텝=배치 1개 처리, 에폭=전체 데이터 한 바퀴\n")

    print("[3] 학습 — 표준 템플릿 (에폭마다 train 한 바퀴 + valid 채점)")
    model = build_model()
    loss_fn = nn.BCEWithLogitsLoss()              # Sigmoid+BCE 를 합친 안정 버전
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    history = {"train": [], "valid": [], "acc": []}
    n_epochs = 40
    for epoch in range(1, n_epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, loss_fn, opt)
        va_loss, va_acc = run_epoch(model, valid_loader, loss_fn)
        history["train"].append(tr_loss)
        history["valid"].append(va_loss)
        history["acc"].append(va_acc)
        if epoch in (1, 5, 10, 20, 30, 40):
            print(f"    epoch {epoch:2d}: train loss {tr_loss:.4f} | "
                  f"valid loss {va_loss:.4f} | valid 정확도 {va_acc:.3f}")

    print("\n[4] 학습곡선 저장")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, n_epochs + 1)
    ax.plot(epochs, history["train"], label="train loss", color="#1f77b4")
    ax.plot(epochs, history["valid"], label="valid loss", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training curve (churn MLP, batch=64)")
    ax.legend()
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "training_curve.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"    저장: {png_path}")

    print("\n[5] 학습곡선 읽는 법")
    print("    - 두 곡선이 같이 내려간다      -> 정상 학습 중")
    print("    - train 만 내려가고 valid 반등 -> 과적합 시작 (level09 주제)")
    print("    - 둘 다 높은 채 평평           -> 과소적합: 모델/에폭/학습률 재검토")
    print(f"    이번 실행: 최종 train {history['train'][-1]:.4f}, valid {history['valid'][-1]:.4f}, "
          f"valid 정확도 {history['acc'][-1]:.3f}")


if __name__ == "__main__":
    main()
