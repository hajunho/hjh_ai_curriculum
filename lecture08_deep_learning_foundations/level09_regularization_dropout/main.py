"""
과적합을 일부러 일으킨 뒤 세 가지 처방의 효과를 비교합니다.
구독 이탈 데이터에서 학습 데이터를 120건만 주고 큰 MLP 를 외우게 만든 뒤,
(1) 드롭아웃 (2) weight decay (3) 조기종료가 각각
검증(valid) 성능을 얼마나 회복시키는지 곡선 PNG 와 표로 확인합니다.
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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FEATURES = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
N_EPOCHS = 600


def load_data():
    """churn_table 2000건 중 '겨우 120건'만 학습에 사용 -> 과적합 유도.
    나머지 1880건은 검증용이라 성적표가 아주 안정적입니다."""
    rows = hjh_data.churn_table(n=2000, seed=7)
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[float(r["churned"])] for r in rows], dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr, va = idx[:120], idx[120:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8   # 학습셋 통계만 사용
    X = (X - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(y[tr]), t(X[va]), t(y[va])


def build_model(dropout=0.0):
    """데이터(120건) 대비 일부러 큰 MLP (약 4.7천 파라미터) = 통암기 유도 장치."""
    layers = [nn.Linear(6, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))       # 학습 중 뉴런 무작위 결근
    layers += [nn.Linear(64, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers)


def train(name, X_tr, y_tr, X_va, y_va, dropout=0.0, weight_decay=0.0):
    """full-batch Adam 학습. 에폭별 valid loss/정확도 기록을 반환합니다."""
    torch.manual_seed(1)                         # 모든 조건에 같은 초기 가중치
    model = build_model(dropout)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"tr_loss": [], "va_loss": [], "va_acc": []}
    for _ in range(N_EPOCHS):
        model.train()                            # 드롭아웃은 train 모드에서만 작동
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        model.eval()                             # 평가 때는 결근 없이 전원 출근
        with torch.no_grad():
            logits = model(X_va)
            hist["va_loss"].append(loss_fn(logits, y_va).item())
            hist["va_acc"].append(((logits > 0) == y_va.bool()).float().mean().item())
        hist["tr_loss"].append(loss.item())
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(X_tr) > 0) == y_tr.bool()).float().mean().item()
    print(f"    {name:26s}: train 정확도 {tr_acc:.3f} | 최종 valid 정확도 {hist['va_acc'][-1]:.3f}")
    return hist


def main():
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 실험 설계: 학습 120건 vs 파라미터 약 4,700개 — 통암기(과적합)가 일어나는 조건")
    X_tr, y_tr, X_va, y_va = load_data()
    print(f"    구독 이탈 이진 분류, 학습 {len(X_tr)}건 / 검증 {len(X_va)}건, {N_EPOCHS}에폭 full-batch\n")

    print("[2] 조건별 학습 (같은 시드, 같은 초기 가중치)")
    runs = {
        "baseline": train("baseline (무방비)", X_tr, y_tr, X_va, y_va),
        "dropout": train("dropout p=0.5", X_tr, y_tr, X_va, y_va, dropout=0.5),
        "weight_decay": train("weight decay 3e-2", X_tr, y_tr, X_va, y_va, weight_decay=3e-2),
    }

    print("\n[3] 조기종료(early stopping) — baseline 을 '가장 좋았던 순간'에 멈췄다면?")
    va = runs["baseline"]["va_loss"]
    best_epoch = int(np.argmin(va))              # valid loss 최저 에폭
    es_acc = runs["baseline"]["va_acc"][best_epoch]
    print(f"    valid loss 최저점: epoch {best_epoch + 1} (600에폭 중!)")
    print(f"    그 시점 valid 정확도 {es_acc:.3f} vs 끝까지 돌린 baseline {runs['baseline']['va_acc'][-1]:.3f}\n")

    print("[4] 최종 비교 표 (valid 정확도가 성적표)")
    rows = [("무방비 (600에폭 완주)", runs["baseline"]["va_acc"][-1]),
            ("드롭아웃 0.5", runs["dropout"]["va_acc"][-1]),
            ("weight decay 3e-2", runs["weight_decay"]["va_acc"][-1]),
            (f"조기종료 (epoch {best_epoch + 1})", es_acc)]
    print(f"    {'처방':26s} {'valid 정확도':>12s}")
    for name, acc in rows:
        print(f"    {name:28s} {acc:>10.3f}")
    assert max(r[1] for r in rows[1:]) > rows[0][1] + 0.02, "처방 효과가 재현되지 않았습니다"

    print("\n[5] 곡선 PNG 저장")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, N_EPOCHS + 1)
    colors = {"baseline": "#d62728", "dropout": "#1f77b4", "weight_decay": "#2ca02c"}
    labels = {"baseline": "baseline", "dropout": "dropout 0.5", "weight_decay": "weight decay 3e-2"}
    for key, hist in runs.items():
        axes[0].plot(epochs, hist["va_loss"], label=f"{labels[key]} (valid)", color=colors[key])
        axes[1].plot(epochs, hist["va_acc"], label=labels[key], color=colors[key])
    axes[0].plot(epochs, runs["baseline"]["tr_loss"], "--", color="#d62728",
                 alpha=0.5, label="baseline (train)")
    axes[0].axvline(best_epoch + 1, color="gray", ls=":", label="early stop point")
    axes[0].set_title("Loss: train memorizes, valid gets worse")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend(fontsize=8)
    axes[1].axvline(best_epoch + 1, color="gray", ls=":")
    axes[1].set_title("Valid accuracy by remedy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("valid accuracy")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "regularization_compare.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"    저장: {png_path}")

    print("\n[6] 정리: 과적합 처방은 '외우지 못하게 방해하기'입니다.")
    print("    결근 훈련(드롭아웃), 큰 가중치에 세금(weight decay), 성적 정점에서 멈추기(조기종료).")
    print("    실무에서는 셋을 조합해 쓰고, 최고의 처방은 언제나 '데이터 더 모으기'입니다.")


if __name__ == "__main__":
    main()
