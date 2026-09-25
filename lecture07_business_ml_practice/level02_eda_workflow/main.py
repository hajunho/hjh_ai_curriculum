"""
churn_table 자동 EDA 리포트.
가설 수립 -> 기초 문진 -> 단변량 -> 이변량(타깃별 차이, 상관) -> 가설 판정 순서로 진행하고
분포 비교 그림과 상관 히트맵 PNG 2장을 outputs/ 에 저장합니다.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트가 있으면 사용 (없으면 영문 라벨만 깨지지 않게 유지)
for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if any(ft.name == _f for ft in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
NUM_COLS = ["tenure_months", "monthly_fee", "usage_days_30d",
            "support_calls_30d", "plan_changes", "auto_pay"]
TARGET = "churned"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(hjh_data.churn_table(n=2000, seed=7))

    print("=" * 62)
    print(" churn_table 자동 EDA 리포트")
    print("=" * 62)

    # [1] 가설 먼저 -----------------------------------------------------
    hypotheses = [
        ("H1", "최근 30일 이용일수가 적을수록 이탈이 많을 것", "usage_days_30d", "low"),
        ("H2", "고객센터 문의가 많을수록 이탈이 많을 것", "support_calls_30d", "high"),
        ("H3", "가입 기간이 길수록 이탈이 적을 것", "tenure_months", "low"),
    ]
    print("\n[1] 검증할 가설 (그래프보다 문장이 먼저)")
    for hid, text, _, _ in hypotheses:
        print(f"    {hid}. {text}")

    # [2] 기초 문진 -----------------------------------------------------
    print("\n[2] 기초 문진")
    print(f"    행 x 열: {df.shape[0]} x {df.shape[1]}")
    print(f"    결측치 총합: {int(df.isna().sum().sum())}개")
    print(f"    타깃(churned) 비율: {df[TARGET].mean():.1%}  <- 불균형! 정확도 평가 금지")

    # [3] 단변량 --------------------------------------------------------
    print("\n[3] 단변량 요약 (수치형)")
    desc = df[NUM_COLS].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.round(2).to_string())

    # [4] 이변량: 타깃 그룹별 평균 + 상관 -------------------------------
    print("\n[4] 이변량: 이탈(1)/잔류(0) 그룹별 평균 비교")
    grp = df.groupby(TARGET)[NUM_COLS].mean().T
    grp.columns = ["잔류(0)", "이탈(1)"]
    grp["차이배율"] = (grp["이탈(1)"] / grp["잔류(0)"]).round(2)
    print(grp.round(2).to_string())

    corr = df[NUM_COLS + [TARGET]].corr()
    rank = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n    타깃과의 상관계수 순위 (절대값 큰 순):")
    for name, v in rank.items():
        print(f"      {name:<18} {v:+.3f}")
    if rank.abs().max() > 0.9:
        print("      ! 상관 0.9 초과 변수 발견 -> 누설 의심, 생성 시점 확인 필요")

    # [5] 가설 판정 -----------------------------------------------------
    print("\n[5] 가설 판정")
    for hid, text, col, direction in hypotheses:
        churn_mean, stay_mean = grp.loc[col, "이탈(1)"], grp.loc[col, "잔류(0)"]
        supported = churn_mean < stay_mean if direction == "low" else churn_mean > stay_mean
        verdict = "지지" if supported else "기각"
        print(f"    {hid} [{verdict}] 이탈 {churn_mean:.2f} vs 잔류 {stay_mean:.2f} — {text}")

    # [6] 그림 저장 ------------------------------------------------------
    plot_cols = ["usage_days_30d", "support_calls_30d", "tenure_months", "plan_changes"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), plot_cols):
        stay = df.loc[df[TARGET] == 0, col]
        churn = df.loc[df[TARGET] == 1, col]
        ax.hist(stay, bins=20, alpha=0.6, label="stay(0)", density=True)
        ax.hist(churn, bins=20, alpha=0.6, label="churn(1)", density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Distribution by churn group")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "dist_by_target.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "correlation_heatmap.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    print("\n[6] 그림 저장 완료")
    print(f"    {p1}")
    print(f"    {p2}")
    print("\n    교훈: EDA는 그림 그리기가 아니라 '가설을 문장으로 적고 판정하는 절차'입니다.")


if __name__ == "__main__":
    main()
