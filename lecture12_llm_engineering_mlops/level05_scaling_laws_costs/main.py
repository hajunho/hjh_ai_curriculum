"""
스케일링 법칙과 학습 비용 계산 실습.
- 학습 연산량 FLOPs ≈ 6 x 파라미터 수 x 토큰 수 공식으로 GPU 시간을 견적내고,
  전기료·클라우드 비용까지 돈으로 환산해 봅니다.
- 친칠라 직관(파라미터의 약 20배 토큰)을 검증하고,
  '같은 예산이면 어떤 크기의 모델이 최적인가'를 스케일링 곡선 PNG로 그립니다.
"""
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_korean_font():
    """OS에 설치된 한글 폰트를 찾아 등록합니다 (없으면 기본 폰트)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

MFU = 0.35          # 실제로 뽑아 쓰는 GPU 성능 비율 (Model FLOPs Utilization)
PUE = 1.3           # 데이터센터 전력 효율 계수 (냉각 등 부대 전력 포함)
ELEC_USD_KWH = 0.12 # 산업용 전기요금 (달러/kWh, 대략값)

GPUS = {  # 이름: (bf16 최대 TFLOPS, 소비전력 W, 클라우드 시간당 달러)
    "H200":     (989.0, 700, 3.50),
    "A100":     (312.0, 400, 1.80),
    "4090급 소비자용": (165.0, 450, 0.45),
}

# 친칠라 계열 논문에서 보고된 근사 계수로 만든 손실 예측 함수
def predicted_loss(n_params, n_tokens):
    """L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 — 크기·데이터가 loss를 결정."""
    return 1.69 + 406.4 / n_params ** 0.34 + 410.7 / n_tokens ** 0.28


def fmt(x):
    """큰 숫자를 읽기 쉬운 한국식 단위로."""
    for unit, div in [("조", 1e12), ("억", 1e8), ("만", 1e4)]:
        if abs(x) >= div:
            return f"{x / div:,.1f}{unit}"
    return f"{x:,.0f}"


def estimate(name, n_params, n_tokens, gpu="H200", n_gpus=1):
    """FLOPs → GPU시간 → 전기료·클라우드 비용 견적을 출력한다."""
    tflops, watt, usd_hr = GPUS[gpu]
    flops = 6.0 * n_params * n_tokens                      # 학습 연산량 근사 공식
    gpu_hours = flops / (tflops * 1e12 * MFU) / 3600.0     # 총 GPU 시간
    wall_days = gpu_hours / n_gpus / 24.0                  # 벽시계 소요일
    kwh = watt * gpu_hours * PUE / 1000.0                  # 전력 사용량
    elec = kwh * ELEC_USD_KWH
    cloud = gpu_hours * usd_hr
    ratio = n_tokens / n_params
    verdict = "적정(친칠라 근처)" if 10 <= ratio <= 40 else \
              ("데이터 부족 — 모델이 덜 배움" if ratio < 10 else "데이터 과다 — 작은 모델엔 낭비일 수도")
    print(f"  ▶ {name}")
    print(f"     파라미터 {fmt(n_params)}개 x 토큰 {fmt(n_tokens)}개 (토큰/파라미터 = {ratio:,.1f} → {verdict})")
    dur = f"GPU {gpu_hours:,.1f}시간, 약 {wall_days:,.1f}일" if gpu_hours >= 0.1 \
        else f"GPU {gpu_hours * 3600:.2f}초"
    print(f"     FLOPs 6ND = {flops:.2e} | {gpu} x {n_gpus:,}장 기준 {dur}")
    print(f"     전기 {kwh:,.0f} kWh ≈ ${elec:,.0f} | 클라우드 임대 ≈ ${cloud:,.0f}")
    if n_params >= 1e7:  # 손실 예측 공식은 대형 모델용 근사라 초소형에는 무의미
        print(f"     예상 최종 loss ≈ {predicted_loss(n_params, n_tokens):.3f}")
    else:
        print("     (loss 예측 공식은 대형 모델용 근사라 이 크기에는 적용하지 않습니다)")


def draw_scaling_chart(out_path):
    """같은 컴퓨트 예산에서 모델 크기별 loss 곡선 + 최적 프런티어를 그린다."""
    # dataviz 기본 팔레트: 파랑/주황/아쿠아 3계열 (고정 순서), 밝은 표면
    colors = {"1억": "#2a78d6", "10억": "#eb6834", "100억": "#1baf7a"}
    sizes = {"1억": 1e8, "10억": 1e9, "100억": 1e10}
    C = np.logspace(19.5, 24, 200)                         # 컴퓨트 예산 축
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for label, n in sizes.items():
        D = C / (6.0 * n)                                  # 예산을 다 쓰면 토큰 수가 결정됨
        L = predicted_loss(n, D)
        valid = D >= 1e8                                   # 토큰이 너무 적은 구간은 제외
        ax.plot(C[valid], L[valid], lw=2, color=colors[label],
                label=f"파라미터 {label} 개")
        ax.annotate(f"{label}", (C[valid][-1], L[valid][-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=colors[label], fontsize=10, va="center")
    n_opt = np.sqrt(C / 120.0)                             # C=6ND, D=20N → N=√(C/120)
    ax.plot(C, predicted_loss(n_opt, 20 * n_opt), ls="--", lw=1.6,
            color="#52514e", label="컴퓨트 최적 프런티어 (친칠라)")
    ax.set_xscale("log")
    ax.set_xlabel("컴퓨트 예산 (학습 FLOPs)", color="#0b0b0b")
    ax.set_ylabel("예측 손실 (낮을수록 좋음)", color="#0b0b0b")
    ax.set_title("같은 예산에서도 모델 크기에 따라 도달 가능한 loss가 다르다", color="#0b0b0b")
    ax.grid(True, alpha=0.25, lw=0.6)                      # 격자는 은은하게
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def main():
    np.random.seed(0)  # 관례상 seed 고정 (이 레벨은 난수 미사용)
    set_korean_font()

    # [1] 학습 비용의 기본 공식 --------------------------------------------
    print("[1] 기본 공식: 학습 FLOPs ≈ 6 x N(파라미터) x D(토큰)")
    print("    순전파 2ND + 역전파 4ND. 여기에 GPU 실효 성능(MFU)을 나누면 시간이 나옵니다.")
    print(f"    가정: MFU {MFU:.0%}, 전기 ${ELEC_USD_KWH}/kWh, PUE {PUE}\n")

    # [2] 규모별 견적 — 미니어처부터 대형까지 -------------------------------
    print("[2] 규모별 학습 비용 견적")
    estimate("우리 TinyGPT (Level 04에서 직접 학습한 모델)",
             n_params=1.11e5, n_tokens=2.15e6, gpu="4090급 소비자용", n_gpus=1)
    estimate("소형 모델 (1.2억 파라미터)", 1.2e8, 2.4e9, gpu="A100", n_gpus=8)
    estimate("중형 모델 (70억 파라미터)", 7e9, 1.4e11, gpu="H200", n_gpus=256)
    estimate("대형 모델 (700억 파라미터)", 7e10, 1.4e12, gpu="H200", n_gpus=2048)

    # [3] 친칠라 직관 — 예산이 정해지면 최적 크기도 정해진다 -----------------
    print("\n[3] 친칠라 직관: 컴퓨트 예산 C가 정해지면 최적 조합은 D ≈ 20N")
    print("    C = 6ND 에 D = 20N 을 넣으면 N_opt = √(C/120)")
    for C in (1e21, 1e22, 1e23, 1e24):
        n_opt = (C / 120.0) ** 0.5
        d_opt = 20.0 * n_opt
        print(f"    예산 {C:.0e} FLOPs → 최적 파라미터 {fmt(n_opt)}개, 토큰 {fmt(d_opt)}개, "
              f"loss ≈ {predicted_loss(n_opt, d_opt):.3f}")
    print("    → 예산 대비 모델이 너무 크면 '덜 배운 큰 모델', 너무 작으면 '한계에 막힌 작은 모델'이 됩니다.")

    # [4] 스케일링 곡선 그리기 ----------------------------------------------
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "scaling_curves.png"
    draw_scaling_chart(out_path)
    print(f"\n[4] 스케일링 곡선 저장: {out_path}")
    print("    곡선 읽는 법: 예산이 작을 때는 작은 모델이 유리하지만, 예산이 커지면")
    print("    곡선이 교차하며 큰 모델이 역전합니다. 점선(친칠라 프런티어)이 각 예산에서의 한계선입니다.")

    # [5] 실무 감각 정리 ----------------------------------------------------
    print("\n[5] 실무 감각 요약")
    print("    - 사전학습 비용은 수백만~수천만 달러 → 직접 하는 회사는 극소수")
    print("    - 대부분의 기업은 파인튜닝(다음 레벨)이나 API 활용으로 충분")
    print("    - 견적 회의에서 'FLOPs=6ND'와 '토큰은 파라미터의 약 20배' 두 개만 기억해도")
    print("      제안서의 규모가 타당한지 어림 검산할 수 있습니다.")


if __name__ == "__main__":
    main()
