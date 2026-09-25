"""
level09 — 가설검정과 p-value

두 지점(부산점 vs 대전점)의 건당 매출 차이가 우연인지 재판에 부칩니다.
1) 순열검정: 지점 라벨을 10,000번 섞어 '우연의 차이' 분포에서 p-value 를 직접 셈
2) scipy 의 Welch t-검정과 결과 비교
3) 진짜 차이가 없는 표본으로 대조 실험 (1종 오류의 의미)
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
N_PERM = 10_000


def set_korean_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_two_stores() -> tuple[np.ndarray, np.ndarray]:
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(909)
    a = rng.choice(df[df["store"] == "부산점"]["revenue"].to_numpy(), 150, replace=False)
    b = rng.choice(df[df["store"] == "대전점"]["revenue"].to_numpy(), 150, replace=False)
    return a.astype(float), b.astype(float)


def permutation_test(a: np.ndarray, b: np.ndarray, seed: int = 910) -> tuple[float, np.ndarray, float]:
    """라벨 섞기 검정: '우연만으로 생기는 차이'의 분포를 직접 만듭니다."""
    rng = np.random.default_rng(seed)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    fake_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(pooled)                      # H0 세계: 라벨은 무의미
        fake_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    # 양측검정: 관측 차이의 절대값 이상이 우연히 나온 비율
    p = (np.abs(fake_diffs) >= abs(observed)).mean()
    return p, fake_diffs, observed


def trial_of_two_stores() -> None:
    a, b = load_two_stores()
    print("[1] 피고: '부산점과 대전점의 매출 차이는 우연이다' (귀무가설 H0)")
    print(f"    부산점 표본 평균 {a.mean() / 1e4:8.1f} 만 원 (n={len(a)})")
    print(f"    대전점 표본 평균 {b.mean() / 1e4:8.1f} 만 원 (n={len(b)})")
    print(f"    관측된 차이     {(a.mean() - b.mean()) / 1e4:+8.1f} 만 원")

    p_perm, fake_diffs, observed = permutation_test(a, b)
    print()
    print(f"[2] 순열검정 — 라벨을 {N_PERM:,}번 섞어 만든 '우연의 차이' 분포와 비교")
    print(f"    우연의 차이 분포: 평균 {fake_diffs.mean() / 1e4:+.2f}, "
          f"표준편차 {fake_diffs.std() / 1e4:.2f} (만 원)")
    print(f"    p-value(순열) = {p_perm:.4f}")

    t_stat, p_scipy = stats.ttest_ind(a, b, equal_var=False)   # Welch t-검정
    print()
    print("[3] scipy 한 줄 검정 — stats.ttest_ind(a, b, equal_var=False)")
    print(f"    t = {t_stat:.3f}, p-value(t검정) = {p_scipy:.4f}")
    print(f"    -> 시뮬레이션 p({p_perm:.4f}) ≈ 공식 p({p_scipy:.4f}). 같은 것을 계산합니다.")
    verdict = "기각 — 우연으로 보기 어렵다 (차이 인정)" if p_scipy < 0.05 else "기각 실패 — 우연으로도 설명 가능"
    print(f"    판결(유의수준 5%): H0 {verdict}")

    # 그림: 우연의 분포 + 관측 차이 위치
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(fake_diffs / 1e4, bins=60, color="#9fbce8", edgecolor="white",
            label="우연만으로 생긴 차이 (라벨 섞기 10,000회)")
    ax.axvline(observed / 1e4, color="#d1495b", lw=2,
               label=f"실제 관측 차이 {observed / 1e4:+.1f}만 원")
    ax.axvline(-observed / 1e4, color="#d1495b", lw=1, ls="--")
    ax.set_title(f"p-value = 빨간 선 밖 꼬리의 비율 = {p_perm:.4f}")
    ax.set_xlabel("두 지점 평균 차이 (만 원)")
    ax.set_ylabel("빈도")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "permutation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 순열분포 그림 저장: {path}")


def control_experiment() -> None:
    """진짜 차이가 '없는' 두 표본 — 같은 지점을 반으로 쪼개 검정."""
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(911)
    hongdae = df[df["store"] == "홍대점"]["revenue"].to_numpy(dtype=float).copy()
    rng.shuffle(hongdae)
    half1, half2 = hongdae[:100], hongdae[100:200]

    t_stat, p = stats.ttest_ind(half1, half2, equal_var=False)
    print()
    print("[5] 대조 실험 — 같은 홍대점 매출을 무작위로 반씩 쪼개 검정")
    print(f"    (정의상 진짜 차이 없음) t = {t_stat:.3f}, p = {p:.4f}")
    print("    -> p 가 큼 = '우연으로 충분히 설명된다'. 단, 이것이 '차이 없음의")
    print("       증명'은 아닙니다. 무죄 판결과 결백 증명은 다릅니다.")
    print("    -> 이런 '차이 없는 검정'도 100번 중 약 5번은 p<0.05 가 나옵니다.")
    print("       그것이 유의수준 5% = 1종 오류(오탐)를 감수한다는 계약의 뜻입니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    trial_of_two_stores()
    control_experiment()


if __name__ == "__main__":
    main()
