"""
level05 — 상관관계와 인과관계

1부: 카페 매출 데이터에서 광고비-매출의 진짜 상관을 관찰합니다.
2부: 교란변수(더위)가 만든 '가짜 상관'을 합성 데이터로 제조한 뒤,
     층화(교란변수 구간별 분석)로 상관이 무너지는 것을 확인합니다.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def set_korean_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def part1_real_correlation() -> None:
    """광고비와 매출의 상관계수 + 산점도."""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    daily = df.groupby("day_index").agg(ad=("ad_cost", "sum"), rev=("revenue", "sum"))

    r = np.corrcoef(daily["ad"], daily["rev"])[0, 1]
    print("[1] 1부 — 일별 광고비 vs 일별 매출")
    print(f"    피어슨 상관계수 r = {r:.3f}")
    print("    -> 양의 상관이 있습니다. 하지만 이것만으로 '광고가 매출을 올렸다'고")
    print("       말할 수 없습니다. (성수기·주말 같은 교란변수가 남아 있습니다)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8, s=18, alpha=0.5, color="#4878cf")
    ax.set_title(f"광고비 vs 매출 (r = {r:.2f}) — 상관 ≠ 인과")
    ax.set_xlabel("일별 광고비 (백만 원)")
    ax.set_ylabel("일별 매출 (억 원)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ad_revenue.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] 산점도 저장: {path}")


def part2_confounder() -> None:
    """교란변수 실험: Z(더위) -> X(아이스크림), Z -> Y(물놀이 사고)."""
    rng = np.random.default_rng(505)  # seed 고정
    n = 400
    noise_scale = 2.0

    heat = rng.uniform(0, 10, size=n)                     # 교란변수 Z: 더위 지수
    icecream = 20 + 8 * heat + rng.normal(0, 8 * noise_scale, n)   # X = f(Z)+잡음
    accidents = 1 + 0.9 * heat + rng.normal(0, 0.9 * noise_scale, n)  # Y = g(Z)+잡음
    # 주목: icecream 과 accidents 는 서로를 전혀 참조하지 않습니다!

    r_total = np.corrcoef(icecream, accidents)[0, 1]
    print()
    print("[3] 2부 — 가짜 상관 제조")
    print("    생성 규칙: X(아이스크림) = f(더위)+잡음, Y(사고) = g(더위)+잡음")
    print("    X 와 Y 는 서로 독립적으로 만들어졌지만...")
    print(f"    전체 상관계수 r = {r_total:.3f}  <- 강한 상관처럼 보입니다!")

    # 층화: 더위 지수를 5개 구간으로 나눠 구간 '안'에서만 상관 계산
    bins = np.linspace(0, 10, 6)
    labels = np.digitize(heat, bins[1:-1])                # 0~4 구간 번호
    print()
    print("[4] 층화 실험 — 더위가 비슷한 날들끼리만 다시 보면")
    inner_rs = []
    for k in range(5):
        mask = labels == k
        r_k = np.corrcoef(icecream[mask], accidents[mask])[0, 1]
        inner_rs.append(r_k)
        print(f"    더위 구간 {bins[k]:.0f}~{bins[k + 1]:.0f} ({mask.sum():3d}일): r = {r_k:+.3f}")
    print(f"    구간 내 상관의 평균 = {np.mean(inner_rs):+.3f}  (0 근처로 무너짐)")
    print("    -> 전체 상관은 두 변수의 관계가 아니라 '더위'라는 몸통의 작품이었습니다.")

    # 그림: 전체(왜곡된 인상) vs 구간별 색칠(진실)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.scatter(icecream, accidents, s=14, alpha=0.5, color="#555555")
    left.set_title(f"전체로 보면: r = {r_total:.2f} (관계 있어 보임)")
    left.set_xlabel("아이스크림 판매량")
    left.set_ylabel("물놀이 사고 건수")

    cmap = ["#3b6bb5", "#5da05d", "#e1a03c", "#d1495b", "#7d4fa3"]
    for k in range(5):
        mask = labels == k
        right.scatter(icecream[mask], accidents[mask], s=14, alpha=0.6,
                      color=cmap[k], label=f"더위 {bins[k]:.0f}~{bins[k + 1]:.0f} (r={inner_rs[k]:+.2f})")
    right.set_title("더위 구간별로 보면: 덩어리 안은 무질서")
    right.set_xlabel("아이스크림 판매량")
    right.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "confounder.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[5] 교란변수 그림 저장: {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    part1_real_correlation()
    part2_confounder()
    print()
    print("[6] 결론: 상관은 '조사해 볼 가치가 있는 단서'이지 '행동해도 되는 증거'가")
    print("    아닙니다. 인과의 증거가 필요하면 실험(A/B 테스트, level10)으로 갑니다.")


if __name__ == "__main__":
    main()
