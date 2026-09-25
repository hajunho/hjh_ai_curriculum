"""
level04 — 좋은 그래프 vs 나쁜 그래프

완전히 같은 데이터로 '왜곡 차트'와 '정직한 차트'를 나란히 그립니다.
1) 축 절단으로 3% 차이를 압도적 격차처럼 보이기
2) 3D 느낌의 파이차트 vs 정렬된 가로 막대
3) y축 범위 조작으로 미세 등락을 롤러코스터로 만들기
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

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


def load_data():
    rows = hjh_data.sales_table(n_days=90, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    store_rev = (df.groupby("store")["revenue"].sum() / 1e8)  # 억 원
    return df, store_rev


def crime1_truncated_axis(store_rev: pd.Series) -> None:
    """죄목 1: 막대그래프 y축 절단."""
    two = store_rev.sort_values(ascending=False).iloc[[1, 2]]  # 차이 작은 두 지점
    a, b = two.index
    gap_pct = (two[a] - two[b]) / two[b] * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#d1495b", "#9aa7b5"]

    bad.bar(two.index, two.values, color=colors)
    bad.set_ylim(two.min() * 0.985, two.max() * 1.005)   # 축 절단!
    bad.set_title(f"[왜곡] 축 절단 — {gap_pct:.1f}% 차이가 압승처럼 보임")
    bad.set_ylabel("총매출 (억 원)")

    good.bar(two.index, two.values, color=colors)
    good.set_ylim(0, two.max() * 1.15)                   # 0 부터 시작
    good.set_title("[정직] 막대는 0부터 — 실제 격차")
    good.set_ylabel("총매출 (억 원)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "truncated_axis.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] 축 절단 비교 저장: {path}")
    print(f"    실제 차이는 {gap_pct:.1f}% 뿐입니다. 왼쪽 그림의 첫인상과 비교해 보세요.")


def crime2_pie_vs_bar(store_rev: pd.Series) -> None:
    """죄목 2: 비슷한 값들의 파이차트 vs 정렬된 가로 막대."""
    share = store_rev / store_rev.sum() * 100

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.8))
    # 그림자 + 튀어나온 조각 = 회의실 3D 파이의 재현
    bad.pie(share.values, labels=share.index, shadow=True,
            explode=[0.08 if i == 0 else 0 for i in range(len(share))],
            startangle=90)
    bad.set_title("[왜곡] 파이 — 어느 조각이 2등인지 보이나요?")

    ordered = share.sort_values()
    good.barh(ordered.index, ordered.values, color="#4878cf")
    for y, v in enumerate(ordered.values):
        good.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    good.set_title("[정직] 정렬된 막대 — 순위와 격차가 즉시 보임")
    good.set_xlabel("매출 점유율 (%)")
    good.set_xlim(0, ordered.max() * 1.25)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pie_vs_bar.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 파이 vs 막대 저장: {path}")


def crime3_inflated_line(df: pd.DataFrame) -> None:
    """죄목 3: y축 범위를 좁혀 미세 등락을 롤러코스터로."""
    daily = df.groupby("day_index")["revenue"].sum() / 1e8
    weekly = daily.rolling(7).mean().dropna()            # 주 이동평균으로 완만하게

    fig, (bad, good) = plt.subplots(1, 2, figsize=(11, 4.2))
    bad.plot(weekly.index, weekly.values, color="#d1495b", lw=2)
    bad.set_ylim(weekly.min() * 0.998, weekly.max() * 1.002)  # 범위 조작!
    bad.set_title("[왜곡] 축 확대 — 대폭락과 대반등의 드라마?")
    bad.set_xlabel("일차")
    bad.set_ylabel("일매출 7일 평균 (억 원)")

    good.plot(weekly.index, weekly.values, color="#4878cf", lw=2)
    good.set_ylim(0, weekly.max() * 1.2)
    good.set_title("[정직] 맥락 있는 범위 — 사실상 안정적")
    good.set_xlabel("일차")
    good.set_ylabel("일매출 7일 평균 (억 원)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "inflated_line.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    swing = (weekly.max() - weekly.min()) / weekly.mean() * 100
    print(f"[4] 선그래프 범위 조작 저장: {path}")
    print(f"    실제 등락 폭은 평균 대비 ±{swing / 2:.1f}% 수준입니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    df, store_rev = load_data()

    print("[1] 재료가 되는 실제 숫자 (지점별 총매출, 억 원)")
    for store, v in store_rev.sort_values(ascending=False).items():
        print(f"    {store:<6} {v:8.2f}")
    print("    -> 아래 세 그림의 '왜곡'과 '정직'은 전부 이 같은 숫자에서 나옵니다.")

    crime1_truncated_axis(store_rev)
    crime2_pie_vs_bar(store_rev)
    crime3_inflated_line(df)
    print("[5] 결론: 그래프의 첫인상이 숫자 정독의 결론과 다르면, 그 그래프는 나쁜 그래프입니다.")


if __name__ == "__main__":
    main()
