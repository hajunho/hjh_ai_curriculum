"""
level03 — Matplotlib 기본 그래프

figure/axes 구조를 따라 카페 체인 매출로
선그래프(월별 추이) · 막대그래프(지점 비교) · 산점도(광고비-매출)를
outputs/ 아래 PNG 로 저장합니다. 한글 폰트 자동 설정도 포함합니다.
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 반드시 pyplot import '전에' — 화면 대신 파일로 저장
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def set_korean_font() -> None:
    """macOS/Windows/Linux 대표 한글 폰트를 순서대로 찾아 지정합니다."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            print(f"[0] 한글 폰트 설정: {cand}")
            break
    else:
        print("[0] 한글 폰트를 찾지 못해 기본 폰트를 사용합니다 (한글이 깨질 수 있음)")
    plt.rcParams["axes.unicode_minus"] = False


def load_sales() -> pd.DataFrame:
    """매출 데이터 정제 + 월 컬럼 추가."""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["month"] = df["date"].str[:7]            # '2025-03' 형태
    print(f"[1] 데이터 준비 완료: {len(df):,}행, 기간 {df['date'].min()} ~ {df['date'].max()}")
    return df


def plot_line_monthly(df: pd.DataFrame) -> None:
    """선그래프: 시간에 따른 변화는 선으로."""
    monthly = df.groupby("month")["revenue"].sum() / 1e8  # 억 원 단위

    fig, ax = plt.subplots(figsize=(8, 4.5))              # 종이 + 문단
    ax.plot(monthly.index, monthly.values,                # 문장(선 긋기)
            marker="o", color="#4878cf", lw=2)
    for x, y in monthly.items():                          # 값 라벨
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("월별 총매출 추이")                        # 소제목
    ax.set_xlabel("월")
    ax.set_ylabel("총매출 (억 원)")
    ax.set_ylim(0, monthly.max() * 1.2)                   # 0 부터 시작 (level04 예고)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "line_monthly.png")
    fig.savefig(path, dpi=120)                            # 제출(저장)
    plt.close(fig)                                        # 종이 치우기
    print(f"[2] 선그래프 저장: {path}")


def plot_bar_stores(df: pd.DataFrame) -> None:
    """막대그래프: 범주 비교는 막대로. 1등만 색으로 강조."""
    stores = (df.groupby("store")["revenue"].sum() / 1e8).sort_values(ascending=False)
    colors = ["#d1495b" if i == 0 else "#9aa7b5" for i in range(len(stores))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(stores.index, stores.values, color=colors)
    ax.set_title("지점별 총매출 비교 (강조 = 1위 지점)")
    ax.set_xlabel("지점")
    ax.set_ylabel("총매출 (억 원)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "bar_stores.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 막대그래프 저장: {path} (1위: {stores.index[0]})")


def plot_scatter_ad(df: pd.DataFrame) -> None:
    """산점도: 두 수치의 관계는 점으로. (다음다음 레벨의 예고편)"""
    daily = df.groupby("day_index").agg(
        ad=("ad_cost", "sum"), rev=("revenue", "sum"))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(daily["ad"] / 1e6, daily["rev"] / 1e8,
               s=18, alpha=0.5, color="#2e6f40")
    ax.set_title("일별 광고비 vs 일별 매출")
    ax.set_xlabel("광고비 (백만 원)")
    ax.set_ylabel("매출 (억 원)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "scatter_ad.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 산점도 저장: {path}")
    print("    점 구름이 오른쪽 위로 기울어 보이나요? 이 '관계'는 level05 에서 파헤칩니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    df = load_sales()
    plot_line_monthly(df)
    plot_bar_stores(df)
    plot_scatter_ad(df)
    print("[5] 모든 그래프는 '종이->문단->문장->소제목->저장' 다섯 단계로 만들어졌습니다.")


if __name__ == "__main__":
    main()
