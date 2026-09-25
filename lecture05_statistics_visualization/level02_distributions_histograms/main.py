"""
level02 — 분포와 히스토그램

카페 체인 매출 데이터의 분포를 히스토그램 PNG 3종으로 저장합니다.
1) bin 개수가 이야기(모양)를 바꾸는 실험
2) 오른쪽 꼬리(왜도)와 평균·중앙값의 어긋남
3) 평일/주말이 섞여 만들어진 두 봉우리 분포
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def set_korean_font() -> None:
    """OS 에 설치된 한글 폰트를 찾아 등록합니다 (없으면 기본 폰트)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


def load_sales() -> pd.DataFrame:
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0].copy()
    df["revenue_man"] = df["revenue"] / 10_000  # 만 원 단위로 읽기 쉽게
    return df


def describe(df: pd.DataFrame) -> None:
    rev = df["revenue_man"]
    mean, med = rev.mean(), rev.median()
    p95 = np.percentile(rev, 95)
    skew = rev.skew()
    print("[1] 매출(건당, 만 원) 분포 요약")
    print(f"    평균 {mean:.1f} | 중앙값 {med:.1f} | p95 {p95:.1f} | 왜도 {skew:.2f}")
    print(f"    -> 평균 > 중앙값, 왜도 > 0 : 오른쪽 꼬리가 긴 분포입니다.")


def plot_bins_experiment(df: pd.DataFrame) -> None:
    """같은 데이터를 bin 5 / 30 / 200 으로 그려 비교합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, bins in zip(axes, [5, 30, 200]):
        ax.hist(df["revenue_man"], bins=bins, color="#4878cf", edgecolor="white")
        ax.set_title(f"bins = {bins}")
        ax.set_xlabel("건당 매출 (만 원)")
    axes[0].set_ylabel("빈도")
    fig.suptitle("같은 데이터, 다른 bin — 축척이 이야기를 바꾼다")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bins.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[2] bin 실험 저장: {path}")
    print("    bins=5 는 봉우리를 뭉개고, bins=200 은 난수 요철까지 그립니다.")


def plot_skew(df: pd.DataFrame) -> None:
    """오른쪽 꼬리 분포 위에 평균/중앙값 세로선을 겹칩니다."""
    rev = df["revenue_man"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rev, bins=40, color="#9fbce8", edgecolor="white")
    ax.axvline(rev.mean(), color="#d1495b", lw=2, label=f"평균 {rev.mean():.0f}")
    ax.axvline(rev.median(), color="#2e6f40", lw=2, ls="--",
               label=f"중앙값 {rev.median():.0f}")
    ax.set_title("오른쪽 꼬리가 평균을 끌고 간다")
    ax.set_xlabel("건당 매출 (만 원)")
    ax.set_ylabel("빈도")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_skew.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[3] 왜도 그림 저장: {path}")


def plot_bimodal(df: pd.DataFrame) -> None:
    """평일/주말을 나눠 겹쳐 그리면 숨은 두 집단이 보입니다."""
    weekend = df[df["weekday"].isin(["토", "일"])]["revenue_man"]
    weekday = df[~df["weekday"].isin(["토", "일"])]["revenue_man"]
    bins = np.linspace(df["revenue_man"].min(), df["revenue_man"].max(), 40)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # density=True: 표본 수가 달라도 비율로 공정하게 비교
    ax.hist(weekday, bins=bins, density=True, alpha=0.6, label="평일", color="#4878cf")
    ax.hist(weekend, bins=bins, density=True, alpha=0.6, label="주말", color="#e1a03c")
    ax.set_title("하나로 보이던 분포는 사실 두 집단의 합이었다")
    ax.set_xlabel("건당 매출 (만 원)")
    ax.set_ylabel("비율 밀도")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "hist_bimodal.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 평일/주말 분리 그림 저장: {path}")
    print(f"    평일 중앙값 {weekday.median():.0f} vs 주말 중앙값 {weekend.median():.0f} (만 원)")
    print("    -> 집단이 섞인 분포를 통째로 해석하면 두 이야기를 하나로 뭉개게 됩니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_korean_font()
    df = load_sales()
    describe(df)
    plot_bins_experiment(df)
    plot_skew(df)
    plot_bimodal(df)


if __name__ == "__main__":
    main()
