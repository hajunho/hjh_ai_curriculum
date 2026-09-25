"""
level01 — 평균·중앙값·분산·표준편차

카페 체인 매출 데이터(hjh_data)로 대표값들을 계산·교차검증하고,
극단값(대형 단체주문 1건)이 평균·중앙값·표준편차를 각각 얼마나
끌고 가는지 단계별로 실험합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_clean_sales() -> pd.DataFrame:
    """매출 데이터를 불러와 결측·음수 오염값을 제거합니다."""
    rows = hjh_data.sales_table(n_days=180, seed=42)
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.dropna(subset=["revenue"])          # 결측 제거
    df = df[df["revenue"] > 0].copy()           # 음수 오염값 제거
    df["revenue"] = df["revenue"].astype(float)
    print(f"[1] 데이터 정제: {before:,}행 -> {len(df):,}행 "
          f"(결측/음수 {before - len(df)}건 제거)")
    return df


def describe_revenue(df: pd.DataFrame) -> None:
    """pandas 계산값과 numpy 직접 계산값을 교차 검증합니다."""
    rev = df["revenue"].to_numpy()
    mean_np = rev.sum() / len(rev)              # 평균을 손으로
    var_np = ((rev - mean_np) ** 2).mean()      # 분산(모분산)을 손으로
    std_np = var_np ** 0.5

    print()
    print("[2] 전체 매출(건당)의 대표값 — pandas vs numpy 직접 계산")
    print(f"    평균     : {df['revenue'].mean():>14,.0f} 원 | 직접 계산 {mean_np:>14,.0f} 원")
    print(f"    중앙값   : {df['revenue'].median():>14,.0f} 원")
    print(f"    분산     : {df['revenue'].var(ddof=0):>14,.0f} 원^2 | 직접 계산 {var_np:>14,.0f} 원^2")
    print(f"    표준편차 : {df['revenue'].std(ddof=0):>14,.0f} 원 | 직접 계산 {std_np:>14,.0f} 원")
    print("    -> 평균 > 중앙값 : 매출 분포의 꼬리가 큰 값 쪽으로 길다는 신호입니다.")


def by_store(df: pd.DataFrame) -> None:
    """지점별 평균·중앙값·표준편차·변동계수를 비교합니다."""
    g = df.groupby("store")["revenue"].agg(["mean", "median", "std"])
    g["cv"] = g["std"] / g["mean"]              # 변동계수
    print()
    print("[3] 지점별 대표값 (변동계수 = 표준편차/평균)")
    print(f"    {'지점':<6} {'평균':>12} {'중앙값':>12} {'표준편차':>12} {'변동계수':>8}")
    for store, row in g.sort_values("mean", ascending=False).iterrows():
        print(f"    {store:<6} {row['mean']:>12,.0f} {row['median']:>12,.0f} "
              f"{row['std']:>12,.0f} {row['cv']:>8.2f}")


def outlier_experiment(df: pd.DataFrame) -> None:
    """극단값 1건이 표본 30건의 통계를 얼마나 움직이는지 실험합니다."""
    rng = np.random.default_rng(7)              # seed 고정
    gangnam = df[df["store"] == "강남점"]["revenue"].to_numpy()
    sample = rng.choice(gangnam, size=30, replace=False)

    def report(tag: str, values: np.ndarray) -> tuple[float, float, float]:
        m, md, sd = values.mean(), float(np.median(values)), values.std()
        print(f"    {tag:<24} 평균 {m:>12,.0f} | 중앙값 {md:>12,.0f} | 표준편차 {sd:>12,.0f}")
        return m, md, sd

    print()
    print("[4] 극단값 실험 — 강남점 매출 30건 표본에 대형 단체주문 1건 추가")
    base = report("원래 표본(30건)", sample)
    outlier = 500_000_000.0                     # 5억 원짜리 단체주문
    spiked = np.append(sample, outlier)
    after = report("극단값 추가(31건)", spiked)

    print()
    print("    값 '하나'가 만든 변화:")
    print(f"      평균     : {after[0] - base[0]:>+14,.0f} 원  (크게 끌려감)")
    print(f"      중앙값   : {after[1] - base[1]:>+14,.0f} 원  (거의 제자리 = 강건함)")
    print(f"      표준편차 : {after[2] - base[2]:>+14,.0f} 원  (제곱 계산이라 폭발)")
    print("    => 연봉·매출처럼 꼬리 긴 데이터의 보고서에는 중앙값을 함께 적으세요.")


def main() -> None:
    df = load_clean_sales()
    describe_revenue(df)
    by_store(df)
    outlier_experiment(df)


if __name__ == "__main__":
    main()
