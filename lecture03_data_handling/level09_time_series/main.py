"""
시계열(time series) 데이터 다루기 실습.
카페 체인 매출을 시간 축 위에 올려서
  - 잘못된 날짜 문자열 진단 (to_datetime errors="coerce")
  - 일별 -> 주별/월별 리샘플링 (resample)
  - 7일 이동평균 (rolling) 과 전주 대비 성장률 (shift / pct_change)
을 차례로 실행하고, 추세 그래프를 outputs/ 에 저장합니다.
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 그림을 저장합니다.
import matplotlib.pyplot as plt
import pandas as pd

# 공용 데이터 모듈(hjh_data)을 불러오기 위한 경로 설정
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"


def load_clean_sales() -> pd.DataFrame:
    """매출 데이터를 만들고 결측·음수 오염을 제거합니다 (level06 복습)."""
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    before = len(df)
    df = df.dropna(subset=["revenue"])          # 결측 매출 제거
    df = df[df["revenue"] > 0].copy()           # 음수 이상치 제거
    print(f"    정제: {before:,}행 -> {len(df):,}행 (결측·음수 {before - len(df)}건 제거)")
    return df


def step1_diagnose_dates(df: pd.DataFrame) -> pd.DataFrame:
    """[1] 날짜 문자열 오염을 진단하고, 믿을 수 있는 날짜를 새로 만듭니다."""
    print("\n[1] 날짜 오염 진단 — 문자열은 아직 날짜가 아닙니다")
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad = int(parsed.isna().sum())
    bad_examples = df.loc[parsed.isna(), "date"].unique()[:5]
    print(f"    errors='coerce' 변환 결과: NaT(날짜 결측) {n_bad:,}건 발생")
    print(f"    달력에 없는 날짜 예시: {list(bad_examples)}")
    print("    -> 이 데이터의 date 컬럼은 '한 달=31일' 가정으로 찍혀 있어 신뢰할 수 없습니다.")

    # 신뢰할 수 있는 day_index(개장 후 경과일)로 진짜 날짜를 재구성합니다.
    df = df.copy()
    df["real_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["day_index"], unit="D")
    print(f"    day_index 로 재구성한 기간: {df['real_date'].min().date()} ~ {df['real_date'].max().date()}")
    return df


def step2_daily_series(df: pd.DataFrame) -> pd.Series:
    """[2] 전사 일별 매출 시계열을 만들고 DatetimeIndex 를 확인합니다."""
    print("\n[2] 일별 전사 매출 시계열 만들기")
    daily = df.groupby("real_date")["revenue"].sum().sort_index()
    print(f"    일수: {len(daily)}일 / 인덱스 타입: {type(daily.index).__name__}")
    print(f"    첫 3일:\n{(daily.head(3) / 1e6).round(1).to_string()}  (단위: 백만원)")
    # dt 접근자: 날짜 컬럼에서 '월' 같은 부품을 꺼낼 수 있습니다.
    month_of_first_rows = df["real_date"].dt.month.head(3).tolist()
    print(f"    dt 접근자 예시 — 앞 3행의 월: {month_of_first_rows}")
    return daily


def step3_resample(daily: pd.Series) -> pd.Series:
    """[3] resample: 일별 -> 주별 -> 월별로 시간 단위를 바꿉니다."""
    print("\n[3] resample — 일별 데이터를 주간·월간 상자에 담아 합산")
    weekly = daily.resample("W").sum()
    monthly = daily.resample("ME").sum()
    print(f"    주간 합계: {len(weekly)}개 주 (첫 주는 부분 주라 값이 작을 수 있음)")
    print(f"    월간 합계 (단위: 억원):")
    for ts, val in monthly.items():
        print(f"      {ts.strftime('%Y-%m')}: {val / 1e8:6.2f}억")
    return weekly


def step4_rolling(daily: pd.Series) -> pd.Series:
    """[4] rolling(7): 7일 이동평균으로 요일 출렁임을 지웁니다."""
    print("\n[4] rolling(7) — 이동평균으로 소음 제거")
    ma7 = daily.rolling(7).mean()
    print(f"    처음 6일은 창을 못 채워 NaN: 앞쪽 NaN 개수 = {int(ma7.isna().sum())}")
    sample_day = daily.index[9]
    print(f"    예시) {sample_day.date()} 원본 {daily.iloc[9]/1e6:.1f}백만원 "
          f"vs 7일 평균 {ma7.iloc[9]/1e6:.1f}백만원")
    weekend_std = daily.std()
    smooth_std = ma7.dropna().std()
    print(f"    표준편차 비교: 원본 {weekend_std/1e6:.1f} -> 이동평균 {smooth_std/1e6:.1f} (백만원)"
          f" — 출렁임이 줄었습니다")
    return ma7


def step5_growth(weekly: pd.Series) -> None:
    """[5] shift / pct_change: 전주 대비 성장률을 구합니다."""
    print("\n[5] 전주 대비 성장률 — shift 와 pct_change")
    # 양 끝의 부분 주(7일을 못 채운 주)는 성장률을 왜곡하므로 잘라 냅니다.
    full_weeks = weekly.iloc[1:-1]
    growth = full_weeks.pct_change() * 100
    # shift 로 같은 계산을 직접 재현해 검증합니다.
    manual = (full_weeks - full_weeks.shift(1)) / full_weeks.shift(1) * 100
    assert ((growth - manual).abs().dropna() < 1e-9).all(), "pct_change 와 shift 계산이 달라요"
    print("    검증: pct_change == (이번주-지난주)/지난주  (shift 로 재현, 일치)")
    top = growth.nlargest(3)
    bottom = growth.nsmallest(3)
    print("    성장률 상위 3주:")
    for ts, val in top.items():
        print(f"      {ts.date()} 마감 주: {val:+.1f}%")
    print("    성장률 하위 3주:")
    for ts, val in bottom.items():
        print(f"      {ts.date()} 마감 주: {val:+.1f}%")


def step6_plot(daily: pd.Series, ma7: pd.Series) -> None:
    """[6] 일별 매출 + 7일 이동평균 그래프를 PNG 로 저장합니다."""
    print("\n[6] 그래프 저장")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    # 한글 폰트가 없는 환경에서도 깨지지 않도록 그래프 안 글자는 영어를 씁니다.
    ax.plot(daily.index, daily.values / 1e6, color="#9ecae1", linewidth=0.8,
            label="daily revenue")
    ax.plot(ma7.index, ma7.values / 1e6, color="#08519c", linewidth=2.0,
            label="7-day moving average")
    ax.set_title("Cafe chain daily revenue (2025)")
    ax.set_ylabel("revenue (million KRW)")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "daily_trend.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    저장 완료: {out_path}")


def main() -> None:
    print("=" * 60)
    print("Level 09 — 시계열 데이터 다루기")
    print("=" * 60)
    df = load_clean_sales()
    df = step1_diagnose_dates(df)
    daily = step2_daily_series(df)
    weekly = step3_resample(daily)
    ma7 = step4_rolling(daily)
    step5_growth(weekly)
    step6_plot(daily, ma7)
    print("\n완료! 음료 성수기(봄철 상승 곡선)가 이동평균에 드러나는지 그래프로 확인해 보세요.")


if __name__ == "__main__":
    main()
