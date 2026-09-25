"""
결측치·이상치 처리 실습.
카페 체인 매출 데이터의 오염(결측, 음수)을 진단하고,
도메인 규칙과 IQR로 이상치를 찾은 뒤,
dropna / 전체 중앙값 대체 / 그룹별 중앙값 대체 세 전략을
통계량으로 비교해 어떤 전략이 언제 적절한지 확인합니다.
"""

import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 그림 저장 가능하게
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def diagnose(df: pd.DataFrame) -> None:
    """오염 상태를 진단해 출력합니다."""
    n = len(df)
    n_missing = int(df["revenue"].isna().sum())
    n_negative = int((df["revenue"] < 0).sum())  # NaN 은 비교에서 자동 제외
    print(f"  전체 행 수        : {n:,}")
    print(f"  결측(NaN) 개수    : {n_missing:,}건 ({n_missing / n:.2%})")
    print(f"  음수 매출 개수    : {n_negative:,}건 ({n_negative / n:.2%})")
    desc = df["revenue"].describe()
    print(f"  describe() 요약   : min={desc['min']:,.0f}  "
          f"median={desc['50%']:,.0f}  max={desc['max']:,.0f}")
    if desc["min"] < 0:
        print("  -> min 이 음수! 매출 데이터로서 말이 안 되는 값이 섞여 있습니다.")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    print("[1] 오염 진단 — 세어 보기 전에는 아무것도 모릅니다")
    rows = hjh_data.sales_table(n_days=365, seed=42)  # seed 고정: 항상 같은 데이터
    df = pd.DataFrame(rows)
    diagnose(df)

    # ------------------------------------------------------------------
    print("\n[2] 도메인 규칙 적용 — '매출은 음수일 수 없다'")
    df_rule = df.copy()  # 원본은 증거물로 보존
    neg_mask = df_rule["revenue"] < 0
    print(f"  음수 {int(neg_mask.sum())}건을 삭제하지 않고 NaN(값 미상)으로 표시합니다.")
    df_rule.loc[neg_mask, "revenue"] = np.nan
    print(f"  처리 후 결측 개수 : {int(df_rule['revenue'].isna().sum()):,}건"
          f" (원래 결측 + 음수였던 값)")

    # ------------------------------------------------------------------
    print("\n[3] IQR 방법 — 통계적으로 수상한 값 찾기")
    rev = df_rule["revenue"].dropna()
    q1, q3 = rev.quantile(0.25), rev.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((rev < low) | (rev > high)).sum())
    print(f"  Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
    print(f"  정상 범위: [{low:,.0f}, {high:,.0f}]")
    print(f"  범위 밖 이상치 후보: {n_out}건 ({n_out / len(rev):.2%})")
    print("  -> 주말·성수기 매출이 섞여 있을 수 있으므로 '후보'일 뿐, 자동 삭제는 금물입니다.")

    # ------------------------------------------------------------------
    print("\n[4] 처리 전략 비교 — 같은 데이터, 다른 숫자")
    # (a) 결측 행 삭제
    df_a = df_rule.dropna(subset=["revenue"])
    # (b) 전체 중앙값으로 대체
    overall_median = df_rule["revenue"].median()
    df_b = df_rule.copy()
    df_b["revenue"] = df_b["revenue"].fillna(overall_median)
    # (c) 지점·카테고리별 중앙값으로 대체 (transform 은 행 수를 유지)
    group_median = df_rule.groupby(["store", "category"])["revenue"].transform("median")
    df_c = df_rule.copy()
    df_c["revenue"] = df_c["revenue"].fillna(group_median)

    report = pd.DataFrame({
        "행 수": [len(df_a), len(df_b), len(df_c)],
        "총매출(백만원)": [df_a["revenue"].sum() / 1e6,
                       df_b["revenue"].sum() / 1e6,
                       df_c["revenue"].sum() / 1e6],
        "평균 매출": [df_a["revenue"].mean(),
                   df_b["revenue"].mean(),
                   df_c["revenue"].mean()],
    }, index=["(a) dropna", "(b) 전체 중앙값", "(c) 그룹별 중앙값"])
    print(report.round(0).to_string())
    print("  해석: (a)는 행이 줄어 총매출이 가장 작게 나옵니다. 합계 보고서라면 위험!")
    print("        (b)와 (c)는 행을 보존하지만, (c)가 지점·품목 특성을 반영해 더 정교합니다.")

    # ------------------------------------------------------------------
    print("\n[5] 처리 전/후 요약 + 상자그림 저장")
    before = df["revenue"].describe()
    after = df_c["revenue"].describe()
    summary = pd.DataFrame({"처리 전": before, "처리 후(c)": after}).round(0)
    print(summary.loc[["count", "mean", "min", "50%", "max"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(df["revenue"].dropna())
    axes[0].set_title("before (with negatives)")
    axes[1].boxplot(df_c["revenue"])
    axes[1].set_title("after (rule + group median)")
    for ax in axes:
        ax.set_ylabel("revenue (KRW)")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "boxplot_before_after.png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    print(f"  상자그림 저장: {png_path}")
    print("\n정리: 진단 -> 원인 추정 -> 규칙 적용 -> 전략 비교. 순서가 품질을 만듭니다.")


if __name__ == "__main__":
    main()
