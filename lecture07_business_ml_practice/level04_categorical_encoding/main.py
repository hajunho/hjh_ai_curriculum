"""
범주형 인코딩 3종 비교 실험.
sales_table 의 지점·카테고리·요일(문자)로 매출을 예측하는 Ridge 회귀를
원핫 / 순서형 / 타깃 인코딩으로 각각 훈련해 '열 개수 vs 성능'을 비교합니다.
타깃 인코딩은 누설 방지를 위해 학습 데이터에서만 평균을 계산합니다.
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score

CAT_COLS = ["store", "category", "weekday"]
NUM_COLS = ["ad_cost"]
TARGET = "revenue"


def fit_eval(X_tr, X_te, y_tr, y_te, label: str, n_cols: int):
    """분할·모델을 고정한 채 인코딩 결과만 바꿔 성능을 잰다."""
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"    {label:<12} 열 {n_cols:2d}개 | MAE {mae:10,.0f}원 | R2 {r2:.3f}")
    return {"방법": label, "열": n_cols, "MAE": round(mae), "R2": round(r2, 3)}


def main() -> None:
    print("=" * 62)
    print(" 범주형 인코딩 비교: 원핫 vs 순서형 vs 타깃")
    print("=" * 62)

    # [1] 데이터 정제와 범주 확인 ---------------------------------------
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)
    print(f"\n[1] 정제: {n0}행 -> {len(df)}행 (결측·음수 매출 제거)")
    for c in CAT_COLS:
        print(f"    {c:<9}: 범주 {df[c].nunique()}개 예) {sorted(df[c].unique())[:3]} ...")

    y = df[TARGET]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.3, random_state=42)
    results = []

    # [2] 원핫 인코딩 ----------------------------------------------------
    print("\n[2] 원핫: 범주마다 체크박스 열 생성 (서열 왜곡 없음, 열 증가)")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df.loc[tr_idx, CAT_COLS])  # fit 은 학습 데이터에만!
    ohe_cols = list(ohe.get_feature_names_out(CAT_COLS))
    X_tr = pd.DataFrame(ohe.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=ohe_cols)
    X_te = pd.DataFrame(ohe.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=ohe_cols)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "원핫", X_tr.shape[1]))

    # [3] 순서형 인코딩 --------------------------------------------------
    print("\n[3] 순서형: 범주 -> 번호. 열은 적지만 가짜 서열이 생김")
    orde = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    orde.fit(df.loc[tr_idx, CAT_COLS])
    X_tr = pd.DataFrame(orde.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=CAT_COLS)
    X_te = pd.DataFrame(orde.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=CAT_COLS)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "순서형", X_tr.shape[1]))
    print("      ('부산점=3 은 강남점=1 의 3배' 라는 거짓 정보를 선형 모델이 믿습니다)")

    # [4] 타깃 인코딩 ----------------------------------------------------
    print("\n[4] 타깃: 범주 -> '학습 데이터'의 그룹 평균 매출 (누설 주의!)")
    global_mean = y[tr_idx].mean()
    X_tr = pd.DataFrame(index=tr_idx)
    X_te = pd.DataFrame(index=te_idx)
    for c in CAT_COLS:
        # 평균은 반드시 학습 데이터에서만 계산, 미등장 범주는 전체 평균으로
        means = df.loc[tr_idx].groupby(c)[TARGET].mean()
        X_tr[c + "_te"] = df.loc[tr_idx, c].map(means)
        X_te[c + "_te"] = df.loc[te_idx, c].map(means).fillna(global_mean)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "타깃", X_tr.shape[1]))

    # [5] 요약 ------------------------------------------------------------
    print("\n[5] 요약: 열 개수 vs 성능")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    best = summary.loc[summary["R2"].idxmax(), "방법"]
    print(f"\n    이번 데이터의 승자: {best}")
    print("    읽는 법:")
    print("      - 순서형의 R2 하락 = 무서열 범주에 번호를 붙인 대가 (선형 모델)")
    print("      - 타깃 인코딩 = 열 3개로 원핫에 근접 -> 범주가 수천 개(고카디널리티)면 유력한 선택")
    print("      - 단, 타깃 인코딩은 전체 데이터로 평균을 내는 순간 누설 사고가 됩니다.")
    print("\n    교훈: 인코딩은 기술 선택이 아니라 '모델에게 어떤 세계관을 줄 것인가'의 선택입니다.")


if __name__ == "__main__":
    main()
