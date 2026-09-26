"""
カテゴリ変数エンコーディング 3 種の比較実験。
sales_table の店舗・カテゴリ・曜日(文字)で売上を予測する Ridge 回帰を
ワンホット / 順序 / ターゲット エンコーディングでそれぞれ訓練し、
「列の数 vs 性能」を比較します。
ターゲットエンコーディングはリーク防止のため学習データのみで平均を計算します。
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
    """分割・モデルを固定したままエンコーディング結果だけ変えて性能を測る。"""
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"    {label:<12} 列 {n_cols:2d}個 | MAE {mae:10,.0f}ウォン | R2 {r2:.3f}")
    return {"方法": label, "列": n_cols, "MAE": round(mae), "R2": round(r2, 3)}


def main() -> None:
    print("=" * 62)
    print(" カテゴリ変数エンコーディング比較: ワンホット vs 順序 vs ターゲット")
    print("=" * 62)

    # [1] データ整形とカテゴリ確認 ---------------------------------------
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)
    print(f"\n[1] 整形: {n0}行 -> {len(df)}行 (欠損・マイナス売上を除去)")
    for c in CAT_COLS:
        print(f"    {c:<9}: カテゴリ {df[c].nunique()}個 例) {sorted(df[c].unique())[:3]} ...")

    y = df[TARGET]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.3, random_state=42)
    results = []

    # [2] ワンホットエンコーディング ----------------------------------------------------
    print("\n[2] ワンホット: カテゴリごとにチェックボックス列を生成 (序列の歪みなし、列は増える)")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df.loc[tr_idx, CAT_COLS])  # fit は学習データにだけ!
    ohe_cols = list(ohe.get_feature_names_out(CAT_COLS))
    X_tr = pd.DataFrame(ohe.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=ohe_cols)
    X_te = pd.DataFrame(ohe.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=ohe_cols)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "ワンホット", X_tr.shape[1]))

    # [3] 順序エンコーディング --------------------------------------------------
    print("\n[3] 順序: カテゴリ -> 番号。列は少ないが偽の序列が生まれる")
    orde = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    orde.fit(df.loc[tr_idx, CAT_COLS])
    X_tr = pd.DataFrame(orde.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=CAT_COLS)
    X_te = pd.DataFrame(orde.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=CAT_COLS)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "順序", X_tr.shape[1]))
    print("      (「渋谷店=3 は大阪店=1 の3倍」という嘘の情報を線形モデルは信じます)")

    # [4] ターゲットエンコーディング ----------------------------------------------------
    print("\n[4] ターゲット: カテゴリ -> 「学習データ」のグループ平均売上 (リーク注意!)")
    global_mean = y[tr_idx].mean()
    X_tr = pd.DataFrame(index=tr_idx)
    X_te = pd.DataFrame(index=te_idx)
    for c in CAT_COLS:
        # 平均は必ず学習データからのみ計算、未登場カテゴリは全体平均で
        means = df.loc[tr_idx].groupby(c)[TARGET].mean()
        X_tr[c + "_te"] = df.loc[tr_idx, c].map(means)
        X_te[c + "_te"] = df.loc[te_idx, c].map(means).fillna(global_mean)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "ターゲット", X_tr.shape[1]))

    # [5] まとめ ------------------------------------------------------------
    print("\n[5] まとめ: 列の数 vs 性能")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    best = summary.loc[summary["R2"].idxmax(), "方法"]
    print(f"\n    今回のデータの勝者: {best}")
    print("    読み方:")
    print("      - 順序の R2 低下 = 無序列カテゴリに番号を振った代償 (線形モデル)")
    print("      - ターゲットエンコーディング = 列3つでワンホットに肉薄 -> カテゴリが数千個(高カーディナリティ)なら有力な選択")
    print("      - ただしターゲットエンコーディングは、全データで平均を取った瞬間にリーク事故になります。")
    print("\n    教訓: エンコーディングは技術の選択ではなく、「モデルにどんな世界観を与えるか」の選択です。")


if __name__ == "__main__":
    main()
