"""
A three-way categorical-encoding comparison.
Trains a Ridge regression that predicts revenue from sales_table's
store/category/weekday (text) columns using one-hot / ordinal / target
encoding, and compares 'number of columns vs performance'.
Target encoding computes its means from the training data only, to prevent leakage.
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
    """Measure performance with the split and model fixed, swapping only the encoding output."""
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    print(f"    {label:<12} {n_cols:2d} cols | MAE {mae:10,.0f} KRW | R2 {r2:.3f}")
    return {"method": label, "cols": n_cols, "MAE": round(mae), "R2": round(r2, 3)}


def main() -> None:
    print("=" * 62)
    print(" Categorical encoding comparison: one-hot vs ordinal vs target")
    print("=" * 62)

    # [1] Clean the data and check the categories -------------------------
    df = pd.DataFrame(hjh_data.sales_table(n_days=365, seed=42))
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)
    print(f"\n[1] Cleaning: {n0} rows -> {len(df)} rows (dropped missing/negative revenue)")
    for c in CAT_COLS:
        print(f"    {c:<9}: {df[c].nunique()} categories e.g. {sorted(df[c].unique())[:3]} ...")

    y = df[TARGET]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.3, random_state=42)
    results = []

    # [2] One-hot encoding --------------------------------------------------
    print("\n[2] One-hot: a checkbox column per category (no rank distortion, more columns)")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df.loc[tr_idx, CAT_COLS])  # fit on the training data only!
    ohe_cols = list(ohe.get_feature_names_out(CAT_COLS))
    X_tr = pd.DataFrame(ohe.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=ohe_cols)
    X_te = pd.DataFrame(ohe.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=ohe_cols)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "one-hot", X_tr.shape[1]))

    # [3] Ordinal encoding ----------------------------------------------------
    print("\n[3] Ordinal: category -> number. Few columns, but a fake ranking appears")
    orde = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    orde.fit(df.loc[tr_idx, CAT_COLS])
    X_tr = pd.DataFrame(orde.transform(df.loc[tr_idx, CAT_COLS]),
                        index=tr_idx, columns=CAT_COLS)
    X_te = pd.DataFrame(orde.transform(df.loc[te_idx, CAT_COLS]),
                        index=te_idx, columns=CAT_COLS)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "ordinal", X_tr.shape[1]))
    print("      (the linear model believes the lie that 'store #3 is three times store #1')")

    # [4] Target encoding ------------------------------------------------------
    print("\n[4] Target: category -> group-mean revenue from the TRAINING data (leakage hazard!)")
    global_mean = y[tr_idx].mean()
    X_tr = pd.DataFrame(index=tr_idx)
    X_te = pd.DataFrame(index=te_idx)
    for c in CAT_COLS:
        # Means must be computed from training data only; unseen categories get the global mean
        means = df.loc[tr_idx].groupby(c)[TARGET].mean()
        X_tr[c + "_te"] = df.loc[tr_idx, c].map(means)
        X_te[c + "_te"] = df.loc[te_idx, c].map(means).fillna(global_mean)
    X_tr[NUM_COLS], X_te[NUM_COLS] = df.loc[tr_idx, NUM_COLS], df.loc[te_idx, NUM_COLS]
    results.append(fit_eval(X_tr, X_te, y[tr_idx], y[te_idx], "target", X_tr.shape[1]))

    # [5] Summary ----------------------------------------------------------------
    print("\n[5] Summary: columns vs performance")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    best = summary.loc[summary["R2"].idxmax(), "method"]
    print(f"\n    Winner on this data: {best}")
    print("    How to read it:")
    print("      - ordinal's R2 drop = the price of numbering unordered categories (linear model)")
    print("      - target encoding = close to one-hot with 3 columns -> a strong choice when categories number in the thousands (high cardinality)")
    print("      - but the moment target encoding averages over the FULL dataset, it becomes a leakage accident.")
    print("\n    Lesson: encoding is not a technical choice — it is choosing 'which worldview to give the model'.")


if __name__ == "__main__":
    main()
