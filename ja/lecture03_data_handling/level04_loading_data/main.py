"""
ファイルから DataFrame へデータを持ち込む実習です。
(1) 正常な CSV の保存/読み込み、(2) カンマ数字・「-」の欠損・セミコロン区切りが混ざった
厄介者の CSV を read_csv のオプションで復旧、(3) cp932 エンコーディングエラーの再現と解決、
(4) records 形式の JSON の往復まで — 実務の通関事故4種をコードで解決します。
"""

import os
import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parent
OUT = BASE / "outputs"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 110)

    # ------------------------------------------------------------------
    print("[1] 正常な CSV — 保存して読み直す")
    rows = hjh_data.sales_table(n_days=30, seed=42)   # seed 固定
    df = pd.DataFrame(rows)
    csv_path = OUT / "sales_30d.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")   # index=False を忘れずに!
    print(f"    保存: {csv_path} ({len(df)}行)")
    loaded = pd.read_csv(csv_path)
    print(f"    読み直し: {loaded.shape[0]}行 {loaded.shape[1]}列, dtype の要約:")
    print("    " + ", ".join(f"{c}={t}" for c, t in loaded.dtypes.items()))
    print("    -> 欠損のある revenue は float64 として読まれます。")

    # ------------------------------------------------------------------
    print("\n[2] 厄介者の CSV — カンマ数字、'-' の欠損、セミコロン区切り")
    messy_path = OUT / "messy.csv"
    with open(messy_path, "w", encoding="utf-8") as f:
        f.write("date;store;revenue\n")
        f.write("2025-01-01;渋谷店;1,234,000\n")
        f.write("2025-01-02;渋谷店;-\n")            # 欠損を「-」と表記するシステム
        f.write("2025-01-03;新宿店;987,500\n")
        f.write("2025-01-04;名古屋店;1,050,000\n")
    print(f"    生成: {messy_path}")

    naive = pd.read_csv(messy_path)                  # オプションなしで読むと?
    print(f"    オプションなしで読む -> 列の数 {naive.shape[1]}個 (セミコロンを認識できずひと塊に!)")
    print(f"      columns = {list(naive.columns)}")

    fixed = pd.read_csv(messy_path, sep=";", thousands=",", na_values="-")
    print("    オプション3つ (sep=';', thousands=',', na_values='-') の適用後:")
    print(fixed.to_string(index=False))
    print(f"      revenue dtype = {fixed['revenue'].dtype} -> 合計 {fixed['revenue'].sum():,.0f}ウォンの計算が可能")

    # ------------------------------------------------------------------
    print("\n[3] エンコーディング事故 — cp932 のファイルを utf-8 の鍵で開くと?")
    cp949_path = OUT / "sales_cp949.csv"
    df.head(5).to_csv(cp949_path, index=False, encoding="cp932")  # 古いシステム/Excel の真似
    print(f"    生成: {cp949_path} (cp932 で保存)")
    try:
        pd.read_csv(cp949_path, encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"    utf-8 で読む -> UnicodeDecodeError が発生!")
        print(f"      メッセージの一部: {str(e)[:70]}...")
    rescued = pd.read_csv(cp949_path, encoding="cp932")
    print("    encoding='cp932' で読み直すと、日本語は無事です:")
    print(rescued[["date", "store", "category", "revenue"]].head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[4] JSON — records 形式で保存して読み直す")
    json_path = OUT / "sales_records.json"
    sample = df.head(3)[["date", "store", "category", "revenue"]]
    sample.to_json(json_path, orient="records", force_ascii=False)
    print(f"    保存: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        raw = f.read()
    print(f"    ファイル内容の先頭部分: {raw[:80]}...")
    from_json = pd.read_json(json_path, orient="records")
    print("    pd.read_json で読み直した表:")
    print(from_json.to_string(index=False))
    print("    -> 行1つ = 辞書1つ。システム間のデータ受け渡しの標準形です。")

    # 参考: Excel ファイルは openpyxl をインストールした後、以下のように読みます (ここでは概念のみ)。
    #   df = pd.read_excel("report.xlsx", sheet_name="1月", header=2)

    print("\nまとめ: ファイルの問題は原本を直さず、read_csv のオプションで解決しましょう。")
    print("      sep / encoding / thousands / na_values の4つで、事故の90%は片付きます。")


if __name__ == "__main__":
    main()
