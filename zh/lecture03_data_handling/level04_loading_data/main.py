"""
把数据从文件导入 DataFrame 的实战。
(1) 正常 CSV 的保存/读取，(2) 用 read_csv 参数抢救混了千位逗号、
'-' 缺失标记和分号分隔符的麻烦 CSV，(3) 复现并解决 gbk 编码报错，
(4) records 形态 JSON 的来回 — 用代码解决 4 类实务"通关事故"。
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
    print("[1] 正常 CSV — 保存再读回来")
    rows = hjh_data.sales_table(n_days=30, seed=42)   # 固定 seed
    df = pd.DataFrame(rows)
    csv_path = OUT / "sales_30d.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")   # 别忘了 index=False!
    print(f"    保存: {csv_path} ({len(df)} 行)")
    loaded = pd.read_csv(csv_path)
    print(f"    读回: {loaded.shape[0]} 行 {loaded.shape[1]} 列，dtype 摘要:")
    print("    " + ", ".join(f"{c}={t}" for c, t in loaded.dtypes.items()))
    print("    -> 有缺失的 revenue 会被读成 float64。")

    # ------------------------------------------------------------------
    print("\n[2] 麻烦 CSV — 千位逗号数字、'-' 缺失标记、分号分隔符")
    messy_path = OUT / "messy.csv"
    with open(messy_path, "w", encoding="utf-8") as f:
        f.write("date;store;revenue\n")
        f.write("2025-01-01;朝阳店;1,234,000\n")
        f.write("2025-01-02;朝阳店;-\n")            # 把缺失写成 '-' 的那种系统
        f.write("2025-01-03;海淀店;987,500\n")
        f.write("2025-01-04;浦东店;1,050,000\n")
    print(f"    生成: {messy_path}")

    naive = pd.read_csv(messy_path)                  # 不加参数直接读会怎样?
    print(f"    不加参数读 -> 列数 {naive.shape[1]} 个 (认不出分号，全挤成一坨!)")
    print(f"      columns = {list(naive.columns)}")

    fixed = pd.read_csv(messy_path, sep=";", thousands=",", na_values="-")
    print("    加上 3 个参数(sep=';', thousands=',', na_values='-') 之后:")
    print(fixed.to_string(index=False))
    print(f"      revenue dtype = {fixed['revenue'].dtype} -> 合计 {fixed['revenue'].sum():,.0f} 韩元，算得出来了")

    # ------------------------------------------------------------------
    print("\n[3] 编码事故 — 用 utf-8 这把钥匙去开 gbk 的锁会怎样?")
    gbk_path = OUT / "sales_gbk.csv"
    df.head(5).to_csv(gbk_path, index=False, encoding="gbk")  # 模仿老 POS 系统导出
    print(f"    生成: {gbk_path} (以 gbk 保存)")
    try:
        pd.read_csv(gbk_path, encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"    用 utf-8 读 -> 抛出 UnicodeDecodeError!")
        print(f"      消息片段: {str(e)[:70]}...")
    rescued = pd.read_csv(gbk_path, encoding="gbk")
    print("    换成 encoding='gbk' 再读，中文就完好无损:")
    print(rescued[["date", "store", "category", "revenue"]].head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[4] JSON — 以 records 形态保存再读回来")
    json_path = OUT / "sales_records.json"
    sample = df.head(3)[["date", "store", "category", "revenue"]]
    sample.to_json(json_path, orient="records", force_ascii=False)
    print(f"    保存: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        raw = f.read()
    print(f"    文件内容开头: {raw[:80]}...")
    from_json = pd.read_json(json_path, orient="records")
    print("    用 pd.read_json 读回来的表:")
    print(from_json.to_string(index=False))
    print("    -> 一行 = 一个字典。这是系统之间传数据的标准形态。")

    # 参考: Excel 文件在装好 openpyxl 后按下面这样读 (这里只讲概念)。
    #   df = pd.read_excel("report.xlsx", sheet_name="1月", header=2)

    print("\n小结: 文件的毛病别去改原件，用 read_csv 的参数解决。")
    print("      sep / encoding / thousands / na_values 这四个就能搞定 90% 的事故。")


if __name__ == "__main__":
    main()
