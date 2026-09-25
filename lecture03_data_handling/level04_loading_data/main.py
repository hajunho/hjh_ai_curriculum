"""
파일에서 DataFrame 으로 데이터를 들여오는 실습입니다.
(1) 정상 CSV 저장/읽기, (2) 콤마 숫자·'-' 결측·세미콜론 구분자가 섞인
골칫덩이 CSV 를 read_csv 옵션으로 복구, (3) cp949 인코딩 오류 재현과 해결,
(4) records 형태 JSON 왕복까지 — 실무 통관 사고 4종을 코드로 해결합니다.
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
    print("[1] 정상 CSV — 저장하고 되읽기")
    rows = hjh_data.sales_table(n_days=30, seed=42)   # seed 고정
    df = pd.DataFrame(rows)
    csv_path = OUT / "sales_30d.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")   # index=False 잊지 않기!
    print(f"    저장: {csv_path} ({len(df)}행)")
    loaded = pd.read_csv(csv_path)
    print(f"    되읽기: {loaded.shape[0]}행 {loaded.shape[1]}열, dtype 요약:")
    print("    " + ", ".join(f"{c}={t}" for c, t in loaded.dtypes.items()))
    print("    -> 결측이 있는 revenue 는 float64 로 읽힙니다.")

    # ------------------------------------------------------------------
    print("\n[2] 골칫덩이 CSV — 콤마 숫자, '-' 결측, 세미콜론 구분자")
    messy_path = OUT / "messy.csv"
    with open(messy_path, "w", encoding="utf-8") as f:
        f.write("date;store;revenue\n")
        f.write("2025-01-01;강남점;1,234,000\n")
        f.write("2025-01-02;강남점;-\n")            # 결측을 '-' 로 표기한 시스템
        f.write("2025-01-03;홍대점;987,500\n")
        f.write("2025-01-04;부산점;1,050,000\n")
    print(f"    생성: {messy_path}")

    naive = pd.read_csv(messy_path)                  # 옵션 없이 읽으면?
    print(f"    옵션 없이 읽음 -> 열 개수 {naive.shape[1]}개 (세미콜론을 못 알아봐 한 덩어리!)")
    print(f"      columns = {list(naive.columns)}")

    fixed = pd.read_csv(messy_path, sep=";", thousands=",", na_values="-")
    print("    옵션 3개(sep=';', thousands=',', na_values='-') 적용 후:")
    print(fixed.to_string(index=False))
    print(f"      revenue dtype = {fixed['revenue'].dtype} -> 합계 {fixed['revenue'].sum():,.0f}원 계산 가능")

    # ------------------------------------------------------------------
    print("\n[3] 인코딩 사고 — cp949 파일을 utf-8 열쇠로 열면?")
    cp949_path = OUT / "sales_cp949.csv"
    df.head(5).to_csv(cp949_path, index=False, encoding="cp949")  # 옛 시스템/엑셀 흉내
    print(f"    생성: {cp949_path} (cp949 로 저장)")
    try:
        pd.read_csv(cp949_path, encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"    utf-8 로 읽기 -> UnicodeDecodeError 발생!")
        print(f"      메시지 일부: {str(e)[:70]}...")
    rescued = pd.read_csv(cp949_path, encoding="cp949")
    print("    encoding='cp949' 로 다시 읽으면 한글이 멀쩡합니다:")
    print(rescued[["date", "store", "category", "revenue"]].head(3).to_string(index=False))

    # ------------------------------------------------------------------
    print("\n[4] JSON — records 형태로 저장하고 되읽기")
    json_path = OUT / "sales_records.json"
    sample = df.head(3)[["date", "store", "category", "revenue"]]
    sample.to_json(json_path, orient="records", force_ascii=False)
    print(f"    저장: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        raw = f.read()
    print(f"    파일 내용 앞부분: {raw[:80]}...")
    from_json = pd.read_json(json_path, orient="records")
    print("    pd.read_json 으로 되읽은 표:")
    print(from_json.to_string(index=False))
    print("    -> 행 하나 = 딕셔너리 하나. 시스템 간 데이터 전달의 표준 형태입니다.")

    # 참고: 엑셀 파일은 openpyxl 설치 후 아래처럼 읽습니다 (여기서는 개념만).
    #   df = pd.read_excel("report.xlsx", sheet_name="1월", header=2)

    print("\n정리: 파일 문제는 원본을 고치지 말고 read_csv 옵션으로 해결하세요.")
    print("      sep / encoding / thousands / na_values 네 가지면 사고의 90%가 끝납니다.")


if __name__ == "__main__":
    main()
