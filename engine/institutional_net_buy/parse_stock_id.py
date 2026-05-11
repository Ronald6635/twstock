import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="從 CSV 檔案提取 stock_id 並儲存至文字檔")
    parser.add_argument("--input-csv", type=str, default="trend_candidates_2026-02-09_2026-05-08.csv", help="輸入的 CSV 檔案路徑")
    parser.add_argument("--output-txt", type=str, default="targets.txt", help="輸出的目標 TXT 檔案路徑")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_txt)

    if not input_path.exists():
        print(f"錯誤：找不到輸入檔案 '{input_path}'")
        return

    print(f"讀取檔案: {input_path} ...")
    # 強制將 stock_id 讀取為字串，避免遺失前面的 0
    df = pd.read_csv(input_path, dtype={"stock_id": str})

    if "stock_id" not in df.columns:
        print("錯誤：CSV 中找不到 'stock_id' 欄位")
        return

    # 取得所有的 stock_id
    stock_ids = df["stock_id"].dropna().astype(str).tolist()

    print(f"寫入檔案: {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for sid in stock_ids:
            f.write(f"{sid}\n")

    print(f"成功提取並寫入 {len(stock_ids)} 筆 stock_id 至 {output_path}")

if __name__ == "__main__":
    main()
