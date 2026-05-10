"""
# ==========================================
# 1. 程式說明
# ==========================================
這個程式的主要功能是從 FinMind API 獲取指定股票在近三個月內的法人買賣超資料，並將結果整合成一個 DataFrame，最後輸出到 CSV 檔案中。使用者需要在同一個資料夾下建立一個名為 `targets.txt` 的純文字檔，並在其中每行輸入一個股票代碼。
# ==========================================
# 2. 程式架構
# ==========================================
1. 初始化與參數設定：載入環境變數、設定 API token、定義查詢的日期範圍和目標股票清單檔案名稱。
2. 定義讀取外部 TXT 清單的函數：從 `targets.txt` 讀取股票代碼，並返回一個列表。
3. 定義資料獲取與整合函數：對每個股票代碼呼叫 FinMind API 獲取法人買賣超資料，整合成一個 DataFrame，並進行必要的資料清洗與轉換。
4. 主程式執行區：讀取股票清單，獲取資料，並輸出結果到 CSV 檔案。
# ==========================================
# 3. 注意事項
# ==========================================
- 請確保已經安裝了 `FinMind` 和 `pandas` 等必要的 Python 套件。
- 請在同一個資料夾下建立 `targets.txt`，並在其中每行輸入一個股票代碼，例如：
```
2330
2317
2454
```
- API 的免費版本可能有頻率限制，程式中已加入暫停以避免觸發限制，但如果有大量股票，可能需要調整暫停時間。
- 輸出的 CSV 檔案會包含日期、股票代碼、法人名稱、買超、賣超和淨買超等欄位，方便後續分析使用。
# =========================================="""
import os
from dotenv import load_dotenv
import pandas as pd
from FinMind.data import DataLoader
import time
from datetime import datetime

load_dotenv()

# ==========================================
# 1. 初始化與參數設定
# ==========================================
token = os.getenv("FINMIND_API_KEY")
if not token:
    raise ValueError("請設定 FINMIND_API_KEY")

api = DataLoader()
api.login_by_token(api_token=token)

target_end_date = "2026-05-08"  # 設定查詢的結束交易日
target_start_date = (
    pd.Timestamp(target_end_date) - pd.Timedelta(days=90)
).strftime("%Y-%m-%d")  # 近三個月起始日
target_file = "targets.txt"  # 設定存放股票清單的文字檔名稱

# ==========================================
# 2. 定義讀取外部 TXT 清單的函數
# ==========================================
def load_target_stocks(filename):
    """
    從指定的純文字檔讀取股票代碼，每行一個代碼。
    """
    # 檢查檔案是否存在
    if not os.path.exists(filename):
        print(f"⚠️ 錯誤：找不到名為 '{filename}' 的檔案。")
        print("請在同一個資料夾下建立該檔案，並在每一行輸入一個股票代碼。")
        return []
    
    # 開啟檔案並讀取內容 (設定 encoding='utf-8' 確保相容性)
    with open(filename, 'r', encoding='utf-8') as file:
        # 讀取每一行，去除前後空白與換行符號 (.strip())，並過濾掉空行
        stocks = [line.strip() for line in file if line.strip()]
        
    print(f"成功從 {filename} 讀取了 {len(stocks)} 檔目標股票。\n")
    return stocks

# ==========================================
# 3. 定義資料獲取與整合函數
# ==========================================
def get_smart_money_consensus(stock_list, start_date, end_date):
    all_data = []
    
    for stock_id in stock_list:
        try:
            df = api.taiwan_stock_institutional_investors(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                price_df = api.taiwan_stock_daily(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                if price_df is not None and not price_df.empty and "close" in price_df.columns:
                    price_df = price_df[["date", "close"]].copy()
                    price_df["date"] = price_df["date"].astype(str)
                    df["date"] = df["date"].astype(str)
                    df = df.merge(price_df, on="date", how="left")
                else:
                    df["close"] = float("nan")
                    
                all_data.append(df)
            
            time.sleep(0.5)  # 暫停 0.5 秒，避免觸發免費 API 頻率限制
            
        except Exception as e:
            print(f"獲取 {stock_id} 時發生錯誤: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    # 合併所有股票的資料
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if combined_df.empty:
        return pd.DataFrame()
        
    # 將法人名稱翻成繁體中文
    name_map = {
        'Foreign_Investor': '外資',
        'ForeignInvestor': '外資',
        'Investment_Trust': '投信',
        'Dealer_self': '自營商',
        'Dealer_Hedging': '避險自營商',
        'Foreign_Dealer_Self': '外資自營商',
    }
    combined_df['name'] = combined_df['name'].replace(name_map)

    # 直接針對所有法人 (外資、投信、自營商) 計算淨買超 (單位：股)
    combined_df['net_buy'] = combined_df['buy'] - combined_df['sell']
    
    # 將單位從「股」轉換為「張」，並去除小數點
    combined_df['buy'] = (combined_df['buy'] / 1000).astype(int)
    combined_df['sell'] = (combined_df['sell'] / 1000).astype(int)
    combined_df['net_buy'] = (combined_df['net_buy'] / 1000).astype(int)

    # 選取我們需要的欄位
    consensus_df = combined_df[['date', 'stock_id', 'name', 'buy', 'sell', 'net_buy', 'close']]
    
    # 排序邏輯：先依照 stock_id (股票代碼) 排序，再依照 date (日期) 排序，同一檔股票內再依照 net_buy (淨買超) 由大到小排序
    consensus_df = consensus_df.sort_values(by=['stock_id', 'date', 'net_buy'], ascending=[True, True, False])
    
    return consensus_df

# ==========================================
# 4. 主程式執行區
# ==========================================
if __name__ == "__main__":
    # 步驟 A：讀取股票清單
    target_stocks = load_target_stocks(target_file)
    
    # 步驟 B：如果有成功讀取到清單，才執行後續的 API 抓取
    if target_stocks:
        print(
            f"正在獲取近三個月的法人買賣超資料：{target_start_date} 到 {target_end_date} ..."
        )
        consensus_result = get_smart_money_consensus(
            target_stocks, target_start_date, target_end_date
        )

        print("\n=== 經理人共同意圖 (投信買賣超) 監測結果 ===")
        if not consensus_result.empty:
            print(consensus_result.to_string(index=False))
            
            # 匯出 CSV
            csv_filename = f"institutional_net_buy_{target_start_date}_{target_end_date}.csv"
            consensus_result.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            print(f"\n✅ 分析完成！資料已成功匯出至：{csv_filename}")
        else:
            print("找不到指定目標與日期的投信資料。")