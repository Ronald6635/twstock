import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties
import argparse # Added argparse

# ==========================================
# 1. 設定與字體處理 (解決中文亂碼問題)
# ==========================================
# 台灣 Windows 使用者通常有微軟正黑體，Mac 使用者則有標楷體或 Apple LiGothic
# 這裡嘗試設定一個通用的中文字體，若執行時標題出現框框，請手動指定路徑
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# Removed hardcoded csv_filename and targets_filename

# ==========================================
# 2. 資料讀取與處理邏輯
# ==========================================
def load_target_stock_ids(filename: str) -> set[str]: # Added type hint
    if not os.path.exists(filename):
        print(f"[WARNING] 找不到股票清單檔案 '{filename}'，將停止繪圖。")
        return set()

    with open(filename, "r", encoding="utf-8") as file:
        target_ids = {line.strip() for line in file if line.strip()}

    if not target_ids:
        print(f"[WARNING] 股票清單檔案 '{filename}' 為空，將停止繪圖。")

    return target_ids


def analyze_institutional_trends(csv_filename: str, targets_filename: str) -> None:
    if not os.path.exists(csv_filename): # Use csv_filename
        print(f"[WARNING] 錯誤：找不到檔案 '{csv_filename}'，請確認檔案名稱與路徑。")
        return

    target_stock_ids = load_target_stock_ids(targets_filename) # Pass targets_filename
    if not target_stock_ids:
        return

    # 讀取 CSV（明確指定 stock_id 型別，避免 mixed types 警告）
    df = pd.read_csv(
        csv_filename,
        dtype={"stock_id": "string"},
        low_memory=False,
    )
    df["stock_id"] = df["stock_id"].str.strip()

    # 轉換日期格式並排序
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["stock_id", "date"])

    # 選擇法人名稱欄位，兼容舊版輸出
    investor_column = 'name' if 'name' in df.columns else 'investor_name' if 'investor_name' in df.columns else None
    if investor_column is None:
        print("[WARNING] 找不到法人名稱欄位，請確認 CSV 包含 'name' 或 'investor_name'。")
        return

    # 計算累積買賣超 (按股票與法人名稱分組後累加)
    df['cum_net_buy'] = df.groupby(['stock_id', investor_column])['net_buy'].transform('cumsum')

    if 'close' not in df.columns:
        print("[WARNING] 找不到股價欄位 'close'，請確認 CSV 包含 close 欄位。")
        return

    df['stock_id'] = df['stock_id'].astype(str)
    filtered_df = df[df['stock_id'].isin(target_stock_ids)].copy()
    if filtered_df.empty:
        print("[WARNING] CSV 中找不到 targets.txt 指定的股票代碼，未產生任何圖表。")
        return

    # 只保留最近 90 天的資料
    cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=90)
    filtered_df = filtered_df[filtered_df['date'] >= cutoff_date].copy()
    if filtered_df.empty:
        print("[WARNING] 最近 90 天內沒有符合條件的資料，未產生任何圖表。")
        return

    # 取得清單中所有的股票代碼
    unique_stocks = filtered_df['stock_id'].unique()

    print(f"偵測到 {len(unique_stocks)} 檔目標股票資料，開始繪製趨勢圖...")

    # 為每檔股票繪製圖表
    for stock_id in unique_stocks:
        stock_df = filtered_df[filtered_df['stock_id'] == stock_id].copy()
        stock_df = stock_df.sort_values(by='date')

        plt.figure(figsize=(14, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # 取得該股票內出現的所有法人名稱 (外資、投信、自營商等)
        investors = stock_df[investor_column].unique()

        for investor in investors:
            investor_df = stock_df[stock_df[investor_column] == investor]
            line = ax1.plot(
                investor_df['date'],
                investor_df['cum_net_buy'],
                label=investor,
                marker='o',
                markersize=4,
                linewidth=1,
            )
            
            color = line[0].get_color()
            
            # Calculate pure buy dynamic cost array
            current_inventory = 0
            current_cost = 0
            costs = []
            for _, row in investor_df.iterrows():
                buy_vol = row.get('buy', 0)
                # Fallback to close if vwap does not exist or is missing
                price = row['vwap'] if 'vwap' in row and pd.notna(row['vwap']) else row['close']
                
                if pd.notna(buy_vol) and buy_vol > 0:
                    current_cost = (current_inventory * current_cost + buy_vol * price) / (current_inventory + buy_vol)
                    current_inventory += buy_vol

                # Ignore sells. Only keep tracking pure buy cost without decrementing inventory.
                costs.append(current_cost if current_inventory > 0 else float('nan'))
                
            ax2.plot(
                investor_df['date'],
                costs,
                label=f'{investor} 動態成本',
                color=color,
                linestyle=':',
                alpha=0.7,
                linewidth=2,
            )
            
            if costs and costs[-1] > 0:
                last_date = investor_df['date'].iloc[-1]
                last_cost = costs[-1]
                ax2.text(last_date, last_cost, f'{last_cost:.2f}', color=color, va='center', ha='left', fontsize=10, fontweight='bold')

        # 股價走勢使用次座標軸
        ax2.plot(
            stock_df['date'],
            stock_df['close'],
            label='收盤價',
            color='black',
            linestyle='--',
            marker='*',
            markersize=8,
            linewidth=2,
        )

        # 圖表美化
        ax1.set_title(f"股票代碼：{stock_id} - 三大法人累積買賣超與股價/動態成本趨勢", fontsize=16)
        ax1.set_xlabel("日期", fontsize=12)
        ax1.set_ylabel("累積淨買超 (張)", fontsize=12)
        ax2.set_ylabel("收盤價 / 動態成本", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        plt.xticks(rotation=45)
        plt.tight_layout()

        # 儲存圖檔
        stock_name = str(stock_df['stock_name'].iloc[0]) if 'stock_name' in stock_df.columns else ""
        safe_stock_name = "".join(c for c in stock_name if c not in r'\/*?:"<>|')
        name_suffix = f"_{safe_stock_name}" if safe_stock_name else ""
        
        end_date_str = stock_df['date'].max().strftime('%Y-%m-%d')
        output_image = f"trend_{stock_id}{name_suffix}_{end_date_str}.png"
        plt.savefig(output_image)
        plt.close()
        print(f"已產生圖表：{output_image}")

# ==========================================
# 3. 執行分析
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="繪製法人累積買賣超與股價/動態成本趨勢圖")
    parser.add_argument("--csv-filename", type=str, required=True, help="輸入的 CSV 檔案路徑 (fetcher 輸出)")
    parser.add_argument("--targets-filename", type=str, default="targets.txt", help="目標股票 ID 檔案路徑")
    args = parser.parse_args()
    
    analyze_institutional_trends(args.csv_filename, args.targets_filename) # Pass arguments
    print("\n所有分析圖表已完成，請查看資料夾中的 .png 檔案。")