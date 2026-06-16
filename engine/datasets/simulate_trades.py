"""
Simulation Trades Module (V5 Prototype)

此模組負責讀取 V4.1 篩選出的候選名單，並在「進場後」模擬不同的出場策略（停損/停利）。
這是一個獨立於篩選邏輯的驗證工具，用於優化交易管理。

主要功能:
- 讀取 v41_passed_YYYY-MM-DD.csv 檔案。
- 抓取進場後 60 天的價格資料。
- 模擬多種出場規則：固定天數、Supertrend 停損、MA20 停損。
- 產出策略對比報告，評估期望值提升。
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from FinMind.data import DataLoader
from tabulate import tabulate
from dotenv import load_dotenv

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 動態加入 indicators.py 所在路徑
sys.path.append(os.path.dirname(__file__))
from indicators import compute_supertrend, compute_ma

load_dotenv()

class TradeSimulator:
    def __init__(self, token: str):
        self.api = DataLoader()
        self.api.login_by_token(token)
        self.price_cache = {}

    def fetch_future_data(self, stock_id: str, entry_date: str, days: int = 60) -> pd.DataFrame:
        """抓取進場日後一段時間的價格資料用於模擬。"""
        if stock_id in self.price_cache:
            return self.price_cache[stock_id]

        # 為了計算指標，我們需要往前抓一點資料 (比如 60 天)
        start_date = (datetime.strptime(entry_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(entry_date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
        
        df = self.api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "date": "date", "open": "open", "max": "high", "min": "low", "close": "close", "Trading_Volume": "volume"
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 預計算所有需要的指標
        df['ma_20'] = compute_ma(df, col="close", period=20)
        st_line, st_dir = compute_supertrend(df)
        df['supertrend'] = st_line
        df['st_direction'] = st_dir
        
        self.price_cache[stock_id] = df
        return df

    def simulate_strategy(self, stock_id: str, entry_date: str, rule: str = 'fixed_20') -> Dict[str, Any]:
        """
        模擬單一出場策略。
        rule options: 'fixed_20', 'supertrend_stop', 'ma20_stop'
        """
        df = self.fetch_future_data(stock_id, entry_date)
        if df.empty:
            return None

        # 找到進場點的索引
        entry_dt = pd.to_datetime(entry_date)
        entry_idx_list = df.index[df['date'] == entry_dt].tolist()
        if not entry_idx_list:
            return None
        
        entry_idx = entry_idx_list[0]
        entry_price = df.loc[entry_idx, 'close']
        
        # 模擬持倉期間 (最多 60 天，或到資料結束)
        sim_df = df.iloc[entry_idx:].copy().reset_index(drop=True)
        
        exit_idx = len(sim_df) - 1
        exit_reason = 'data_end'
        
        for i in range(1, len(sim_df)):
            current_row = sim_df.iloc[i]
            
            # 策略規則 A: 固定 20 天
            if rule == 'fixed_20' and i >= 20:
                exit_idx = 20 if len(sim_df) > 20 else len(sim_df) - 1
                exit_reason = 'fixed_timeout'
                break
            
            # 策略規則 B: Supertrend 停損 (跌破支撐)
            if rule == 'supertrend_stop':
                # ST 方向變為 0 (或 -1) 代表轉空
                if current_row['st_direction'] != 1:
                    exit_idx = i
                    exit_reason = 'supertrend_break'
                    break
            
            # 策略規則 C: MA20 停損
            if rule == 'ma20_stop':
                if current_row['close'] < current_row['ma_20']:
                    exit_idx = i
                    exit_reason = 'ma20_break'
                    break

        exit_price = sim_df.loc[exit_idx, 'close']
        ret = (exit_price - entry_price) / entry_price * 100
        
        # 計算期間最大漲幅
        max_high = sim_df.iloc[:exit_idx+1]['high'].max()
        max_ret = (max_high - entry_price) / entry_price * 100

        return {
            'strategy': rule,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_date': sim_df.loc[exit_idx, 'date'].strftime('%Y-%m-%d'),
            'hold_days': exit_idx,
            'return%': round(ret, 2),
            'max_up%': round(max_ret, 2),
            'reason': exit_reason
        }

def run_simulation(csv_file: str, token: str):
    if not os.path.exists(csv_file):
        logging.error(f"找不到檔案: {csv_file}")
        return

    candidates = pd.read_csv(csv_file)
    # 判斷日期 (從檔名 v41_passed_2026-01-15.csv 提取)
    try:
        entry_date = csv_file.split('_')[-1].replace('.csv', '')
        datetime.strptime(entry_date, '%Y-%m-%d')
    except:
        logging.error("無法從檔名辨識日期，請確保格式為 v41_passed_YYYY-MM-DD.csv")
        return

    simulator = TradeSimulator(token)
    all_results = []

    strategies = ['fixed_20', 'supertrend_stop', 'ma20_stop']
    
    logging.info(f"🚀 開始模擬交易驗證... 樣本數: {len(candidates)} | 進場日: {entry_date}")

    for _, row in candidates.iterrows():
        stock_id = str(row['代號'])
        stock_name = row['名稱']
        
        stock_perf = {'代號': stock_id, '名稱': stock_name}
        
        for strat in strategies:
            res = simulator.simulate_strategy(stock_id, entry_date, strat)
            if res:
                stock_perf[f'{strat}_ret%'] = res['return%']
                stock_perf[f'{strat}_days'] = res['hold_days']
            else:
                stock_perf[f'{strat}_ret%'] = np.nan
        
        all_results.append(stock_perf)
        logging.info(f"分析完成: [{stock_id} {stock_name}]")

    report_df = pd.DataFrame(all_results)
    
    # 統計報表
    summary = []
    for strat in strategies:
        col = f'{strat}_ret%'
        valid_rets = report_df[col].dropna()
        summary.append({
            '策略': strat,
            '平均報酬%': round(valid_rets.mean(), 2),
            '勝率%': round((valid_rets > 0).sum() / len(valid_rets) * 100, 1),
            '中位數%': round(valid_rets.median(), 2),
            '最大虧損%': round(valid_rets.min(), 2),
            '最大獲利%': round(valid_rets.max(), 2)
        })

    summary_df = pd.DataFrame(summary)

    print("\n📊 出場策略對比報告")
    print(tabulate(summary_df, headers='keys', tablefmt='github', showindex=False))
    
    print(f"\n📈 詳細交易清單 (共{len(report_df)} 檔)")
    print(tabulate(report_df, headers='keys', tablefmt='github', showindex=False))

    # 依照輸入檔名輸出結果 CSV
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_sim_file = f"{base_name}_simulation.csv"
    out_summary_file = f"{base_name}_simulation_summary.csv"
    report_df.to_csv(out_sim_file, index=False, encoding='utf-8-sig')
    summary_df.to_csv(out_summary_file, index=False, encoding='utf-8-sig')
    logging.info(f"💾 模擬交易報告已輸出: {out_sim_file}")
    logging.info(f"💾 模擬交易統計已輸出: {out_summary_file}")

if __name__ == "__main__":
    load_dotenv()
    FM_TOKEN = os.getenv("FINMIND_API_KEY")
    
    # 預設抓取最近產出的檔案 (可自行修改)
    import glob
    csv_files = glob.glob("v41_passed_*.csv")
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        run_simulation(latest_csv, FM_TOKEN)
    else:
        print("❌ 找不到任何 v41_passed_*.csv 檔案。請先執行 backtest_v4.1.py")
