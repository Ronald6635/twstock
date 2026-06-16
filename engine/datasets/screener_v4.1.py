"""
Screener V4.1 Module

This module implements an advanced stock screening process, extending V3.0 with
additional technical indicators and institutional investor (籌碼) data.

Key features:
- 讀取 `fundamental_pass_2026Q1.csv` 作為篩選候選。
- 透過 FinMind API 抓取日線價格資料與外資/融資籌碼資料。
- 計算多種技術指標，包括均線、Supertrend、RSI、MACD、ADX、Stochastic、OBV。
- 綜合成交值、趨勢、Supertrend、技術面、盈虧比及籌碼健康度進行多層次篩選。
- 將篩選結果輸出至終端，並將篩選原因回寫至 `fundamental_pass_2026Q1.csv`。
"""

import os
import sys
import pandas as pd
import time
import logging
from dotenv import load_dotenv
from FinMind.data import DataLoader

# =============================================================================
# CONFIGURATION AND INITIALIZATION
# =============================================================================

# 動態加入 indicators.py
sys.path.append(os.path.dirname(__file__))
from indicators import (
    compute_supertrend, compute_ma, compute_rsi, compute_stochastic,
    compute_macd, compute_obv, compute_adx, generate_trading_analysis,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# DATA FETCHING
# =============================================================================


def fetch_stock_data(api: DataLoader, stock_id: str, start_date: str, end_date: str | None = None) -> pd.DataFrame | None:
    """Fetch OHLCV data for a single stock and normalise column names.

    This function retrieves daily stock data from the FinMind API for a given stock ID
    and date range, then renames columns to a standardized format expected by
    technical indicator calculations.

    Args:
        api (DataLoader): An authenticated FinMind DataLoader instance.
        stock_id (str): The unique identifier of the stock (e.g., "0050").
        start_date (str): The start date for fetching data in "YYYY-MM-DD" format.
        end_date (Optional[str]): The end date in "YYYY-MM-DD" format. If None, uses API default (latest).

    Returns:
        pd.DataFrame | None: A Pandas DataFrame containing 'date', 'open', 'high', 'low',
                             'close', 'volume' columns, or None if data fetching fails,
                             is empty, or missing required columns.

    Raises:
        Exception: Catches and logs any exceptions that occur during API calls or
                   data processing, returning None in such cases.
    """
    try:
        if end_date:
            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
        else:
            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df if {'date', 'open', 'high', 'low', 'close', 'volume'} <= set(df.columns) else None
    except Exception as e:
        logging.error(f"抓取 {stock_id} 價格資料失敗: {e}")
        return None


def fetch_chip_data(api, stock_id, chip_start_date, chip_end_date=None):
    """Fetches institutional investor and margin trading data for a given stock.

    This function retrieves foreign investor net buys/sells and margin purchase balance
    data from the FinMind API for the specified stock and date range.

    Args:
        api (DataLoader): An authenticated FinMind DataLoader instance.
        stock_id (str): The unique identifier of the stock.
        chip_start_date (str): The start date for fetching chip data in "YYYY-MM-DD" format.
        chip_end_date (Optional[str]): The end date in "YYYY-MM-DD" format. If None, uses API default (latest).

    Returns:
        dict: A dictionary containing lists for 'foreign_investor_net' and 'margin_balance'
              if available. Returns an empty dictionary if data fetching fails or no data is found.

    """
    fundamental_data = {}
    try:
        if chip_end_date:
            ins_df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=chip_start_date, end_date=chip_end_date)
        else:
            ins_df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=chip_start_date)
        # 先假設欄位存在
        if not ins_df.empty:
            ins_df['net'] = ins_df['buy'] - ins_df['sell']
            # 取 foreign_investor_net on per-date basis
            foreign = ins_df.loc[ins_df['name'].isin(['Foreign_Investor','ForeignInvestor']), ['date','net']].groupby('date')['net'].sum()
            fundamental_data['foreign_investor_net'] = foreign.tolist()  # 或直接 Series
            # (= preproc 行為)
    except:
        pass

    try:
        if chip_end_date:
            margin_df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=chip_start_date, end_date=chip_end_date)
        else:
            margin_df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=chip_start_date)
        if not margin_df.empty:
            margin_df = margin_df.sort_values('date')
            fundamental_data['margin_balance'] = margin_df['MarginPurchaseTodayBalance'].tolist()
    except:
        pass

    return fundamental_data


# =============================================================================
# CORE SCREENER LOGIC
# =============================================================================

def run_v4_screener(csv_path: str, token: str | None = None) -> None:
    """Executes the V4.1 stock screening process.

    This function reads a list of candidate stock IDs from a CSV file, fetches
    their historical price and institutional investor data, calculates various
    technical indicators, applies a series of filtering rules (turnover, trend,
    Supertrend, ADX, RSI, MACD, risk/reward ratio, and chip health), then
    outputs the results to the console and updates the original CSV with filter reasons.

    Args:
        csv_path (str): The absolute path to the CSV file containing the candidate stock list.
                        The CSV is expected to have a '代號' column for stock IDs and '名稱' for stock names.
        token (str | None): Optional. The FinMind API token. If None, it will be fetched
                            from the 'FINMIND_API_KEY' environment variable.

    Returns:
        None: This function does not return any value but prints the screening results
              and modifies the input CSV file in place.

    Raises:
        ValueError: If the FinMind API token is not provided and not found in environment variables.
        Exception: Catches and logs various exceptions that may occur during CSV reading,
                   data fetching, or technical analysis, skipping to the next stock.
    """
    token = token or os.getenv("FINMIND_API_KEY")
    if not token:
        raise ValueError("請設定 FINMIND_API_KEY")

    api = DataLoader()
    api.login_by_token(token)

    candidate_df = pd.read_csv(csv_path, dtype={'代號': str})  # Force string dtype to prevent int64 inference
    candidate_df['stock_id'] = candidate_df['代號'].str.extract(r'(\d+)')
    stocks = candidate_df['stock_id'].dropna().tolist()
    name_mapping = candidate_df.set_index(candidate_df['stock_id'].astype(str))['名稱'].to_dict()

    # 動態起始日期（2年歷史），明確終止日期為今天
    today = pd.Timestamp.today().normalize()
    price_start = (today - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    chip_start = (today - pd.Timedelta(days=180)).strftime("%Y-%m-%d")   # 籌碼只需180天
    price_end = today.strftime("%Y-%m-%d")
    chip_end = price_end

    logging.info(f"🚀 V4.1 啟動！（已加入外資 + 融資籌碼）")
    logging.info(f"價格資料起始: {price_start} | 價格資料結束: {price_end} | 籌碼資料起始: {chip_start} | 籌碼資料結束: {chip_end}")
    logging.info(f"開始掃描 {len(stocks)} 檔...")

    results = []
    filter_reasons = {}

    for stock_id in stocks:
        stock_name = name_mapping.get(str(stock_id), "Unknown")
        logging.info(f"分析: {stock_id} ({stock_name})")

        df = fetch_stock_data(api, stock_id, price_start, price_end)
        if df is None or len(df) < 150:
            reason = f"資料不足 ({len(df) if df is not None else 0} 筆)"
            filter_reasons[str(stock_id)] = reason
            logging.warning(f"跳過 {stock_id}：{reason}")
            continue

        try:
            # 計算技術指標（同 V4）
            df['ma_20'] = compute_ma(df, col='close', period=20)
            df['ma_60'] = compute_ma(df, col='close', period=60)
            df['ma_120'] = compute_ma(df, col='close', period=120)
            supertrend_line, direction = compute_supertrend(df, period=10, multiplier=3.0)
            df['supertrend'] = supertrend_line
            df['direction'] = direction

            df['rsi_14'] = compute_rsi(df, period=14)
            df['macd'], df['macd_signal'], df['macd_histogram'] = compute_macd(df)
            df['adx'], _, _ = compute_adx(df, period=14)
            df['stoch_k'], df['stoch_d'] = compute_stochastic(df)
            df['obv'] = compute_obv(df)

            latest = df.iloc[-1]
            current_price = latest['close']
            support = latest['supertrend']
            st_dir = latest['direction']
            prev_high = df['high'].iloc[-120:-1].max()
            daily_turnover = current_price * latest['volume']

            # V4 技術濾網（不變）
            is_trending = current_price > df['ma_20'].iloc[-1] > df['ma_60'].iloc[-1]
            is_bullish_st = st_dir == 1
            adx_ok = not pd.isna(latest['adx']) and latest['adx'] >= 25
            rsi_ok = latest['rsi_14'] <= 70
            macd_ok = latest['macd_histogram'] > 0

            # 逐關淘汰（同 V4）
            if daily_turnover < 300_000_000:
                reason = f"成交值僅 {daily_turnover/1e8:.1f} 億"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id}] ❌ {reason}")
                continue
            if not is_trending or not is_bullish_st or not adx_ok or not rsi_ok or not macd_ok:
                reason = "技術面未達標"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id}] ❌ {reason}")
                continue

            # 盈虧比
            risk = current_price - support
            reward = prev_high - current_price
            if risk <= 0 or (reward / risk) < 3.0:
                reason = "盈虧比不足或已破支撐"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id}] ❌ {reason}")
                continue

            # === V4.1 新增：抓外資 + 融資籌碼 ===
            logging.info(f"[{stock_id} {stock_name}] ✅ 技術+盈虧比通過 → 抓取外資/融資籌碼")
            fundamental_data = fetch_chip_data(api, stock_id, chip_start, chip_end)

            # 14 日外資 + 融資計算
            foreign_list = fundamental_data.get('foreign_investor_net', [])
            margin_list = fundamental_data.get('margin_balance', [])
            foreign_14_sum = sum(foreign_list[-14:]) if foreign_list else 0.0
            margin_14_change = None
            if len(margin_list) >= 2:
                last14 = margin_list[-14:] if len(margin_list) >= 14 else margin_list
                margin_14_change = last14[-1] - last14[0]

            # 產生完整 X 光報告（現在包含籌碼加權分數）
            generate_trading_analysis(
                results_df=df,
                sample_df=df,
                period=10,
                multiplier=3.0,
                fundamental_data=fundamental_data,
                analysis_days=14,
            )

            filter_reasons[str(stock_id)] = "符合V4.1（含籌碼）"
            results.append({
                'Stock ID': stock_id,
                'Stock Name': stock_name,
                'Price': round(current_price, 2),
                'Turnover (100M)': round(daily_turnover / 1e8, 2),
                'Support': round(support, 2),
                'Target': round(prev_high, 2),
                'R:R': round(reward / risk, 2),
                'ADX': round(latest['adx'], 1),
                'RSI': round(latest['rsi_14'], 1),
                'MACD Hist': round(latest['macd_histogram'], 2),
                'Foreign14Net': round(foreign_14_sum, 2),
                'Margin14Change': round(margin_14_change, 2) if margin_14_change is not None else None,
            })

        except Exception as e:
            logging.error(f"處理 {stock_id} 錯誤: {e}")
            continue

        time.sleep(0.5)

    # =============================================================================
    # RESULTS HANDLING
    # =============================================================================

    # 寫回 CSV
    candidate_df['filter_reason'] = candidate_df['stock_id'].astype(str).map(filter_reasons).fillna('未執行')
    candidate_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 最終報告
    final = pd.DataFrame(results)
    if not final.empty:
        with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.colheader_justify', 'center'):
            print("\n🎯 V4.1 最終結果（技術 + 外資/融資籌碼加權）")
            print(final.sort_values(by='R:R', ascending=False).to_string(index=False, col_space=13, justify='right'))
    else:
        print("\n🛡️ 本次無符合 V4.1 全條件標的（含外資/融資健康）")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Main execution block for the screener script.

    This block is executed when the script is run directly.
    It retrieves the FinMind API token from environment variables,
    constructs the path to the fundamental_pass_2026Q1.csv file, and then
    calls the `run_v4_screener` function to start the screening process.

    Raises:
        RuntimeError: If the 'FINMIND_API_KEY' environment variable is not set.
    """
    token = os.getenv('FINMIND_API_KEY')
    if not token:
        raise RuntimeError('請設定 FINMIND_API_KEY')
    csv_path = os.path.join(os.path.dirname(__file__), 'fundamental_pass_2026Q1.csv')
    run_v4_screener(csv_path, token)