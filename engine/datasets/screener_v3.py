"""
Screener V3 module

This module performs a technical screening process based on a fundamental candidate list.

Key features:
- 讀取 `fundamental_pass.csv` 為篩選候選。
- 透過 FinMind API 抓取日線資料並計算 20/60 日均線、Supertrend。
- 依成交值、趨勢、Supertrend、盈虧比等條件篩選，並回寫 `filter_reason`。
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

# 動態加入 indicators.py 所在路徑
sys.path.append(os.path.dirname(__file__))
from indicators import compute_supertrend, compute_ma

# 載入 .env
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_stock_data(api: DataLoader, stock_id: str, start_date: str) -> pd.DataFrame | None:
    """Fetch OHLCV data for a single stock and normalise column names.

    This function retrieves daily stock data from the FinMind API for a given stock ID
    and date range, then renames columns to a standardized format expected by
    technical indicator calculations.

    Args:
        api (DataLoader): An authenticated FinMind DataLoader instance.
        stock_id (str): The unique identifier of the stock (e.g., "0050").
        start_date (str): The start date for fetching data in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame | None: A Pandas DataFrame containing 'date', 'open', 'high', 'low',
                             'close', 'volume' columns, or None if data fetching fails
                             or no data is available.

    Raises:
        Exception: Catches and logs any exceptions that occur during API calls or
                   data processing, returning None in such cases.
    """
    try:
        df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
        if df is None or df.empty:
            logging.warning(f"No data fetched for {stock_id} from {start_date}")
            return None

        # DEBUG: log actual columns so we can verify the mapping
        logging.debug(f"Raw columns for {stock_id}: {df.columns.tolist()}")

        # FIX: map FinMind column names → indicators.py expected names
        df = df.rename(columns={
            "max":              "high",
            "min":              "low",
            "Trading_Volume":   "volume",
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Guard: confirm required columns exist after rename
        required = {'date', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            logging.error(f"{stock_id}: missing columns after rename: {missing}")
            return None

        return df

    except Exception as e:
        logging.error(f"Error fetching {stock_id}: {e}")
        return None


# =============================================================================
# CORE SCREENER LOGIC
# =============================================================================

def run_v3_screener(csv_path: str, token: str | None = None) -> None:
    """Executes the V3 stock screening process.

    This function reads a list of candidate stock IDs from a CSV file, fetches
    their historical data, calculates technical indicators, applies a series of
    filtering rules (turnover, trend, Supertrend, risk/reward ratio), and then
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
        raise ValueError("請設定 FINMIND_API_KEY 環境變數，或傳入 token。")

    api = DataLoader()
    api.login_by_token(token)

    # 1. 讀取基本面篩選名單
    try:
        candidate_df = pd.read_csv(csv_path)
        # FIX: raw string to suppress SyntaxWarning
        candidate_df['stock_id'] = candidate_df['代號'].str.extract(r'(\d+)')
        stocks = candidate_df['stock_id'].dropna().tolist()
    except Exception as e:
        logging.error(f"Cannot read CSV: {e}")
        return

    results = []
    filter_reasons = {}
    # 建立 stock_id -> 名稱 的對照，用於日誌顯示
    name_mapping = candidate_df.set_index(candidate_df['stock_id'].astype(str))['名稱'].to_dict()
    start_date = "2025-06-01"  # long enough to compute MA60

    logging.info(f"開始掃描 {len(stocks)} 檔基本面篩選合格標的...")

    for stock_id in stocks:
        stock_name = name_mapping.get(str(stock_id), "Unknown")
        logging.info(f"正在分析: {stock_id} ({stock_name})")

        # 2. 抓取成交價格資料
        df = fetch_stock_data(api, stock_id, start_date)
        if df is None or len(df) < 60:
            reason = f"資料不足 ({len(df) if df is not None else 0} 筆)"
            filter_reasons[str(stock_id)] = reason
            logging.warning(f"跳過 {stock_id}：{reason}")
            continue

        try:
            # 3. 計算技術指標
            # compute_ma(df, col, period) — col is the column name string
            df['ma_20'] = compute_ma(df, col='close', period=20)
            df['ma_60'] = compute_ma(df, col='close', period=60)

            # compute_supertrend returns (supertrend_line, direction)
            supertrend_line, direction = compute_supertrend(df, period=10, multiplier=3.0)
            df['supertrend'] = supertrend_line
            df['st_dir']     = direction

            # 4. Read latest row
            latest        = df.iloc[-1]
            current_price = latest['close']
            ma20          = latest['ma_20']
            ma60          = latest['ma_60']
            st_dir        = latest['st_dir']

            # Support = supertrend level (valid in bullish regime)
            support       = latest['supertrend']

            # Target = 60-day high (excluding today)
            prev_high     = df['high'].iloc[-60:-1].max()

            # 5. 執行 V3.0 動能與流動性過濾
            # FIX: 'volume' is now correctly named after rename
            daily_turnover = current_price * latest['volume']

            is_trending   = current_price > ma20 > ma60
            is_bullish_st = st_dir == 1

            # [新增] X 光透視印出原因
            if daily_turnover < 500_000_000:
                reason = f"成交值僅 {daily_turnover/100000000:.1f} 億 (門檻: 5億)"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id} {stock_name}] 淘汰 ❌: {reason}")
                continue

            if not is_trending:
                reason = f"均線未達多頭排列 (C:{current_price:.1f}, M20:{ma20:.1f}, M60:{ma60:.1f})"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id} {stock_name}] 淘汰 ❌: {reason}")
                continue

            if not is_bullish_st:
                reason = "Supertrend 目前為空頭綠燈"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id} {stock_name}] 淘汰 ❌: {reason}")
                continue

            # 4. 盈虧比斷路器
            risk   = current_price - support
            reward = prev_high - current_price

            if risk <= 0:
                reason = "現價已跌破支撐線"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id} {stock_name}] 淘汰 ❌: {reason}")
                continue

            rr_ratio = reward / risk
            if rr_ratio < 3.0:
                reason = f"盈虧比僅 {rr_ratio:.2f} (門檻: 3.0) -> 太貴了，追高風險大"
                filter_reasons[str(stock_id)] = reason
                logging.info(f"[{stock_id} {stock_name}] 淘汰 ❌: {reason}")
                continue

            # 全部條件都通過，顯示 X 光檢查訊息
            logging.info(f"[{stock_id} {stock_name}] 成交值檢查: {daily_turnover:.0f}, 趨勢多頭: {is_trending}, Supertrend方向: {st_dir}, 風險: {risk:.2f}, 盈虧比: {rr_ratio:.2f}")

            # 如果全部過關，才會加進清單！
            filter_reasons[str(stock_id)] = "符合條件"
            results.append({
                'Stock ID':            stock_id,
                'Stock Name':          stock_name,
                'Price':               current_price,
                'Turnover (100M)':     round(daily_turnover / 100_000_000, 2),
                'Support':             round(support, 2),
                'Target':              round(prev_high, 2),
                'R:R Ratio':           round(rr_ratio, 2),
            })

        except Exception as e:
            logging.error(f"處理 {stock_id} 時發生錯誤：{e}")
            continue

        # 遵守 API 速率限制
        time.sleep(0.5)

    # =============================================================================
    # RESULTS HANDLING
    # =============================================================================

    # 6. 將篩選原因寫回原始 fundamental_pass.csv (最後一欄 filter_reason)
    candidate_df['filter_reason'] = candidate_df['stock_id'].astype(str).map(filter_reasons).fillna('未執行篩選')
    candidate_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 7. 輸出最終篩選結果
    final_report = pd.DataFrame(results)
    if not final_report.empty:
        print("\n🎯 V3.0 篩選結果 (R:R ≥ 1:3，多頭趨勢)：")
        print(final_report.sort_values(by='R:R Ratio', ascending=False).to_string(index=False))
    else:
        print("\n🛡️ 目前查無符合「1:3 盈虧比」且「動能充足」的標的。")
        print("💡 建議：繼續抱持現金，或等待名單中的優質股拉回至支撐位。")


if __name__ == "__main__":
    token = os.getenv('FINMIND_API_KEY')
    if not token:
        raise RuntimeError('Please set FINMIND_API_KEY env var before running.')

    # Build path relative to this script so it works from any working directory
    csv_file_path = os.path.join(os.path.dirname(__file__), 'fundamental_pass.csv')
    run_v3_screener(csv_file_path, token)