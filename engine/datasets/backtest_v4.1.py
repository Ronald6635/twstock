"""
Backtest V4.1 Module

此模組實作了 V4.1 股票篩選策略的歷史回測功能。
它旨在評估特定篩選條件在歷史數據上的表現，幫助驗證策略的有效性。

主要功能:
- 讀取包含股票代號及名稱的候選 CSV 檔案。
- 透過 FinMind API 獲取指定回測日期的歷史股價數據。
- 計算多種技術指標（如移動平均線、Supertrend）。
- 根據預設的篩選條件（成交值、均線趨勢、Supertrend 訊號）識別符合策略的股票。
- 計算符合條件股票在指定持有期內的報酬率和期間最大漲幅。
- 在終端輸出詳細的回測報告，包括平均報酬率和勝率。

注意事項:
- 需要設定 FINMIND_API_KEY 環境變數或在函數中傳入 token。
- 股票代號應位於 CSV 檔案的第0欄，名稱應位於 '名稱' 欄位。
- 僅包含簡化版的 V4.1 篩選邏輯，未包含所有籌碼數據和更多技術指標。
"""

import os
import sys
import pandas as pd
import time
import logging
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from FinMind.data import DataLoader
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

load_dotenv()  # 載入 .env 檔案中的環境變數

# Helper: fetch and normalize FinMind /v2/user_info response
def fetch_finmind_usage(token: str, timeout: int = 10) -> Dict[str, Any]:
    """取得 FinMind API 使用量資訊並回傳包含剩餘次數與配額的字典。"""
    url = "https://api.web.finmindtrade.com/v2/user_info"
    headers = {"Authorization": f"Bearer {token}"}
    result: Dict[str, Any] = {
        'ok': False,
        'status_code': None,
        'api_request_limit': None,
        'api_requests_remaining': None,
        'user_count': None,
        'error': None,
        'text': ''
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        result['status_code'] = resp.status_code
        result['ok'] = resp.ok
        result['text'] = resp.text[:200] if hasattr(resp, 'text') else ''

        if not resp.ok:
            result['error'] = f"HTTP {resp.status_code}"
            return result

        data = resp.json() if resp.headers.get('content-type', '').lower().startswith('application/json') else {}
        if isinstance(data, dict):
            result['user_count'] = data.get('user_count')
            result['api_request_limit'] = data.get('api_request_limit')
            result['api_requests_remaining'] = data.get('api_requests_remaining')
            if result['api_requests_remaining'] is None and \
               result['api_request_limit'] is not None and \
               result['user_count'] is not None:
                try:
                    result['api_requests_remaining'] = int(result['api_request_limit']) - int(result['user_count'])
                except Exception:
                    pass

        return result

    except Exception as e:
        result['error'] = str(e)
        return result

# 動態加入 indicators.py 所在路徑
sys.path.append(os.path.dirname(__file__))
from indicators import compute_supertrend, compute_ma # 明確列出導入的函數


def fetch_chip_data(api: DataLoader, stock_id: str, chip_start_date: str, chip_end_date: Optional[str] = None) -> Dict[str, Any]:
    """Fetch institutional investor / margin chip data for backtest + compute net flow."""
    collected: Dict[str, Any] = {}

    try:
        if chip_end_date:
            ins_df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=chip_start_date, end_date=chip_end_date)
        else:
            ins_df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=chip_start_date)

        if ins_df is not None and not ins_df.empty:
            ins_df['net'] = ins_df['buy'] - ins_df['sell']
            foreign_df = ins_df.loc[ins_df['name'].isin(['Foreign_Investor', 'ForeignInvestor']), ['date', 'net']]
            if not foreign_df.empty:
                foreign_series = foreign_df.groupby('date')['net'].sum().sort_index()
                collected['foreign_by_date'] = foreign_series

    except Exception as e:
        logging.warning(f"[{stock_id}] 取得外資籌碼失敗: {e}")

    try:
        if chip_end_date:
            margin_df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=chip_start_date, end_date=chip_end_date)
        else:
            margin_df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=chip_start_date)

        if margin_df is not None and not margin_df.empty:
            margin_df = margin_df.sort_values('date')
            collected['margin_by_date'] = margin_df.set_index('date')['MarginPurchaseTodayBalance']

    except Exception as e:
        logging.warning(f"[{stock_id}] 取得融資籌碼失敗: {e}")

    return collected


# =============================================================================
# CONFIGURATION AND INITIALIZATION
# =============================================================================

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# DATA FETCHING AND PERFORMANCE CALCULATION
# =============================================================================

def get_backtest_performance(api: DataLoader, stock_id: str, test_date: str, hold_days: int = 20) -> Tuple[Optional[float], Optional[float]]:
    """
    計算從回測日開始持有指定天數後的報酬率與期間最大漲幅。

    Args:
        api (DataLoader): FinMind DataLoader 實例，用於獲取股票資料。
        stock_id (str): 股票代號。
        test_date (str): 回測開始日期 (YYYY-MM-DD)。
        hold_days (int, optional): 模擬持有股票的天數。預設為 20 天。

    Returns:
        Tuple[Optional[float], Optional[float]]:
            - 持有期報酬率 (百分比)。
            - 期間最大漲幅 (百分比)。
            若資料不足或無法計算，則回傳 (None, None)。

    Example:
        >>> from FinMind.data import DataLoader
        >>> api_loader = DataLoader()
        >>> api_loader.login_by_token("YOUR_FINMIND_TOKEN")
        >>> ret, max_ret = get_backtest_performance(api_loader, "2330", "2023-01-01", 10)
        >>> print(f"報酬率: {ret:.2f}%, 最大漲幅: {max_ret:.2f}%")
    """
    end_date: str = (datetime.strptime(test_date, "%Y-%m-%d") + timedelta(days=hold_days + 15)).strftime("%Y-%m-%d")
    df_perf: pd.DataFrame = api.taiwan_stock_daily(stock_id=stock_id, start_date=test_date, end_date=end_date)

    if df_perf is None or df_perf.empty:
        logging.warning(f"[{stock_id}] 從 {test_date} 起沒有績效計算資料。")
        return None, None

    # 統一 FinMind 欄位名稱 (改為小寫，與主篩選邏輯一致)
    df_perf = df_perf.rename(columns={
        "date": "date",
        "open": "open",
        "max": "high",
        "min": "low",
        "close": "close",
        "Trading_Volume": "volume"
    })
    
    # 檢查是否至少有 hold_days 筆數據可用於計算
    if len(df_perf) < hold_days:
        logging.warning(f"[{stock_id}] 績效計算資料不足 {hold_days} 天。實際天數: {len(df_perf)}。")
        return None, None
    
    entry_price: float = df_perf.iloc[0]['close']
    exit_price: float = df_perf.iloc[hold_days-1]['close']
    # 修正: 從小寫 'high' 欄位取得最大值
    max_high: float = df_perf.iloc[:hold_days]['high'].max()
    
    ret: float = (exit_price - entry_price) / entry_price * 100
    max_ret: float = (max_high - entry_price) / entry_price * 100
    
    return round(ret, 2), round(max_ret, 2)

# =============================================================================
# BACKTESTING LOGIC
# =============================================================================

def run_backtest_v41(csv_file_name: str, token: Optional[str] = None, backtest_date: str = "2026-02-23", hold_days: int = 20) -> None:
    """
    執行 V4.1 策略的歷史回測。

    此函數讀取篩選候選清單，抓取歷史股價數據並計算技術指標，
    模擬在指定回測日期使用 V4.1 篩選邏輯，並計算持有指定天數後的績效。

    Args:
        csv_file_name (str): 包含篩選候選股票代號及名稱的 CSV 檔案名稱。
                             檔案預期與此腳本位於同一目錄。
        token (Optional[str], optional): FinMind API 驗證 token。
                                        如果未提供，將嘗試從環境變數 FINMIND_API_KEY 讀取。
                                        Defaults to None.
        backtest_date (str, optional): 回測的基準日期 (YYYY-MM-DD)。
                                       Defaults to "2026-02-23".
        hold_days (int, optional): 模擬持有股票的天數。Defaults to 20.

    Returns:
        None: 函數直接在終端輸出回測報告。

    Raises:
        ValueError: 如果 FinMind API token 未設定。
        FileNotFoundError: 如果候選 CSV 檔案不存在。
        KeyError: 如果 CSV 檔案缺少必要的欄位（如 '名稱'）。
        Exception: 處理股票資料或計算指標時發生的任何未預期錯誤。

    Example:
        >>> # 從環境變數讀取 token
        >>> os.environ['FINMIND_API_KEY'] = "YOUR_FINMIND_TOKEN"
        >>> run_backtest_v41('fundamental_pass.csv', backtest_date='2023-01-01')
        >>> # 直接傳入 token
        >>> run_backtest_v41('fundamental_pass.csv', 'YOUR_FINMIND_TOKEN', '2023-03-15', 30)
    """
    if token is None:
        token = os.getenv("FINMIND_API_KEY")

    if not token or token == "YOUR_FINMIND_TOKEN":
        raise ValueError("請設定 FINMIND_API_KEY 環境變數，或在函數中傳入 token。")
        
    api: DataLoader = DataLoader()
    api.login_by_token(token)

    # 檢查 FinMind API 使用量
    usage = fetch_finmind_usage(token)
    if usage.get('ok'):
        logging.info(f"FinMind 使用量: limit={usage.get('api_request_limit')} remaining={usage.get('api_requests_remaining')} user_count={usage.get('user_count')}")
        if usage.get('api_requests_remaining') is not None and usage.get('api_requests_remaining') < 20:
            logging.warning("FinMind API 剩餘可用次數低於 20，請謹慎執行或調整回測資料量。")
    else:
        logging.warning(f"無法取得 FinMind 使用量: {usage.get('error') or usage.get('text')}")

    # 構建 CSV 檔案的完整路徑
    script_dir: Path = Path(__file__).parent
    full_csv_path: Path = script_dir / csv_file_name

    # 讀取 CSV
    candidate_df: pd.DataFrame
    try:
        candidate_df = pd.read_csv(full_csv_path)
        # 確保 stock_id 欄位是正確的，並且處理可能存在的 Excel 引用格式
        # 假設 stock_id 在第0欄 (iloc[:, 0])，並且是數字
        candidate_df['stock_id'] = candidate_df.iloc[:, 0].astype(str).str.extract(r'(\d+)')
        # 獲取公司名稱，為日誌和報告準備
        candidate_df['stock_name'] = candidate_df['名稱']
        stocks: List[str] = candidate_df['stock_id'].dropna().tolist()
        name_mapping: Dict[str, str] = candidate_df.set_index('stock_id')['stock_name'].to_dict()

    except FileNotFoundError:
        logging.error(f"讀取失敗: 找不到檔案 '{full_csv_path}'。請確認檔案是否存在。")
        return
    except KeyError as e:
        logging.error(f"CSV 檔案缺少必要欄位: {e}。請確認 'stock_id' 和 '名稱' 欄位。")
        return
    except Exception as e:
        logging.error(f"讀取 CSV 檔案時發生未知錯誤: {e}")
        return

    results: List[Dict[str, Any]] = []
    # 往前抓半年資料以計算長天期指標 (MA60, ADX)
    data_start: str = (datetime.strptime(backtest_date, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")

    logging.info(f"🚀 開始回測日期: {backtest_date} | 持有天數: {hold_days}天")

    for stock_id in stocks:
        stock_name: str = name_mapping.get(stock_id, stock_id) # 取得公司名稱，若無則顯示代號
        
        df: Optional[pd.DataFrame] = None
        try:
            # 抓取至回測日為止的資料進行篩選
            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=data_start, end_date=backtest_date)
            
            if df is None or df.empty or len(df) < 60: # 需要足夠數據計算 MA60
                logging.info(f"[{stock_id} {stock_name}] 數據不足或 FinMind API 無資料，跳過。")
                continue
            
            # 統一 FinMind 欄位名稱 (改為小寫，與 indicators.py 預期一致)
            df = df.rename(columns={
                "date": "date", 
                "open": "open", 
                "max": "high", 
                "min": "low", 
                "close": "close", 
                "Trading_Volume": "volume"
            })
            
            # 將 'date' 欄位轉換為 datetime 物件
            df['date'] = pd.to_datetime(df['date'])

            # 檢查關鍵欄位是否存在 (使用小寫欄位名稱)
            required_cols: List[str] = ['date','open','high','low','close','volume']
            missing_cols: List[str] = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"[{stock_id} {stock_name}] 數據缺少必要欄位：{missing_cols}，跳過。")
                continue

            # --- 執行 V4.1 篩選邏輯 (簡化版) ---
            # 為了計算指標，確保數據是按日期排序
            df = df.sort_values(by="date").reset_index(drop=True)

            # 修正: 將指標結果賦值到 DataFrame 的新欄位，而不是覆蓋 df
            df['ma_20'] = compute_ma(df, col="close", period=20)
            df['ma_60'] = compute_ma(df, col="close", period=60)
            
            supertrend_line: pd.Series
            dir_series: pd.Series
            supertrend_line, dir_series = compute_supertrend(df) # compute_supertrend 會使用 df['high'], df['low'], df['close']
            df['supertrend'] = supertrend_line # 將 supertrend line 加到 df
            df['direction'] = dir_series # 將 supertrend direction 加到 df
            
            # 確保指標計算結果可用 (檢查新添加的欄位)
            if 'ma_20' not in df.columns or 'ma_60' not in df.columns or \
               'supertrend' not in df.columns or 'direction' not in df.columns or \
               df['supertrend'].empty or df['direction'].empty: # 檢查結果 Series 是否為空
                logging.warning(f"[{stock_id} {stock_name}] 無法計算技術指標或指標為空，跳過。")
                continue

            latest: pd.Series = df.iloc[-1]
            
            current_price: float = latest['close']
            ma20: float = latest['ma_20']
            ma60: float = latest['ma_60']
            
            daily_turnover: float = current_price * latest['volume']
            is_trending: bool = current_price > ma20 > ma60
            is_bullish_st: bool = latest['direction'] == 1 # 從 df 中的 'direction' 欄位獲取

            # === 新增籌碼：回推 14 天外資淨買賣、融資餘額變化 ===
            chip_start: str = (datetime.strptime(backtest_date, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
            chip_data: Dict[str, Any] = fetch_chip_data(api, stock_id, chip_start, backtest_date)

            foreign_series = chip_data.get('foreign_by_date')
            foreign_14_sum = float(foreign_series.sum()) if foreign_series is not None and not foreign_series.empty else 0.0

            margin_series = chip_data.get('margin_by_date')
            margin_change = None
            if margin_series is not None and len(margin_series) >= 2:
                margin_change = float(margin_series.iloc[-1] - margin_series.iloc[0])

            chip_ok = (foreign_14_sum > 0) and (margin_change is None or margin_change <= 0)

            # 篩選條件：成交值 > 3億 & 均線多頭 & Supertrend 多頭
            if daily_turnover > 300_000_000 and is_trending and is_bullish_st:
                # 計算績效
                ret, max_ret = get_backtest_performance(api, stock_id, backtest_date, hold_days)
                
                if ret is not None and max_ret is not None:
                    results.append({
                        '代號': stock_id,
                        '名稱': stock_name, # 添加公司名稱
                        '進場價': current_price,
                        '持有期報酬(%)': ret,
                        '期間最大漲幅(%)': max_ret,
                        '外資14天淨買(累計)': round(foreign_14_sum, 2),
                        '融資14天變化': round(margin_change, 2) if margin_change is not None else None
                    })
                    logging.info(f"✅ [{stock_id} {stock_name}] 符合篩選! 進場價: {current_price:.2f}, 預期報酬率: {ret:.2f}%")
                else:
                    logging.warning(f"[{stock_id} {stock_name}] 無法計算績效，跳過。")
            else:
                # 詳細日誌記錄淘汰原因
                reason: str = ""
                if daily_turnover <= 300_000_000:
                    reason = f"成交值僅 {daily_turnover/100000000:.1f} 億 (門檻: 3億)"
                elif not is_trending:
                    reason = f"均線未達多頭排列 (C:{current_price:.1f}, M20:{ma20:.1f}, M60:{ma60:.1f})"
                elif not is_bullish_st:
                    reason = "Supertrend 目前為空頭綠燈"
                logging.info(f"❌ [{stock_id} {stock_name}] 淘汰: {reason}")
            
        except Exception as e:
            logging.error(f"處理 {stock_id} ({stock_name}) 錯誤: {e}")
            continue # 跳過當前股票，繼續處理下一支股票

        time.sleep(0.6) # 避免 API 被封鎖

    # =============================================================================
    # BACKTESTING REPORT
    # =============================================================================

    # 輸出總結
    report: pd.DataFrame = pd.DataFrame(results)
    if not report.empty:
        # 排序報告以便閱讀
        report = report.sort_values(by='持有期報酬(%)', ascending=False).reset_index(drop=True)
        # 調整顯示設定，固定欄寬便於 terminal 對齊
        with pd.option_context('display.max_columns', None, 'display.width', 180, 'display.colheader_justify', 'center'):
            print(f"\n📈 V4.1 歷史回測報告 ({backtest_date})")
            print(report.to_string(index=False, col_space=13, justify='right'))
        print(f"\n總結: 篩選出 {len(report)} 檔符合標的。")
        print(f"平均報酬率: {report['持有期報酬(%)'].mean():.2f}%")
        print(f"勝率: {(report['持有期報酬(%)'] > 0).sum() / len(report) * 100:.1f}%")
    else:
        print(f"\n此日期 ({backtest_date}) 無符合標的。")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 從環境變數讀取 FinMind API Token
    FINMIND_TOKEN: Optional[str] = os.getenv("FINMIND_API_KEY")

    # 範例執行，請替換為您的實際檔案名稱和回測日期
    try:
        run_backtest_v41('fundamental_pass.csv', 
                         token=FINMIND_TOKEN, 
                         backtest_date="2026-01-15", # 將日期改為更實際的過去日期以確保數據可用
                         hold_days=20)
    except ValueError as e:
        logging.error(f"執行錯誤: {e}")
    except Exception as e:
        logging.error(f"回測過程中發生未預期錯誤: {e}")