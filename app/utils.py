import matplotlib.pyplot as plt
import io
import base64
import twstock
from datetime import datetime
import os
import json
import pandas as pd
import requests
import time
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY

def generate_stock_chart(dates, prices, stock_id):
    """Generate a base64 encoded PNG chart for stock prices."""
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, label="Closing Price", color="blue", marker="o")
    plt.title(f"{stock_id} - Last 30 Days Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (NTD)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()
    return img_base64

def generate_kline_chart(stock_data, stock_id, title="K線圖"):
    """
    Generate a candlestick chart from stock data DataFrame.

    Args:
        stock_data (pd.DataFrame): DataFrame with columns 'date', 'open', 'high', 'low', 'close', 'volume' (optional)
        stock_id (str): Stock identifier
        title (str): Chart title

    Returns:
        str: Base64 encoded PNG image
    """
    if not isinstance(stock_data, pd.DataFrame):
        # Fallback to old method if not DataFrame
        if isinstance(stock_data, list) and len(stock_data) > 0:
            dates = [d.get('date', '') for d in stock_data]
            prices = [d.get('close', 0) for d in stock_data]
            return generate_stock_chart(dates, prices, stock_id)
        return ""

    # Ensure we have the required columns
    required_cols = ['date', 'open', 'high', 'low', 'close']
    if not all(col in stock_data.columns for col in required_cols):
        # Fallback to simple line chart
        dates = stock_data['date'].tolist() if 'date' in stock_data.columns else []
        prices = stock_data['close'].tolist() if 'close' in stock_data.columns else []
        return generate_stock_chart(dates, prices, stock_id)

    # Convert date strings to datetime if needed
    if 'date' in stock_data.columns:
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data = stock_data.sort_values('date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # K-line chart
    for idx, row in stock_data.iterrows():
        open_price = row['open']
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']

        # Determine color
        color = 'red' if close_price >= open_price else 'green'

        # Body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)

        # Shadow
        shadow_height = high_price - low_price
        shadow_bottom = low_price

        # Draw shadow
        ax1.plot([idx, idx], [low_price, high_price], color=color, linewidth=1)

        # Draw body
        if body_height > 0:
            ax1.add_patch(Rectangle((idx-0.4, body_bottom), 0.8, body_height,
                                  facecolor=color, edgecolor=color))
        else:
            # Doji
            ax1.plot([idx-0.4, idx+0.4], [open_price, open_price], color=color, linewidth=2)

    # Volume chart
    if 'Trading_Volume' in stock_data.columns or 'volume' in stock_data.columns:
        volume_col = 'Trading_Volume' if 'Trading_Volume' in stock_data.columns else 'volume'
        volumes = stock_data[volume_col]
        colors = ['red' if stock_data.iloc[i]['close'] >= stock_data.iloc[i]['open'] else 'green'
                 for i in range(len(stock_data))]
        ax2.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)

        # Format volume axis
        ax2.set_ylabel('Volume')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

    # Format price axis
    ax1.set_ylabel('Price (NTD)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

    # Set x-axis labels
    if len(stock_data) <= 30:
        ax1.set_xticks(range(len(stock_data)))
        ax1.set_xticklabels([d.strftime('%m/%d') for d in stock_data['date']], rotation=45)
        ax2.set_xticks(range(len(stock_data)))
        ax2.set_xticklabels([d.strftime('%m/%d') for d in stock_data['date']], rotation=45)
    else:
        # For longer periods, show fewer labels
        step = max(1, len(stock_data) // 10)
        ax1.set_xticks(range(0, len(stock_data), step))
        ax1.set_xticklabels([stock_data.iloc[i]['date'].strftime('%m/%d') for i in range(0, len(stock_data), step)], rotation=45)
        ax2.set_xticks(range(0, len(stock_data), step))
        ax2.set_xticklabels([stock_data.iloc[i]['date'].strftime('%m/%d') for i in range(0, len(stock_data), step)], rotation=45)

    ax1.set_title(f"{stock_id} - {title}")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to base64
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=100, bbox_inches='tight')
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()

    return img_base64

def safe_float(value):
    """Safely convert string to float, returning None if invalid."""
    try:
        return float(value) if value != '-' else None
    except (ValueError, TypeError):
        return None

def safe_int(value):
    """Safely convert string to int, returning None if invalid."""
    try:
        return int(value) if value != '-' else None
    except (ValueError, TypeError):
        return None


def load_data_from_datasets(stock_id, api_name, start_date=None, end_date=None, max_age_days=None):
    """Load cached JSON data from datasets folder.

    Parameters:
        stock_id (str): stock code or key used when saving (e.g., '2330', 'all', 'gold')
        api_name (str): api filename token used when saving (e.g., 'finmind_taiwan_stock_price')
        start_date, end_date (str|None): YYYY-MM-DD strings to find exact range files
        max_age_days (int|None): optional freshness check (None means no TTL)

    Returns:
        dict|list|None: parsed JSON content if found and valid; otherwise None
    """
    # Get company name from codes (same logic as save)
    codes = twstock.codes
    code_info = codes.get(stock_id)
    company_name = code_info.name if code_info else stock_id  # Fallback to stock_id

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{company_name}-{stock_id}')
    if not os.path.isdir(folder):
        return None

    # If exact start/end provided, look for the exact filename
    if start_date and end_date:
        target = os.path.join(folder, f"{start_date}_{end_date}_{api_name}.json")
        if os.path.exists(target):
            # Check age if needed
            if max_age_days is not None:
                mtime = os.path.getmtime(target)
                age_days = (time.time() - mtime) / 86400.0
                if age_days > max_age_days:
                    return None
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                # Corrupt file -> ignore
                return None
        return None

    # Otherwise find the newest file matching *_{api_name}.json
    candidates = []
    for fn in os.listdir(folder):
        if fn.endswith(f"_{api_name}.json"):
            fp = os.path.join(folder, fn)
            candidates.append((os.path.getmtime(fp), fp))
    if not candidates:
        return None

    # pick newest
    candidates.sort(reverse=True)
    newest = candidates[0][1]
    if max_age_days is not None:
        mtime = os.path.getmtime(newest)
        age_days = (time.time() - mtime) / 86400.0
        if age_days > max_age_days:
            return None
    try:
        with open(newest, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_data_to_datasets(stock_id, data, api_name, start_date=None, end_date=None):
    """Save API data to datasets/ with subfolder naming.

    If the endpoint provides a start_date and end_date, use them to prefix the
    saved filenames ("{start}_{end}_{api_name}.json/csv"). Otherwise fall back
    to using today's date as before.
    """
    # Get company name from codes
    codes = twstock.codes
    code_info = codes.get(stock_id)
    company_name = code_info.name if code_info else stock_id  # Fallback to stock_id

    # Use absolute path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{company_name}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    # Prefer start/end dates from the caller when available
    if start_date and end_date:
        date_str = f"{start_date}_{end_date}"
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Convert data to list of dicts for JSON serialization
    if isinstance(data, pd.DataFrame):
        json_data = data.to_dict('records')
        df = data
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        json_data = data
        df = pd.DataFrame(data)
    else:
        # Not convertible, save as JSON only
        json_data = data
        df = None
    
    # Save JSON file atomically
    json_filename = os.path.join(folder, f'{date_str}_{api_name}.json')
    try:
        import tempfile
        fd, tmp_json = tempfile.mkstemp(dir=folder, prefix=f'.tmp_{api_name}_', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        os.replace(tmp_json, json_filename)
    except Exception:
        # Fallback to non-atomic write
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    # Save CSV file if we have a DataFrame
    if df is not None:
        try:
            # Add company name and stock code as first two columns
            df.insert(0, '股票代碼', stock_id)
            df.insert(0, '公司名稱', company_name)
            
            # Rename columns to Traditional Chinese based on API
            column_rename_map = {}
            if api_name.startswith('finmind'):
                # Try to get translation from FinMind API
                api_key = os.getenv('FINMIND_API_KEY')
                if api_key:
                    # Map api_name to dataset
                    dataset_map = {
                        'finmind_taiwan_stock_price': 'TaiwanStockDaily',
                        'finmind_per_pbr': 'TaiwanStockPERPBR',
                        'finmind_institutional': 'TaiwanStockInstitutionalInvestorsBuySell',
                        'finmind_margin': 'TaiwanStockTotalMarginPurchaseShortSale',
                        'finmind_revenue': 'TaiwanStockRevenue',
                        'finmind_weekly': 'TaiwanStockWeekly',
                        'finmind_monthly': 'TaiwanStockMonthly'
                    }
                    dataset = dataset_map.get(api_name, api_name.replace('finmind_', ''))
                    url = f"https://api.finmindtrade.com/api/v4/translation?dataset={dataset}&token={api_key}"
                    try:
                        resp = requests.get(url)
                        if resp.status_code == 200:
                            trans_data = resp.json()
                            if 'data' in trans_data and isinstance(trans_data['data'], dict):
                                data = trans_data['data']
                                if 'english' in data and 'name' in data:
                                    column_rename_map = dict(zip(data['english'], data['name']))
                    except:
                        pass
            if not column_rename_map:
                # Fallback to hardcoded
                if api_name == 'finmind_taiwan_stock_price':
                    column_rename_map = {
                        'date': '日期',
                        'stock_id': '股票代碼',
                        'Trading_Volume': '成交量',
                        'Trading_money': '成交金額',
                        'open': '開盤價',
                        'max': '最高價',
                        'min': '最低價',
                        'close': '收盤價',
                        'spread': '漲跌',
                        'Trading_turnover': '成交筆數'
                    }
                elif api_name == 'stock':
                    column_rename_map = {
                        'stock_id': '股票代碼',
                        'current_price': '現價',
                        'open_price': '開盤價',
                        'high_price': '最高價',
                        'low_price': '最低價',
                        'volume': '成交量',
                        'price_change': '漲跌',
                        'best_four_point': '最佳四點',
                        'dates': '日期列表',
                        'prices': '價格列表'
                    }
                elif api_name == 'realtime':
                    column_rename_map = {
                        'stock_id': '股票代碼',
                        'price': '現價',
                        'open': '開盤價',
                        'high': '最高價',
                        'low': '最低價',
                        'volume': '成交量'
                    }
                elif api_name == 'codes':
                    column_rename_map = {
                        'name': '名稱',
                        'type': '類型',
                        'isin': 'ISIN代碼'
                    }
                elif api_name == 'analytics':
                    column_rename_map = {
                        'stock_id': '股票代碼',
                        'best_four_point': '最佳四點',
                        'recommendation': '建議'
                    }
            
            df.rename(columns=column_rename_map, inplace=True)
            
            csv_filename = os.path.join(folder, f'{date_str}_{api_name}.csv')
            # Write CSV atomically
            try:
                fd, tmp_csv = tempfile.mkstemp(dir=folder, prefix=f'.tmp_{api_name}_', suffix='.csv')
                with os.fdopen(fd, 'w', encoding='utf-8-sig') as f:
                    df.to_csv(f, index=False)
                os.replace(tmp_csv, csv_filename)
            except Exception:
                # Fallback to normal write
                df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        except Exception:
            # If CSV conversion fails, just continue without CSV
            pass