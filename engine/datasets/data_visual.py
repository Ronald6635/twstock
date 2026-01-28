"""
Data Visualization Module for Stock Data

This module provides functions for visualizing stock data using various charts and plots.

It includes interactive Plotly charts for K-line analysis with institutional and margin data overlays.

Key features:
- Loads and prepares preprocessed stock data
- Generates auxiliary Plotly figures with multiple subplots
- Creates HTML dashboards with embedded data

Architecture notes:
- Supports flexible data formats (list, dict, nested structures)
- Handles missing data gracefully
- Outputs self-contained HTML files
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re  # used to decode unicode escape sequences in Plotly HTML
import base64
from typing import Optional


def ensure_supertrend(df_price: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach SuperTrend columns to `df_price` in-place if OHLC present and SuperTrend missing.

    Returns the modified DataFrame (same object for chaining). This is idempotent.
    """
    if df_price is None or df_price.empty:
        return df_price
    if {'supertrend', 'supertrend_dir'}.issubset(df_price.columns):
        return df_price
    if not {'high', 'low', 'close'}.issubset(df_price.columns):
        return df_price
    try:
        from engine.datasets.indicators import compute_supertrend
    except Exception:
        try:
            from indicators import compute_supertrend
        except Exception:
            return df_price
    try:
        st, st_dir = compute_supertrend(df_price, period=10, multiplier=3.0)
        df_price['supertrend'] = st
        df_price['supertrend_dir'] = st_dir
    except Exception:
        # non-fatal; leave df_price unchanged
        pass
    return df_price


def supertrend_thumbnail_png(df_price: pd.DataFrame, width: int = 700, height: int = 140) -> Optional[str]:
    """Render a compact PNG thumbnail of Close + SuperTrend and return a data-URL (base64 PNG).

    - Uses matplotlib (imported lazily) so it doesn't add a hard runtime dependency unless called.
    - Returns None if required columns are missing or rendering fails.

    Args:
        df_price: DataFrame containing at least 'date', 'close' and 'supertrend' columns.
        width, height: pixel dimensions of the output image.

    Returns:
        data URL (str) beginning with ``data:image/png;base64,`` or None on failure.
    """
    if df_price is None or df_price.empty:
        return None
    if not {'date', 'close', 'supertrend'}.issubset(df_price.columns):
        return None

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        from io import BytesIO

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.plot(pd.to_datetime(df_price['date']), df_price['close'], color="#1f77b4", linewidth=1.2, label='Close')
        ax.plot(pd.to_datetime(df_price['date']), df_price['supertrend'], color='k', linewidth=1.4, linestyle='--', label='SuperTrend')

        # plot buy/sell markers using same color convention as the interactive chart
        if 'supertrend_dir' in df_price.columns:
            buys = df_price[df_price['supertrend_dir'] == 1]
            sells = df_price[df_price['supertrend_dir'] == -1]
            if not buys.empty:
                ax.scatter(pd.to_datetime(buys['date']), buys['close'], marker='^', c='#FF3232', s=30, zorder=4)
            if not sells.empty:
                ax.scatter(pd.to_datetime(sells['date']), sells['close'], marker='v', c='#00AB5E', s=30, zorder=4)

        ax.set_axis_off()
        ax.margins(0.02)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode('ascii')
        return 'data:image/png;base64,' + data
    except Exception:
        return None

# Try importing from app, with a fallback to add project root to sys.path so the script can be run directly
try:
    from app.utils import generate_plotly_kline_chart
except (ImportError, ModuleNotFoundError):
    # If import fails, attempt to add project root to sys.path and retry
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    app_path = os.path.join(project_root, 'app')
    if os.path.isdir(app_path):
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            from app.utils import generate_plotly_kline_chart
        except Exception as err:
            raise ModuleNotFoundError(
                "Failed to import 'app.utils' even after adding project root to sys.path. "
                "Try running as a module from project root: 'python -m engine.datasets_ml.data_visual' or check your virtual environment/PYTHONPATH."
            ) from err
    else:
        raise ModuleNotFoundError(
            f"'app' package not found at expected location: {app_path}. Ensure you're in the project root or run the module with 'python -m engine.datasets_ml.data_visual'."
        )


def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file to load.

    Returns:
        Dictionary containing the loaded JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_data_for_chart(preprocessed_data: Dict[str, Any] | List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare preprocessed data for the chart function.

    This function is defensive and supports several common shapes:
    - A dict keyed by date -> record (the preprocessing output format)
    - A list of records (each with 'date', 'close', ...)
    - A dict with nested sections like {'prices': [...], 'institutional': [...], ...}

    It also tries to be tolerant of different column name casing.

    Args:
        preprocessed_data: Preprocessed data in various formats.

    Returns:
        Tuple of (price_data, institutional_data, margin_data, revenue_data) lists.
    """
    price_data: list[dict] = []
    institutional_data: list[dict] = []
    margin_data: list[dict] = []
    revenue_data: list[dict] = []

    def extract_price_from_record(rec: dict) -> dict | None:
        # Support multiple name variants and fallbacks for missing OHLC
        date = rec.get('date') or rec.get('Date') or rec.get('trade_date') or rec.get('tradeDate')
        # Prefer explicit close, then daily_revenue, then revenue as a last resort
        close = (rec.get('close') if 'close' in rec else
                 rec.get('Close') if 'Close' in rec else
                 rec.get('closing_price') if 'closing_price' in rec else
                 rec.get('ClosePrice') if 'ClosePrice' in rec else
                 rec.get('daily_revenue') if 'daily_revenue' in rec else
                 rec.get('revenue') if 'revenue' in rec else None)
        if close is None:
            return None
        # Synthesize open/high/low if missing using close
        open_v = rec.get('open') or rec.get('Open') or close
        high_v = rec.get('high') or rec.get('High') or close
        low_v = rec.get('low') or rec.get('Low') or close
        volume_v = rec.get('volume') or rec.get('Volume') or 0
        return {
            'date': date,
            'open': open_v,
            'high': high_v,
            'low': low_v,
            'close': close,
            'volume': volume_v
        }

    # Case: list of records
    if isinstance(preprocessed_data, list):
        for rec in preprocessed_data:
            if not isinstance(rec, dict):
                continue
            p = extract_price_from_record(rec)
            if p:
                price_data.append(p)
            # try to collect institutional/margin/revenue if present as lists
            if 'institutional' in rec and isinstance(rec['institutional'], list):
                institutional_data.extend(rec['institutional'])
            if 'margin' in rec and isinstance(rec['margin'], list):
                margin_data.extend(rec['margin'])
            if 'revenue' in rec and isinstance(rec['revenue'], (list, dict)):
                # keep monthly revenue entries
                if isinstance(rec['revenue'], list):
                    revenue_data.extend(rec['revenue'])
                else:
                    revenue_data.append(rec['revenue'])

    # Case: dict keyed by date or nested dict
    elif isinstance(preprocessed_data, dict):
        # If keys look like dates, values are records
        sample_vals = list(preprocessed_data.values())[:5]
        if sample_vals and isinstance(sample_vals[0], dict) and any(k.lower() in ('open','close','high','low') for k in sample_vals[0].keys()):
            for date, rec in preprocessed_data.items():
                if not isinstance(rec, dict):
                    continue
                # allow date to come from key if missing in record
                r = rec.copy()
                if 'date' not in r and isinstance(date, str):
                    r['date'] = date
                p = extract_price_from_record(r)
                if p:
                    price_data.append(p)
                # collect other sections if present per record
                if 'institutional' in r and isinstance(r['institutional'], list):
                    institutional_data.extend(r['institutional'])
                if 'margin' in r and isinstance(r['margin'], list):
                    margin_data.extend(r['margin'])
                if 'revenue' in r and isinstance(r['revenue'], (list, dict)):
                    if isinstance(r['revenue'], list):
                        revenue_data.extend(r['revenue'])
                    else:
                        revenue_data.append(r['revenue'])
        else:
            # Look for nested sections
            for key in ('prices','price','daily','data'):
                if key in preprocessed_data and isinstance(preprocessed_data[key], list):
                    for rec in preprocessed_data[key]:
                        p = extract_price_from_record(rec)
                        if p:
                            price_data.append(p)
            if 'institutional' in preprocessed_data and isinstance(preprocessed_data['institutional'], list):
                institutional_data.extend(preprocessed_data['institutional'])
            if 'margin' in preprocessed_data and isinstance(preprocessed_data['margin'], list):
                margin_data.extend(preprocessed_data['margin'])
            if 'revenue' in preprocessed_data and isinstance(preprocessed_data['revenue'], (list, dict)):
                if isinstance(preprocessed_data['revenue'], list):
                    revenue_data.extend(preprocessed_data['revenue'])
                else:
                    revenue_data.append(preprocessed_data['revenue'])

    # Sort price_data by date if possible
    try:
        price_data = sorted(price_data, key=lambda r: r['date'])
    except Exception:
        pass

    # Diagnostics: print counts so user can see why chart may be empty
    print(f"prepare_data_for_chart: found {len(price_data)} price rows, {len(institutional_data)} institutional rows, {len(margin_data)} margin rows, {len(revenue_data)} revenue rows")

    return price_data, institutional_data, margin_data, revenue_data


# Module-level helper to build the auxiliary Plotly figure. Made module-level so tests
# can import and inspect traces directly without invoking the script entrypoint.
def build_aux_fig(df_price: pd.DataFrame, df_rec: pd.DataFrame) -> Tuple[go.Figure, Dict[str, List[List]]]:
    """Build and return the auxiliary Plotly figure and data for JS dynamic rescaling."""
    aux_fig = make_subplots(
        rows=10, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.28, 0.10, 0.10, 0.10, 0.08, 0.08, 0.06, 0.06, 0.07, 0.07],
        subplot_titles=(
            'K線圖', '成交量 (Volume)', '日平均/每交易日營收 (daily_revenue)', '外資買賣超 (foreign_investor_net)',
            '投信買賣超 (investment_trust_net)', '自營商買賣超 (dealer_net)', '融資餘額增減 (MarginPurchaseBalanceChange)',
            '融券餘額增減 (ShortSaleBalanceChange)', '每股盈餘 (EPS)', '營業毛利 (Gross Profit 100M)'
        )
    )

    data_for_js = {}  # Dict to hold data for each row: {'row2': [[date_str, value], ...], ...}

    # compute SuperTrend on-the-fly when OHLC exist but indicator columns are missing
    try:
        ensure_supertrend(df_price)
    except Exception:
        pass

    # Row 1: K-line (candlestick) using price data if available
    if not df_price.empty and all(c in df_price.columns for c in ('open','high','low','close')):
        aux_fig.add_trace(go.Candlestick(
            x=df_price['date'],
            open=df_price['open'],
            high=df_price['high'],
            low=df_price['low'],
            close=df_price['close'],
            name='K-Line',
            increasing=dict(line=dict(color='#FF3232', width=1.8), fillcolor='rgba(255,50,50,0.12)'),
            decreasing=dict(line=dict(color='#00AB5E', width=1.8), fillcolor='rgba(0,171,94,0.12)'),
            showlegend=False
        ), row=1, col=1)
        # Disable rangeslider to allow x-axis syncing with other subplots
        aux_fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

        # Plot SuperTrend line if present (or computed earlier by ensure_supertrend)
        if 'supertrend' in df_price.columns and df_price['supertrend'].notna().any():
            aux_fig.add_trace(go.Scatter(
                x=df_price['date'],
                y=df_price['supertrend'],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='SuperTrend',
                hovertemplate='SuperTrend: %{y:.4f}<extra></extra>'
            ), row=1, col=1)

            # Plot buy/sell markers based on supertrend_dir when available
            if 'supertrend_dir' in df_price.columns:
                buys = df_price[df_price['supertrend_dir'] == 1]
                sells = df_price[df_price['supertrend_dir'] == -1]

                if not buys.empty:
                    aux_fig.add_trace(go.Scatter(
                        x=buys['date'],
                        y=buys['close'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', color='#FF3232', size=10),
                        name='SuperTrend Buy',
                        hovertemplate='Buy (SuperTrend): %{y:.2f}<extra></extra>'
                    ), row=1, col=1)

                if not sells.empty:
                    aux_fig.add_trace(go.Scatter(
                        x=sells['date'],
                        y=sells['close'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', color='#00AB5E', size=10),
                        name='SuperTrend Sell',
                        hovertemplate='Sell (SuperTrend): %{y:.2f}<extra></extra>'
                    ), row=1, col=1)

    # Row 2: Volume (from df_rec to ensure it's directly from the volume field in JSON)
    if not df_rec.empty and 'volume' in df_rec.columns:
        vol_x = df_rec['date']
        # Ensure volume is numeric to avoid scale issues
        vol_y = pd.to_numeric(df_rec['volume'], errors='coerce').fillna(0)
        if all(c in df_rec.columns for c in ('open','close')):
            vol_colors = ['#FF3232' if c >= o else '#00AB5E' for c, o in zip(df_rec['close'], df_rec['open'])]
        else:
            vol_colors = '#8884d8'
        aux_fig.add_trace(go.Bar(
            x=vol_x,
            y=vol_y,
            marker=dict(color=vol_colors, line=dict(width=0)),
            opacity=0.85,
            name='Volume',
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        ), row=2, col=1)
        # Fix: Show tick labels, use comma format, and force non-negative range to fix the -50M scale
        # aux_fig.update_yaxes(title_text='Volume', tickformat=',', row=2, col=1, 
                            #  autorange=True, rangemode='nonnegative', showgrid=True)
        aux_fig.update_yaxes(title_text='Volume', tickformat=',', row=2, col=1, 
                             rangemode='nonnegative', showgrid=True)
        
        # Prepare data for JS: list of [date_str, volume_value] pairs
        data_for_js['row2'] = [[str(d), float(v)] for d, v in zip(vol_x, vol_y)]
        
    # Row 3: daily_revenue (from df_rec if available)
    if not df_rec.empty and 'daily_revenue' in df_rec.columns:
        # Normalize daily_revenue to numeric and compute per-month averages
        df_rev = df_rec[['date', 'daily_revenue']].copy()
        df_rev['date'] = pd.to_datetime(df_rev['date'])
        df_rev['daily_revenue'] = pd.to_numeric(df_rev['daily_revenue'], errors='coerce').fillna(0)
        # Compute monthly average daily_revenue
        df_rev['month'] = df_rev['date'].dt.to_period('M')
        monthly_mean = df_rev.groupby('month')['daily_revenue'].mean()
        prev_mean = monthly_mean.shift(1)
        # Build month -> color mapping: red if month_mean > prev_month_mean, green if lower, grey for first month or equal
        month_color_map = {}
        for m in monthly_mean.index:
            pm = prev_mean.get(m, pd.NA)
            if pd.isna(pm):
                month_color_map[m] = '#8884d8'  # neutral for first month
            else:
                month_color_map[m] = '#FF3232' if monthly_mean.loc[m] > pm else ('#00AB5E' if monthly_mean.loc[m] < pm else '#8884d8')
        # Map color to each day
        rev_colors = [month_color_map.get(m, '#8884d8') for m in df_rev['month']]
        aux_fig.add_trace(go.Bar(
            x=df_rev['date'],
            y=df_rev['daily_revenue'],
            marker_color=rev_colors,
            name='daily_revenue',
            hovertemplate='daily_revenue: %{y:,.0f}<extra></extra>'
        ), row=3, col=1)
        aux_fig.update_yaxes(title_text='Daily Revenue', tickformat=',', row=3, col=1)

        # Prepare data for JS
        data_for_js['row3'] = [[str(d), float(v)] for d, v in zip(df_rev['date'], df_rev['daily_revenue'])]

    # Row 4: 外資 (Foreign)
    if not df_rec.empty and 'foreign_investor_net' in df_rec.columns:
        aux_fig.add_trace(go.Bar(
            x=df_rec['date'],
            y=df_rec['foreign_investor_net'],
            marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_rec['foreign_investor_net']],
            name='外資'
        ), row=4, col=1)
        aux_fig.update_yaxes(title_text='外資', tickformat=',', row=4, col=1)

        # Prepare data for JS
        data_for_js['row4'] = [[str(d), float(v)] for d, v in zip(df_rec['date'], df_rec['foreign_investor_net'])]

    # Row 5-6: 投信與自營商 (Investment Trust & Dealer) - two traces in separate rows for clarity
    if not df_rec.empty and 'investment_trust_net' in df_rec.columns:
        aux_fig.add_trace(go.Bar(
            x=df_rec['date'],
            y=df_rec['investment_trust_net'],
            marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_rec['investment_trust_net']],
            name='投信'
        ), row=5, col=1)
    aux_fig.update_yaxes(title_text='投信', tickformat=',', row=5, col=1)
    if not df_rec.empty and 'investment_trust_net' in df_rec.columns:
        # Prepare data for JS
        data_for_js['row5'] = [[str(d), float(v)] for d, v in zip(df_rec['date'], df_rec['investment_trust_net'])]
    if not df_rec.empty and 'dealer_net' in df_rec.columns:
        aux_fig.add_trace(go.Bar(
            x=df_rec['date'],
            y=df_rec['dealer_net'],
            marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_rec['dealer_net']],
            name='自營商'
        ), row=6, col=1)
    aux_fig.update_yaxes(title_text='自營商', tickformat=',', row=6, col=1)
    if not df_rec.empty and 'dealer_net' in df_rec.columns:
        # Prepare data for JS
        data_for_js['row6'] = [[str(d), float(v)] for d, v in zip(df_rec['date'], df_rec['dealer_net'])]

    # Row 7-8: margin/short balance changes
    if not df_rec.empty and 'MarginPurchaseBalanceChange' in df_rec.columns:
        m_vals = pd.to_numeric(df_rec['MarginPurchaseBalanceChange'], errors='coerce').fillna(0)
        m_colors = ['#FF3232' if v > 0 else ('#00AB5E' if v < 0 else '#8884d8') for v in m_vals]
        aux_fig.add_trace(go.Bar(
            x=df_rec['date'],
            y=m_vals,
            marker_color=m_colors,
            name='MarginPurchaseBalanceChange',
            hovertemplate='MarginPurchaseBalanceChange: %{y:,.0f}<extra></extra>'
        ), row=7, col=1)
    aux_fig.update_yaxes(title_text='融資增減', tickformat=',', row=7, col=1)
    if not df_rec.empty and 'MarginPurchaseBalanceChange' in df_rec.columns:
        # Prepare data for JS
        data_for_js['row7'] = [[str(d), float(v)] for d, v in zip(df_rec['date'], m_vals)]
    if not df_rec.empty and 'ShortSaleBalanceChange' in df_rec.columns:
        s_vals = pd.to_numeric(df_rec['ShortSaleBalanceChange'], errors='coerce').fillna(0)
        s_colors = ['#FF3232' if v > 0 else ('#00AB5E' if v < 0 else '#8884d8') for v in s_vals]
        aux_fig.add_trace(go.Bar(
            x=df_rec['date'],
            y=s_vals,
            marker_color=s_colors,
            name='ShortSaleBalanceChange',
            hovertemplate='ShortSaleBalanceChange: %{y:,.0f}<extra></extra>'
        ), row=8, col=1)
    aux_fig.update_yaxes(title_text='融券增減', tickformat=',', row=8, col=1)
    if not df_rec.empty and 'ShortSaleBalanceChange' in df_rec.columns:
        # Prepare data for JS
        data_for_js['row8'] = [[str(d), float(v)] for d, v in zip(df_rec['date'], s_vals)]

    # Row 9: EPS
    if not df_rec.empty and 'eps' in df_rec.columns:
        eps_vals = pd.to_numeric(df_rec['eps'], errors='coerce')
        # Color markers red for positive EPS and green for negative EPS
        eps_marker_colors = ['#FF3232' if v > 0 else '#00AB5E' for v in eps_vals.fillna(0)]
        aux_fig.add_trace(go.Scatter(
            x=df_rec['date'],
            y=eps_vals,
            mode='lines+markers',
            marker=dict(size=4, color=eps_marker_colors),
            name='EPS',
            line=dict(width=2, color='rgba(31,119,180,0.5)'),
            hovertemplate='EPS: %{y:.2f}<extra></extra>'
        ), row=9, col=1)
        aux_fig.update_yaxes(title_text='EPS', row=9, col=1)        

        # Prepare data for JS
        data_for_js['row9'] = [[str(d), float(v) if not pd.isna(v) else 0] for d, v in zip(df_rec['date'], eps_vals)]        

    # Row 10: Gross Profit
    if not df_rec.empty and 'gross_profit' in df_rec.columns:
        gp_vals = pd.to_numeric(df_rec['gross_profit'], errors='coerce').fillna(0)
        # Scale to 100M for readability
        gp_scaled = gp_vals / 1e8
        # Color bars red for positive gross profit, green for negative
        gp_colors = ['#FF3232' if v > 0 else '#00AB5E' for v in gp_scaled]
        aux_fig.add_trace(go.Bar(
            x=df_rec['date'],
            y=gp_scaled,
            marker_color=gp_colors,
            opacity=0.7,
            name='Gross Profit (100M)',
            hovertemplate='Gross Profit: %{y:.2f} 100M<extra></extra>'
        ), row=10, col=1)
        aux_fig.update_yaxes(title_text='毛利', row=10, col=1)

        # Prepare data for JS (use scaled values)
        data_for_js['row10'] = [[str(d), float(v)] for d, v in zip(df_rec['date'], gp_scaled)]

    # Remove gaps for non-trading days by computing calendar dates missing from the records
    if not df_rec.empty and 'date' in df_rec.columns:
        try:
            # Ensure dates are normalized to date-only values
            dates = pd.to_datetime(df_rec['date'])
            start = dates.min().date()
            end = dates.max().date()
            full_range = pd.date_range(start, end, freq='D')
            present = {d.date() for d in dates}
            missing = [d.strftime('%Y-%m-%d') for d in full_range if d.date() not in present]

            # Apply rangebreaks to all subplot x-axes (Plotly accepts a list of dicts)
            if missing:
                rb = [dict(values=missing)]
                for row in range(1, 11):
                    try:
                        aux_fig.update_xaxes(rangebreaks=rb, row=row, col=1)
                    except Exception:
                        pass
        except Exception:
            # Fail silently; do not break chart generation
            pass

    # Final layout: Use 'x unified' hovermode for easier comparison across rows
    aux_fig.update_layout(height=1200, template='plotly_white', showlegend=False, hovermode='x unified')
    return aux_fig, data_for_js

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate chart.html from a preprocessed JSON file')
    parser.add_argument('json_path', nargs='?', default=os.path.join(os.path.dirname(__file__), 'preprocessed_1727.json'), help='Path to preprocessed JSON')
    parser.add_argument('--stock_id', '-s', default=None, help='Stock id to display')
    args = parser.parse_args()

    # Load raw JSON and records
    raw = load_json_data(args.json_path)
    records = None
    if isinstance(raw, dict) and isinstance(raw.get('data'), list):
        records = raw['data']
    elif isinstance(raw, list):
        records = raw
    else:
        # unknown shape
        records = []

    # Derive metadata
    record_count = len(records)
    stock_id = args.stock_id or (records[0].get('stock_id') if record_count > 0 and isinstance(records[0], dict) else '')
    try:
        dates = [r.get('date') for r in records if isinstance(r, dict) and r.get('date')]
        start_date = min(dates) if dates else ''
        end_date = max(dates) if dates else ''
    except Exception:
        start_date = ''
        end_date = ''

    # Main chart generation intentionally removed per user request
    chart_html = None

    # Load Chinese label map from dict.md (if present)
    def load_label_map(md_path: str) -> dict:
        m = {}
        try:
            with open(md_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    # parse markdown table rows like: | key | 中文 |
                    if line.startswith('|') and '|' in line[1:]:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 3:
                            key = parts[1]
                            zh = parts[2]
                            if key != 'Key' and key != '---':
                                m[key] = zh
        except Exception:
            pass
        return m

    label_map = load_label_map(os.path.join(os.path.dirname(__file__), 'dict.md'))

    # Prepare sample keys/values from first record (show all keys found)
    sample = records[0] if record_count > 0 else {}
    def _fmt_val(v):
        if v is None:
            return ''
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
            return s if len(s) <= 300 else s[:300] + '...'
        return str(v)
    sample_keys = sorted(sample.keys())
    sample_rows = []
    for k in sample_keys:
        label = label_map.get(k, k)
        sample_rows.append({ 'key': k, 'label': label, 'value': _fmt_val(sample.get(k, '')) })

    # Embed the preprocessed JSON in the page for client-side download
    raw_json_str = json.dumps(records, ensure_ascii=False)

    # Build output file path
    out_fp = os.path.join(os.path.dirname(args.json_path), 'chart.html') if not os.path.isabs(args.json_path) else os.path.join(os.path.dirname(args.json_path), 'chart.html')

    # Write the dashboard HTML piecewise to avoid string-literal boundary issues
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html lang="zh-TW">\n<head>\n')
        f.write('  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n  <title>FinMind - Chart</title>\n')
        f.write('  <style> body { font-family: Arial, Helvetica, sans-serif; margin: 20px; } .container { max-width: 1400px; margin: 0 auto;} .panel{ padding:12px; background:#f6f6f6; border-radius:6px; margin-bottom:12px;} .controls{ display:flex; gap:12px; align-items:center; margin-bottom:12px;} label{ font-weight:600; margin-right:6px;} table{ border-collapse:collapse; width:100%; max-width:720px;} td, th{ border:1px solid #eaeaea; padding:8px;} .chart-container{ margin-top:18px; border:1px solid #eee; border-radius:8px; overflow:hidden; padding:8px;} </style>\n')
        f.write('</head>\n<body>\n')
        f.write('  <div class="container">\n')
        f.write(f'    <h1>📊 FinMind - Local Chart Preview</h1>\n')
        f.write(f'    <div class="panel"><strong>FinMind Usage:</strong> Parsed {record_count} records from <code>{os.path.basename(args.json_path)}</code></div>\n')

        f.write('    <form class="controls" onsubmit="return false;">\n')
        f.write(f'      <div><label for="stock_id">股票代碼：</label><input id="stock_id" value="{stock_id}" style="width:80px;" /></div>\n')
        f.write(f'      <div><label for="start_date">開始：</label><input id="start_date" value="{start_date}" /></div>\n')
        f.write(f'      <div><label for="end_date">結束：</label><input id="end_date" value="{end_date}" /></div>\n')
        f.write('    </form>\n')

        f.write('    <div class="chart-container">\n')
        f.write('      <!-- Main chart removed per configuration -->\n')

        # Build an auxiliary Plotly figure to show additional per-day series:
        # volume, daily_revenue, foreign_investor_net, investment_trust_net, dealer_net,
        # MarginPurchaseBalanceChange, ShortSaleBalanceChange
        # Use module-level build_aux_fig() defined above
        try:
            price_data, inst_data, margin_data, rev_data = prepare_data_for_chart(raw)
            df_price = pd.DataFrame(price_data)
            if not df_price.empty:
                df_price['date'] = pd.to_datetime(df_price['date'])
            df_rec = pd.DataFrame(records)
            if not df_rec.empty:
                df_rec['date'] = pd.to_datetime(df_rec['date'])

            aux_fig, data_for_js = build_aux_fig(df_price, df_rec)
            aux_html = aux_fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='aux-chart')
            # Decode any literal \\uXXXX unicode escapes so generated HTML contains real Unicode
            aux_html = re.sub(r'\\u([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), aux_html)
            f.write('\n')
            f.write(aux_html)
            
            # Embed data for JS dynamic rescaling
            data_json = json.dumps(data_for_js)
            f.write(f'''
            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                var chartData = {data_json};
                var plotDiv = document.getElementById('aux-chart');

                // Helper: extract x-axis range from relayout event payload in various formats
                function extractXRange(eventdata) {{
                    // Case 1: xaxis.range provided as array
                    if (eventdata['xaxis.range'] && eventdata['xaxis.range'].length === 2) {{
                        return [new Date(eventdata['xaxis.range'][0]).getTime(), new Date(eventdata['xaxis.range'][1]).getTime()];
                    }}
                    // Case 2: xaxis.range[0] / xaxis.range[1]
                    if (typeof eventdata['xaxis.range[0]'] !== 'undefined' && typeof eventdata['xaxis.range[1]'] !== 'undefined') {{
                        return [new Date(eventdata['xaxis.range[0]']).getTime(), new Date(eventdata['xaxis.range[1]']).getTime()];
                    }}
                    // Case 3: keys like 'xaxis2.range[0]' / 'xaxis2.range[1]' or 'xaxis2.range'
                    for (var k in eventdata) {{
                        if (k.match(/^xaxis\d*\.range\[0\]$/)) {{
                            var prefix = k.split('.range')[0];
                            var start = eventdata[prefix + '.range[0]'];
                            var end = eventdata[prefix + '.range[1]'];
                            if (typeof start !== 'undefined' && typeof end !== 'undefined') {{
                                return [new Date(start).getTime(), new Date(end).getTime()];
                            }}
                        }}
                        if (k.match(/^xaxis\d*\.range$/) && Array.isArray(eventdata[k]) && eventdata[k].length === 2) {{
                            return [new Date(eventdata[k][0]).getTime(), new Date(eventdata[k][1]).getTime()];
                        }}
                    }}
                    return null;
                }}

                if (plotDiv && Object.keys(chartData).length > 0) {{
                    plotDiv.on('plotly_relayout', function(eventdata) {{
                        // If relayout signals a full autorange reset, re-enable y autorange for all rows
                        var autorangeReset = false;
                        for (var k in eventdata) {{
                            if (k.match(/^xaxis\d*\.autorange$/) && eventdata[k] === true) {{
                                autorangeReset = true;
                                break;
                            }}
                            // some relayouts indicate axis cleared by providing null or undefined ranges
                            if (k.match(/^xaxis\d*\.range$/) && (eventdata[k] === null || eventdata[k] === undefined)) {{
                                autorangeReset = true;
                                break;
                            }}
                        }}

                        if (autorangeReset) {{
                            var resetObj = {{}};
                            for (var i = 2; i <= 10; i++) {{
                                resetObj['yaxis' + i + '.autorange'] = true;
                            }}
                            Plotly.relayout(plotDiv, resetObj);
                            return;
                        }}

                        var xRangeArr = extractXRange(eventdata);
                        if (!xRangeArr) {{
                            return;
                        }}
                        var xMin = xRangeArr[0];
                        var xMax = xRangeArr[1];

                        // Update y-axis for each row with data
                        Object.keys(chartData).forEach(function(rowKey) {{
                            var rowData = chartData[rowKey];
                            var visibleY = [];

                            rowData.forEach(function(point) {{
                                var dateTime = new Date(point[0]).getTime();
                                if (dateTime >= xMin && dateTime <= xMax) {{
                                    visibleY.push(point[1]);
                                }}
                            }});

                            if (visibleY.length > 0) {{
                                var yMin = Math.min.apply(null, visibleY);
                                var yMax = Math.max.apply(null, visibleY);
                                // handle flat lines (avoid zero padding)
                                if (yMax === yMin) {{
                                    var pad = Math.abs(yMax) * 0.02 || 1.0; // 2% or at least 1
                                    yMin -= pad;
                                    yMax += pad;
                                }} else {{
                                    var yPadding = (yMax - yMin) * 0.1;
                                    yMin -= yPadding;
                                    yMax += yPadding;
                                }}
                                // If all visible values are non-negative, clamp lower bound to 0 to avoid negative axis
                                if (Math.min.apply(null, visibleY) >= 0) {{
                                    yMin = Math.max(0, yMin);
                                }}

                                var rowNum = parseInt(rowKey.replace('row', ''));
                                var yAxisKey = rowNum === 1 ? 'yaxis' : 'yaxis' + rowNum;

                                var relayoutObj = {{}};
                                relayoutObj[yAxisKey + '.range'] = [yMin, yMax];
                                relayoutObj[yAxisKey + '.autorange'] = false;
                                Plotly.relayout(plotDiv, relayoutObj);
                            }}
                        }});
                    }});
                }}
            }});
            </script>
            ''')
        except Exception:
            # if anything fails, skip auxiliary chart cleanly
            pass

        f.write('\n    </div>\n')

        f.write('    <p style="text-align:center; color:#666; font-size:0.95em; margin-top:8px;">平均月營收/交易日 = 月營收 ÷ 該月份的交易日數（僅計算有交易日的月份）</p>\n')
        f.write('  </div>\n')
        f.write('</body>\n</html>\n')

    print(f"Chart saved to {out_fp}")
