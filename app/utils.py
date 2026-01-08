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
import plotly.graph_objects as go

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

def generate_plotly_kline_chart(price_data, institutional_data, margin_data, revenue_data=None, stock_id=None):
    """
    Generate an interactive plotly K-line chart with institutional, margin, and monthly revenue overlays.

    Args:
        price_data (list): List of dicts with 'date', 'open', 'high', 'low', 'close', 'volume'
        institutional_data (list): List of dicts with 'date', 'buy', 'sell' (institutional)
        margin_data (list): List of dicts with 'date', 'margin_purchase', 'margin_sale', 'short_sale', 'short_covering'
        revenue_data (list|None): List of dicts with monthly revenue (fields: 'date' or 'revenue_year'/'revenue_month' and 'revenue')
        stock_id (str|None): Stock identifier

    Returns:
        str: HTML div containing the plotly chart
    """
    if not price_data:
        return "<p>No price data available</p>"

    # Convert to DataFrames
    df_price = pd.DataFrame(price_data)
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price = df_price.sort_values('date')

    # Get stock name if possible
    stock_name = stock_id
    try:
        if stock_id in twstock.codes:
            stock_name = f"{twstock.codes[stock_id].name} ({stock_id})"
    except:
        pass

    # Calculate moving averages
    for ma in [5, 10, 20, 60]:
        df_price[f'MA{ma}'] = df_price['close'].rolling(window=ma).mean()

    # Subplot structure: 1. K-line+MA, 2. Volume, 3. 外資, 4. 投信, 5. 融資, 6. 融券
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.36, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11],
        subplot_titles=(
            f'{stock_name} K線圖', '成交量 (Volume)', '平均月營收/交易日 (Avg Monthly Revenue per Trading Day)',
            '外資買賣超 (Foreign)', '投信買賣超 (Trust)', '融資餘額增減 (Margin)', '融券餘額增減 (Short)'
        )
    )

    # K-line
    fig.add_trace(go.Candlestick(
        x=df_price['date'],
        open=df_price['open'],
        high=df_price['high'],
        low=df_price['low'],
        close=df_price['close'],
        name='K-Line',
        increasing=dict(line=dict(color='#FF3232'), fillcolor='rgba(255,50,50,0.15)'),
        decreasing=dict(line=dict(color='#00AB5E'), fillcolor='rgba(0,171,94,0.12)'),
        showlegend=True
    ), row=1, col=1)

    # Moving averages
    colors = ['#FF9800', '#2196F3', '#9C27B0', '#795548'] # Better contrast colors
    for ma, color in zip([5, 10, 20, 60], colors):
        fig.add_trace(go.Scatter(
            x=df_price['date'],
            y=df_price[f'MA{ma}'],
            mode='lines',
            name=f'MA{ma}',
            line=dict(width=1.5, color=color),
            showlegend=True
        ), row=1, col=1)

    # Volume
    if 'volume' in df_price.columns:
        colors = ['#FF3232' if c >= o else '#00AB5E' for c, o in zip(df_price['close'], df_price['open'])]
        fig.add_trace(go.Bar(
            x=df_price['date'],
            y=df_price['volume'],
            marker_color=colors,
            name='成交量',
            showlegend=False
        ), row=2, col=1)

    # Average monthly revenue per trading day (插入第3列)
    if revenue_data:
        df_rev = pd.DataFrame(revenue_data)
        # Prefer explicit year/month fields to avoid ambiguities
        if 'revenue_year' in df_rev.columns and 'revenue_month' in df_rev.columns:
            df_rev['year_month'] = pd.to_datetime(df_rev['revenue_year'].astype(str) + '-' + df_rev['revenue_month'].astype(str) + '-01').dt.to_period('M')
        else:
            # fallback to using 'date' field
            if 'date' in df_rev.columns:
                df_rev['date'] = pd.to_datetime(df_rev['date'])
                df_rev['year_month'] = df_rev['date'].dt.to_period('M')
            else:
                df_rev['year_month'] = pd.NaT

        # Count trading days per month from price data
        df_price['year_month'] = df_price['date'].dt.to_period('M')
        td = df_price.groupby('year_month').size().reset_index(name='trading_days')

        # Aggregate revenue per month (some APIs may provide multiple rows per month)
        if 'revenue' in df_rev.columns:
            df_rev_grouped = df_rev.groupby('year_month', as_index=False).agg({'revenue':'sum'})
            df_rev_grouped = pd.merge(df_rev_grouped, td, on='year_month', how='left')

            # For months without trading day info (e.g., missing daily price rows), estimate trading days using business days
            import calendar
            est_flags = []
            for i, row in df_rev_grouped.iterrows():
                td_val = row.get('trading_days')
                if pd.isna(td_val) or td_val == 0:
                    # estimate using calendar business days for that month
                    ym = row['year_month']
                    if pd.isna(ym):
                        est_flags.append('')
                        df_rev_grouped.at[i, 'trading_days'] = None
                        continue
                    year = ym.year
                    month = ym.month
                    last_day = calendar.monthrange(year, month)[1]
                    start = pd.Timestamp(year, month, 1)
                    end = pd.Timestamp(year, month, last_day)
                    est_td = pd.bdate_range(start, end).size
                    df_rev_grouped.at[i, 'trading_days'] = est_td
                    est_flags.append('(估)')
                else:
                    est_flags.append('')

            df_rev_grouped['est_flag'] = est_flags

            df_rev_grouped['avg_rev_per_trading_day'] = df_rev_grouped.apply(
                lambda r: (r['revenue'] / r['trading_days']) if r['trading_days'] and r['trading_days'] > 0 else None,
                axis=1
            )

            # Prefer plotting at the **last trading day of that month** so rangebreaks (weekend hiding) don't drop the point.
            try:
                last_trade_map = df_price.groupby('year_month')['date'].max()
                df_rev_grouped['x'] = df_rev_grouped['year_month'].map(last_trade_map)
                # Fallback to month-end timestamp if we still don't have a trading date
                missing_x = df_rev_grouped['x'].isna()
                if missing_x.any():
                    df_rev_grouped.loc[missing_x, 'x'] = df_rev_grouped.loc[missing_x, 'year_month'].dt.to_timestamp('M')
                    df_rev_grouped.loc[missing_x, 'est_flag'] = df_rev_grouped.loc[missing_x, 'est_flag'].astype(str) + ' (no-trade)'
            except Exception:
                # if grouping fails, fallback to month-end timestamps
                df_rev_grouped['x'] = df_rev_grouped['year_month'].dt.to_timestamp('M')

            # Compute percent change and colors compared to previous month
            df_rev_grouped['pct_change'] = df_rev_grouped['avg_rev_per_trading_day'].pct_change() * 100
            df_rev_grouped['change_label'] = df_rev_grouped['pct_change'].apply(lambda x: (f"{x:+.2f}%") if pd.notna(x) else 'N/A')

            # Color: red if up vs previous, green if down, neutral blue for first/unchanged
            df_rev_grouped['color'] = '#4B8BF4'
            increase_mask = df_rev_grouped['avg_rev_per_trading_day'] > df_rev_grouped['avg_rev_per_trading_day'].shift(1)
            decrease_mask = df_rev_grouped['avg_rev_per_trading_day'] < df_rev_grouped['avg_rev_per_trading_day'].shift(1)
            df_rev_grouped.loc[increase_mask, 'color'] = '#FF3232'
            df_rev_grouped.loc[decrease_mask, 'color'] = '#00AB5E'

            # Add bar trace for averaged monthly revenue. Include est_flag and change_label in customdata for tooltip.
            fig.add_trace(go.Bar(
                x=df_rev_grouped['x'],
                y=df_rev_grouped['avg_rev_per_trading_day'],
                name='平均月營收/交易日',
                marker_color=df_rev_grouped['color'],
                customdata=df_rev_grouped[['trading_days', 'revenue', 'est_flag', 'change_label']].values,
                hovertemplate=(
                    '年份/月份: %{x|%Y-%m}<br>'
                    '平均月營收/交易日: %{y:.2s}<br>'
                    '月營收: %{customdata[1]:.2s}<br>'
                    '交易日數: %{customdata[0]} %{customdata[2]}<br>'
                    '較上月: %{customdata[3]}<extra></extra>'
                ),
                showlegend=False
            ), row=3, col=1)
        else:
            # no revenue field -> skip plotting
            pass

    # Process institutional data (外資/投信)
    if institutional_data:
        df_inst_raw = pd.DataFrame(institutional_data)
        df_inst_raw['date'] = pd.to_datetime(df_inst_raw['date'])

        # Defensive column detection: FinMind sometimes uses 'name' for investor type
        inst_col = next((c for c in df_inst_raw.columns if isinstance(c, str) and (c.lower() == 'institutional_investor' or 'institution' in c.lower() or 'investor' in c.lower() or c.lower() == 'name')), None)
        buy_col = next((c for c in df_inst_raw.columns if isinstance(c, str) and 'buy' == c.lower()), None)
        sell_col = next((c for c in df_inst_raw.columns if isinstance(c, str) and 'sell' == c.lower()), None)

        # Looser matches if exact names not found
        if inst_col is None:
            inst_col = next((c for c in df_inst_raw.columns if isinstance(c, str) and ('name' in c.lower() or 'type' in c.lower() or 'invest' in c.lower())), None)
        if buy_col is None:
            buy_col = next((c for c in df_inst_raw.columns if isinstance(c, str) and 'buy' in c.lower()), None)
        if sell_col is None:
            sell_col = next((c for c in df_inst_raw.columns if isinstance(c, str) and 'sell' in c.lower()), None)

        if inst_col is None or buy_col is None or sell_col is None:
            # Cannot reliably parse institutional data; create an empty aligned frame to keep plotting stable
            df_inst = pd.merge(df_price[['date']], pd.DataFrame({'date': df_price['date']}), on='date', how='left').fillna(0)
        else:
            # Normalize column names for pivot
            df_inst_raw = df_inst_raw.rename(columns={inst_col: 'Institutional_Investor', buy_col: 'buy', sell_col: 'sell'})

            try:
                # Aggregate institutional data: pivot from long to wide (summing buy/sell per investor per date)
                df_inst_pivot = df_inst_raw.pivot_table(
                    index='date',
                    columns='Institutional_Investor',
                    values=['buy', 'sell'],
                    aggfunc='sum'
                ).fillna(0)

                # Flatten multi-index columns
                df_inst_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_inst_pivot.columns]
                df_inst_pivot = df_inst_pivot.reset_index()

                # Merge with df_price to align dates
                df_inst = pd.merge(df_price[['date']], df_inst_pivot, on='date', how='left').fillna(0)
            except Exception:
                df_inst = pd.merge(df_price[['date']], pd.DataFrame({'date': df_price['date']}), on='date', how='left').fillna(0)

        # 外資 (Foreign_Investor)
        if 'Foreign_Investor_buy' in df_inst.columns and 'Foreign_Investor_sell' in df_inst.columns:
            df_inst['外資'] = df_inst['Foreign_Investor_buy'] - df_inst['Foreign_Investor_sell']
            fig.add_trace(go.Bar(
                x=df_inst['date'],
                y=df_inst['外資'],
                marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_inst['外資']],
                name='外資',
                showlegend=False
            ), row=4, col=1)
        else:
            # ensure blank series to keep layout consistent (no crash)
            df_inst['外資'] = 0

        # 投信 (Investment_Trust)
        if 'Investment_Trust_buy' in df_inst.columns and 'Investment_Trust_sell' in df_inst.columns:
            df_inst['投信'] = df_inst['Investment_Trust_buy'] - df_inst['Investment_Trust_sell']
            fig.add_trace(go.Bar(
                x=df_inst['date'],
                y=df_inst['投信'],
                marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_inst['投信']],
                name='投信',
                showlegend=False
            ), row=5, col=1)
        else:
            df_inst['投信'] = 0
            fig.add_trace(go.Bar(
                x=df_inst['date'],
                y=df_inst['投信'],
                marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_inst['投信']],
                name='投信',
                showlegend=False
            ), row=5, col=1)

    # 融資/融券
    if margin_data:
        df_margin_raw = pd.DataFrame(margin_data)
        df_margin_raw['date'] = pd.to_datetime(df_margin_raw['date'])
        df_margin_raw = df_margin_raw.sort_values('date')
        
        # Merge with df_price to align dates
        df_margin = pd.merge(df_price[['date']], df_margin_raw, on='date', how='left').fillna(0)

        # 融資增減: MarginPurchaseBuy - MarginPurchaseSell - MarginPurchaseCashRepayment
        if 'MarginPurchaseBuy' in df_margin.columns:
            # If TodayBalance exists, we can use diff, but often Buy-Sell is more immediate
            df_margin['融資'] = df_margin['MarginPurchaseBuy'] - df_margin['MarginPurchaseSell'] - df_margin.get('MarginPurchaseCashRepayment', 0)
            fig.add_trace(go.Bar(
                x=df_margin['date'],
                y=df_margin['融資'],
                marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_margin['融資']],
                name='融資',
                showlegend=False
            ), row=6, col=1)
        
        # 融券增減: ShortSaleBuy - ShortSaleSell - ShortSaleCashRepayment
        if 'ShortSaleBuy' in df_margin.columns:
            # For ShortSale, positive means more shorting (ShortSaleSell > ShortSaleBuy)
            df_margin['融券'] = df_margin['ShortSaleSell'] - df_margin['ShortSaleBuy'] - df_margin.get('ShortSaleCashRepayment', 0)
            fig.add_trace(go.Bar(
                x=df_margin['date'],
                y=df_margin['融券'],
                marker_color=['#FF3232' if v >= 0 else '#00AB5E' for v in df_margin['融券']],
                name='融券',
                showlegend=False
            ), row=7, col=1)

            # Add info icon and tooltip for 融資 and 融券 (zh-TW explanations)
            try:
                mid_idx = len(df_price) // 2
                mid_date = df_price['date'].iat[mid_idx]
            except Exception:
                mid_date = df_price['date'].iloc[0] if len(df_price) > 0 else None

            # 融資 tooltip (placed at row=6)
            if '融資' in df_margin.columns:
                max_abs_m = max(df_margin['融資'].max(), abs(df_margin['融資'].min()))
                y_pos_m = df_margin['融資'].max() if df_margin['融資'].max() > 0 else (df_margin['融資'].min() * 0.5 if df_margin['融資'].min() < 0 else 1)
                m_tooltip = (
                    "融資：當日融資買進減去賣出，反映投資人借入資金(融資)的變化；" 
                    "正值表示買超(籌碼增加)，負值表示賣超(籌碼減少)。"
                )
                fig.add_trace(go.Scatter(
                    x=[mid_date], y=[y_pos_m], mode='text', text=['ⓘ'],
                    textfont=dict(size=14, color='#666'), showlegend=False,
                    hovertemplate=m_tooltip + '<extra></extra>'
                ), row=6, col=1)

            # 融券 tooltip (placed at row=7)
            if '融券' in df_margin.columns:
                y_pos_s = df_margin['融券'].max() if df_margin['融券'].max() > 0 else (df_margin['融券'].min() * 0.5 if df_margin['融券'].min() < 0 else 1)
                s_tooltip = (
                    "融券：當日融券賣出減去買回，反映市場空方(融券)的動向；" 
                    "正值表示放空增加，負值表示買回減少空單。"
                )
                fig.add_trace(go.Scatter(
                    x=[mid_date], y=[y_pos_s], mode='text', text=['ⓘ'],
                    textfont=dict(size=14, color='#666'), showlegend=False,
                    hovertemplate=s_tooltip + '<extra></extra>'
                ), row=7, col=1)


    # UI/UX Enhancements
    fig.update_layout(
        height=1400,
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False, # Disable kline rangeslider to see all subplots better
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        spikedistance=-1, # Crosshair style
    )

    # Hide weekends for all x-axes
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]), # hide weekends
        ],
        showspikes=True, # Show vertical line on hover
        spikemode='across',
        spikethickness=1,
        spikedash='dash',
        spikecolor='#999999',
        gridcolor='#eeeeee'
    )

    # Format Y-axes
    fig.update_yaxes(gridcolor='#eeeeee', fixedrange=False)
    
    # Custom Y-axis formatting for Volume and chips (e.g., 10k instead of 10000)
    fig.update_layout(
        yaxis2_tickformat=".2s", # Volume
        yaxis3_tickformat=".2s", # 平均月營收/交易日
        yaxis4_tickformat=".2s", # 外資
        yaxis5_tickformat=".2s", # 投信
        yaxis6_tickformat=".2s", # 融資
        yaxis7_tickformat=".2s", # 融券
    )

    # Add range selector to the first subplot only
    fig.update_layout(
        xaxis1=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                font=dict(size=11),
                y=1.05
            ),
            type="date"
        )
    )

    # Add a tiny HTML comment to make it easy to locate the 平均月營收 row in tests/UI
    html_div = "<!-- avg_month_title: 平均月營收/交易日 -->\n" + fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
    return html_div

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