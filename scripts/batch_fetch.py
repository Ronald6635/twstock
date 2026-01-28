"""
Batch Fetch Script for FinMind Data

This script fetches Taiwan stock data from cache or FinMind API for multiple stocks
based on settings in settings.json. It saves combined data files for each stock.

Key features:
- Batch processing of multiple stocks
- Automatic cache loading and API fetching
- Combined data file generation
- Error handling with continuation

Usage:
    python scripts/batc_fech.py

Requires:
- settings.json in the same directory
- FINMIND_API_KEY environment variable for API access
"""

import os
import json
from datetime import datetime, timezone
import pandas as pd
from FinMind.data import DataLoader
import twstock
import requests
from app.utils import load_data_from_cache, save_data_to_cache, write_combined_files


def main():
    """
    Main batch fetch function.

    Loads settings, initializes API, and processes each stock.
    """
    # Load settings
    with open('scripts/settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    start_date = settings['start_date']
    end_date = settings['end_date']
    stocks = settings['stocks']

    # Initialize FinMind API if key available
    api_key = os.getenv('FINMIND_API_KEY')
    api = None
    if api_key:
        api = DataLoader()
        api.login_by_token(api_token=api_key)
        print("FinMind API initialized")
    else:
        print("No FINMIND_API_KEY found, will use cache only")

    # Process each stock
    for stock_id in stocks:
        print(f"\nProcessing stock: {stock_id}")

        # Determine company name
        try:
            company = twstock.codes.get(stock_id).name if stock_id in twstock.codes else stock_id
        except Exception:
            company = stock_id

        api_names = [
            'finmind_taiwan_stock_price',
            'finmind_institutional',
            'finmind_margin',
            'finmind_revenue',
            'finmind_financial_statement'
        ]

        combined = {
            'meta': {
                'stock_id': stock_id,
                'company': company,
                'start_date': start_date,
                'end_date': end_date,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'saved_by': 'user:batch',
                'api_names': api_names
            },
            'cache': {}
        }

        results = {}
        for api_name in api_names:
            data = None
            if not api:  # If no API, try cache only
                data = load_data_from_cache(stock_id, api_name, start_date, end_date)
                source = 'cache' if data is not None else None
            else:
                # Try cache first
                data = load_data_from_cache(stock_id, api_name, start_date, end_date)
                source = 'cache' if data is not None else None

                if data is None:
                    try:
                        if api_name == 'finmind_taiwan_stock_price':
                            df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
                            df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'}, inplace=True)
                        elif api_name == 'finmind_institutional':
                            df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=end_date)
                        elif api_name == 'finmind_margin':
                            df = api.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=end_date)
                        elif api_name == 'finmind_revenue':
                            df = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
                        elif api_name == 'finmind_financial_statement':
                            # Financial statements are typically reported quarterly; fetch range if provided
                            df = api.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date, end_date=end_date)

                        data = df.to_dict(orient='records')
                        if api_name != 'finmind_financial_statement':
                            save_data_to_cache(stock_id, data, api_name, start_date, end_date)
                        source = 'finmind'
                    except Exception as e:
                        results[api_name] = {'ok': False, 'error': str(e)}
                        continue

            if data is not None:
                combined['cache'][api_name] = {'source': source or 'cache', 'cached': source == 'cache', 'records': data}
                results[api_name] = {'ok': True}
            else:
                results[api_name] = {'ok': False, 'error': 'No data'}

        # Derive monthly aggregates for revenue (cc. generate_plotly_kline_chart)
        try:
            revenue_ds = combined['cache'].get('finmind_revenue', {})
            rev_records = revenue_ds.get('records') or []
            if rev_records:
                df_rev = pd.DataFrame(rev_records)
                if 'revenue_year' in df_rev.columns and 'revenue_month' in df_rev.columns:
                    df_rev['year_month'] = pd.to_datetime(df_rev['revenue_year'].astype(str) + '-' + df_rev['revenue_month'].astype(str) + '-01').dt.to_period('M')
                elif 'date' in df_rev.columns:
                    df_rev['date'] = pd.to_datetime(df_rev['date'])
                    df_rev['year_month'] = df_rev['date'].dt.to_period('M')

                df_price = load_data_from_cache(stock_id, 'finmind_taiwan_stock_price', start_date, end_date) or []
                df_price = pd.DataFrame(df_price)
                if not df_price.empty:
                    df_price['date'] = pd.to_datetime(df_price['date'])
                    df_price['year_month'] = df_price['date'].dt.to_period('M')
                    td = df_price.groupby('year_month').size().reset_index(name='trading_days')
                else:
                    td = pd.DataFrame(columns=['year_month', 'trading_days'])

                if 'revenue' in df_rev.columns:
                    # Group revenue by month (sum in case multiple entries per month)
                    df_rev_grouped = df_rev.groupby('year_month', as_index=False).agg({'revenue': 'sum'})
                    df_rev_grouped = pd.merge(df_rev_grouped, td, on='year_month', how='left')
                    df_rev_grouped['est_flag'] = df_rev_grouped['trading_days'].isna()
                    df_rev_grouped['trading_days'] = df_rev_grouped['trading_days'].fillna(0).astype(int)
                    df_rev_grouped['avg_per_trading_day'] = df_rev_grouped.apply(lambda r: (r['revenue'] / r['trading_days']) if r['trading_days'] and r['trading_days'] > 0 else None, axis=1)

                    monthly = []
                    for _, r in df_rev_grouped.iterrows():
                        monthly.append({
                            'year_month': str(r['year_month']),
                            'revenue': int(r['revenue']),
                            'trading_days': int(r['trading_days']) if not pd.isna(r['trading_days']) else None,
                            'avg_per_trading_day': float(r['avg_per_trading_day']) if not pd.isna(r['avg_per_trading_day']) else None
                        })

                    combined['cache'].setdefault('finmind_revenue', {})['derived'] = {'monthly_aggregates': monthly}
        except Exception:
            pass

        # Build backward-filled daily financial series (EPS: seasonal; GrossProfit: absolute)
        try:
            fin_ds = combined['cache'].get('finmind_financial_statement', {})
            fin_records = fin_ds.get('records') or []
            price_ds = combined['cache'].get('finmind_taiwan_stock_price', {})
            price_records = price_ds.get('records') or []
            if fin_records and price_records:
                df_fin = pd.DataFrame(fin_records)
                # Pivot long-form (type/value) into wide form if needed
                if 'type' in df_fin.columns and 'value' in df_fin.columns:
                    df_fin_wide = df_fin.pivot_table(index='date', columns='type', values='value', aggfunc='first').reset_index()
                else:
                    df_fin_wide = df_fin.copy()

                # Normalize column names and find EPS / GrossProfit columns
                eps_col = None
                gross_col = None
                for c in df_fin_wide.columns:
                    lc = c.lower()
                    if 'eps' == lc or lc.startswith('eps'):
                        eps_col = c
                    if lc in ('grossprofit','gross_profit','gross'):
                        gross_col = c

                # Prepare index aligned to trading dates and backward-fill
                df_fin_wide['date'] = pd.to_datetime(df_fin_wide['date'])
                df_fin_wide = df_fin_wide.set_index('date').sort_index()
                df_price = pd.DataFrame(price_records)
                df_price['date'] = pd.to_datetime(df_price['date'])
                idx = pd.Index(sorted(df_price['date'].unique()))
                fin_daily = df_fin_wide.reindex(idx).bfill()

                daily = []
                for ddate, row in fin_daily.iterrows():
                    rec = {'date': ddate.strftime('%Y-%m-%d')}
                    rec['eps'] = float(row.get(eps_col)) if eps_col and not pd.isna(row.get(eps_col)) else None
                    rec['gross_profit'] = float(row.get(gross_col)) if gross_col and not pd.isna(row.get(gross_col)) else None
                    daily.append(rec)

                combined['cache'].setdefault('finmind_financial_statement', {}).setdefault('derived', {})['daily_financial_series'] = daily
                # Save processed daily financial series to its own dataset cache
                save_data_to_cache(stock_id, daily, 'finmind_financial_statement', start_date, end_date)
        except Exception:
            pass

        # Persist combined files
        if combined['cache']:
            try:
                files = write_combined_files(company, stock_id, start_date, end_date, combined)
                print(f"Saved files: {files['json']}, {files['csv']}")
            except Exception as e:
                print(f"Failed to write combined files: {e}")

        # Print summary
        print(f"Summary for {stock_id}:")
        for api_name, res in results.items():
            if res['ok']:
                source = combined['cache'][api_name]['source']
                print(f"  {api_name}: OK (source: {source})")
            else:
                print(f"  {api_name}: FAILED - {res['error']}")

    # Show API usage info at the end
    if api_key:
        print("\nFetching API usage info...")
        url = "https://api.web.finmindtrade.com/v2/user_info"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            user_count = data.get('user_count') if isinstance(data, dict) else None
            api_request_limit = data.get('api_request_limit') if isinstance(data, dict) else None
            print(f"API Usage: user_count={user_count}, api_request_limit={api_request_limit}")
        except Exception as e:
            print(f"Failed to get API usage: {e}")
    else:
        print("\nNo API key provided, skipping usage info.")


if __name__ == '__main__':
    main()
