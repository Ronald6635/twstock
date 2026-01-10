"""
Preprocessing Module for FinMind Datasets

This module provides functions to preprocess combined FinMind datasets into
merged daily JSON and CSV files for analysis and visualization.

Key features:
- Merges multiple dataset types into daily records
- Handles institutional, margin, revenue, and financial data
- Filters data to trading dates
- Outputs structured JSON and CSV formats

Architecture notes:
- Supports backward compatibility with old 'datasets' key
- Transforms institutional data into net buy/sell figures
- Derives daily revenue from monthly aggregates
- Computes margin balance changes
"""

import json
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any


def preprocess_data(input_file: str, output_json: str, output_csv: str) -> None:
    """
    Preprocess combined FinMind datasets into merged daily records.

    Loads combined JSON data, filters to trading dates, transforms institutional
    and margin data, derives daily revenue, and outputs merged JSON and CSV files.

    Args:
        input_file: Path to the combined input JSON file
        output_json: Path for the output JSON file
        output_csv: Path for the output CSV file

    Returns:
        None

    Raises:
        KeyError: If input JSON lacks required 'cache' or 'datasets' key
        FileNotFoundError: If input file does not exist
    """
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support both 'cache' (new) and 'datasets' (old) keys for backward compatibility
    datasets = data.get('cache', data.get('datasets'))
    if datasets is None:
        raise KeyError("Input JSON must contain 'cache' or 'datasets' key")

    # Get trading dates from finmind_taiwan_stock_price
    trading_dates = set()
    for record in datasets['finmind_taiwan_stock_price']['records']:
        trading_dates.add(record['date'])
    trading_dates = sorted(trading_dates)

    # Filter all datasets to trading dates
    filtered_datasets = {}
    for key, value in datasets.items():
        if 'records' in value:
            if key == 'finmind_revenue':
                # Revenue is monthly, keep all
                filtered_records = value['records']
            else:
                filtered_records = [r for r in value['records'] if r['date'] in trading_dates]
            filtered_datasets[key] = {'source': value.get('source'), 'cached': value.get('cached'), 'records': filtered_records}

    # Transform finmind_institutional
    institutional_by_date = defaultdict(list)
    for record in filtered_datasets['finmind_institutional']['records']:
        institutional_by_date[record['date']].append(record)

    # Index price records by date so OHLC and volume can be merged into final output
    price_by_date = {}
    for record in filtered_datasets.get('finmind_taiwan_stock_price', {}).get('records', []):
        # normalize keys (some datasets use different casing)
        date = record.get('date') or record.get('Date')
        if not date:
            continue
        price_by_date[date] = record

    institutional_transformed = {}
    for date, records in institutional_by_date.items():
        nets = {}
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Support both 'cache' (new) and 'datasets' (old) keys for backward compatibility
    datasets = data.get('cache', data.get('datasets'))
    if datasets is None:
        raise KeyError("Input JSON must contain 'cache' or 'datasets' key")
    
    # Get trading dates from finmind_taiwan_stock_price
    trading_dates = set()
    for record in datasets['finmind_taiwan_stock_price']['records']:
        trading_dates.add(record['date'])
    trading_dates = sorted(trading_dates)
    
    # Filter all datasets to trading dates
    filtered_datasets = {}
    for key, value in datasets.items():
        if 'records' in value:
            if key == 'finmind_revenue':
                # Revenue is monthly, keep all
                filtered_records = value['records']
            else:
                filtered_records = [r for r in value['records'] if r['date'] in trading_dates]
            filtered_datasets[key] = {'source': value.get('source'), 'cached': value.get('cached'), 'records': filtered_records}
    
    # Transform finmind_institutional
    institutional_by_date = defaultdict(list)
    for record in filtered_datasets['finmind_institutional']['records']:
        institutional_by_date[record['date']].append(record)

    # Index price records by date so OHLC and volume can be merged into final output
    price_by_date = {}
    for record in filtered_datasets.get('finmind_taiwan_stock_price', {}).get('records', []):
        # normalize keys (some datasets use different casing)
        date = record.get('date') or record.get('Date')
        if not date:
            continue
        price_by_date[date] = record
    
    institutional_transformed = {}
    for date, records in institutional_by_date.items():
        nets = {}
        for rec in records:
            name = rec['name']
            net = rec['buy'] - rec['sell']
            if name == 'Foreign_Investor':
                nets['foreign_investor_net'] = net
            elif name == 'Investment_Trust':
                nets['investment_trust_net'] = net
            elif name in ['Dealer_self', 'Dealer_Hedging']:
                nets['dealer_net'] = nets.get('dealer_net', 0) + net
        institutional_transformed[date] = nets
    
    # Propagate finmind_revenue daily
    revenue_records = filtered_datasets['finmind_revenue']['records']
    revenue_by_month = defaultdict(list)
    for rec in revenue_records:
        month_key = f"{rec['revenue_year']}-{rec['revenue_month']:02d}"
        revenue_by_month[month_key].append(rec)
    
    trading_days_by_month = defaultdict(int)
    for date in trading_dates:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month_key = f"{date_obj.year}-{date_obj.month:02d}"
        trading_days_by_month[month_key] += 1
    
    revenue_transformed = {}
    for month, recs in revenue_by_month.items():
        trading_days = trading_days_by_month[month]
        for rec in recs:
            daily_revenue = rec['revenue'] / trading_days if trading_days > 0 else 0
            # Find trading dates in this month
            month_dates = [d for d in trading_dates if d.startswith(month)]
            for date in month_dates:
                if date not in revenue_transformed:
                    revenue_transformed[date] = {}
                revenue_transformed[date].update(rec)
                revenue_transformed[date]['daily_revenue'] = daily_revenue
    
    # Derive margin balance changes
    margin_records = filtered_datasets['finmind_margin']['records']
    margin_by_date = {rec['date']: rec for rec in margin_records}
    
    margin_transformed = {}
    prev_date = None
    for date in sorted(margin_by_date.keys()):
        rec = margin_by_date[date]
        if prev_date and prev_date in margin_by_date:
            prev_rec = margin_by_date[prev_date]
            rec['MarginPurchaseBalanceChange'] = rec['MarginPurchaseTodayBalance'] - prev_rec['MarginPurchaseTodayBalance']
            rec['ShortSaleBalanceChange'] = rec['ShortSaleTodayBalance'] - prev_rec['ShortSaleTodayBalance']
        else:
            rec['MarginPurchaseBalanceChange'] = None  # or 0, but None for missing
            rec['ShortSaleBalanceChange'] = None
        margin_transformed[date] = rec
        prev_date = date
    
    # Handle financial statements (EPS and Gross Profit)
    financial_transformed = {}
    if 'finmind_financial_statement' in datasets:
        fin_ds = datasets['finmind_financial_statement']
        # Prefer daily_financial_series from derived or records if already daily
        fin_records = fin_ds.get('derived', {}).get('daily_financial_series')
        if not fin_records:
            fin_records = fin_ds.get('records', [])
        
        for rec in fin_records:
            date = rec.get('date')
            if date:
                financial_transformed[date] = {
                    'eps': rec.get('eps'),
                    'gross_profit': rec.get('gross_profit')
                }
    
    # Merge all into final data
    final_data = []
    for date in trading_dates:
        entry = {'date': date}

        # Merge price data (open/high/low/close/volume) if available
        p = price_by_date.get(date)
        if p:
            # Accept multiple key names and normalize to lower-case keys
            entry['open'] = p.get('open') if 'open' in p else p.get('Open')
            entry['high'] = p.get('high') if 'high' in p else p.get('High')
            entry['low'] = p.get('low') if 'low' in p else p.get('Low')
            entry['close'] = p.get('close') if 'close' in p else p.get('Close')
            # volume may be 'volume' or 'Trading_turnover' etc.
            entry['volume'] = p.get('volume') if 'volume' in p else p.get('Trading_turnover') if 'Trading_turnover' in p else p.get('Trading_Volume')

        if date in institutional_transformed:
            entry.update(institutional_transformed[date])
        if date in revenue_transformed:
            entry.update(revenue_transformed[date])
        if date in margin_transformed:
            entry.update(margin_transformed[date])
        if date in financial_transformed:
            entry.update(financial_transformed[date])
        final_data.append(entry)
    
    # Output to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({'data': final_data}, f, ensure_ascii=False, indent=2)
    
    # Output to CSV
    df = pd.DataFrame(final_data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    """
    Command-line interface for preprocessing datasets.

    Parses arguments and calls preprocess_data with appropriate paths.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess finmind datasets into merged daily JSON/CSV files"
    )
    parser.add_argument(
        "input_file",
        help="Path to the combined input JSON file (e.g. ./華泰-2329/combined_...json)"
    )
    parser.add_argument(
        "--out-json",
        help="Output JSON path. Defaults to parent directory as preprocessed_<folder>.json",
        default=None,
    )
    parser.add_argument(
        "--out-csv",
        help="Output CSV path. Defaults to parent directory as preprocessed_<folder>.csv",
        default=None,
    )

    args = parser.parse_args()
    input_file = os.path.abspath(args.input_file)
    parent_dir = os.path.dirname(input_file)
    parent_name = os.path.basename(parent_dir) or "output"

    if args.out_json:
        output_json = args.out_json
    else:
        output_json = os.path.join(parent_dir, f"preprocessed_{parent_name}.json")

    if args.out_csv:
        output_csv = args.out_csv
    else:
        output_csv = os.path.join(parent_dir, f"preprocessed_{parent_name}.csv")

    print(f"Input: {input_file}")
    print(f"Writing outputs: {output_json}, {output_csv}")

    preprocess_data(input_file, output_json, output_csv)

    print("Done.")
