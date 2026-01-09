import json
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict

def preprocess_data(input_file, output_json, output_csv):
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    datasets = data['datasets']
    
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "中華化-1727", "combined_2025-01-01_2026-01-09_20260109T044505Z.json")
    output_json = os.path.join(base_dir, "preprocessed_1727.json")
    output_csv = os.path.join(base_dir, "preprocessed_1727.csv")
    preprocess_data(input_file, output_json, output_csv)
