import os, sys, json
# allow running from scripts/ by adding project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import write_combined_files
from app.finmind import load_data_from_cache
import twstock

stock_id = '3034'
start_date = '2019-01-01'
end_date = '2025-09-30'

# Build combined similar to save_dashboard
company = twstock.codes.get(stock_id).name if stock_id in twstock.codes else stock_id
combined = {
    'meta': {
        'stock_id': stock_id,
        'company': company,
        'start_date': start_date,
        'end_date': end_date,
        'created_at': 'test',
        'saved_by': 'test',
        'api_names': []
    },
    'cache': {}
}

for api_name in ['finmind_taiwan_stock_price','finmind_financial_statement']:
    recs = load_data_from_cache(stock_id, api_name, start_date, end_date) or []
    combined['cache'][api_name] = {'source': 'cache', 'cached': True, 'records': recs}

# run the derived daily generation code (simplified)
from datetime import datetime
import pandas as pd
fin_records = combined['cache'].get('finmind_financial_statement', {}).get('records') or []
price_records = combined['cache'].get('finmind_taiwan_stock_price', {}).get('records') or []
if fin_records and price_records:
    df_fin = pd.DataFrame(fin_records)
    if 'type' in df_fin.columns and 'value' in df_fin.columns:
        df_fin_wide = df_fin.pivot_table(index='date', columns='type', values='value', aggfunc='first').reset_index()
    else:
        df_fin_wide = df_fin.copy()
    eps_col = None
    gross_col = None
    for c in df_fin_wide.columns:
        lc=c.lower()
        if lc=='eps' or lc.startswith('eps'):
            eps_col=c
        if lc in ('grossprofit','gross_profit','gross'):
            gross_col=c
    df_fin_wide['date']=pd.to_datetime(df_fin_wide['date'])
    df_fin_wide = df_fin_wide.set_index('date').sort_index()
    df_price = pd.DataFrame(price_records)
    df_price['date']=pd.to_datetime(df_price['date'])
    idx=pd.Index(sorted(df_price['date'].unique()))
    fin_daily = df_fin_wide.reindex(idx).bfill()
    daily=[]
    for ddate,row in fin_daily.iterrows():
        rec={'date': ddate.strftime('%Y-%m-%d')}
        rec['eps']=float(row.get(eps_col)) if eps_col and not pd.isna(row.get(eps_col)) else None
        rec['gross_profit']=float(row.get(gross_col)) if gross_col and not pd.isna(row.get(gross_col)) else None
        daily.append(rec)
    combined['cache'].setdefault('finmind_financial_statement', {}).setdefault('derived', {})['daily_financial_series']=daily

files = write_combined_files(company, stock_id, start_date, end_date, combined)
print('wrote', files)
# print sample of combined csv rows
print('CSV preview:')
print(open(files['csv'], encoding='utf-8-sig').read().splitlines()[:20])

# Check for rows that include eps or gross_profit
import pandas as pd
df = pd.read_csv(files['csv'], encoding='utf-8-sig')
mask = df['eps'].notna() | df['gross_profit'].notna()
print('rows with eps/gross (sample):', df[mask].head(5).to_dict('records'))
