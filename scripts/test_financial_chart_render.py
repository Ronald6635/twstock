import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import generate_financial_chart

folder = 'cache/聯詠-3034'
fin_path = os.path.join(folder, '2019-01-01_2025-09-30_finmind_financial_statement.json')
price_files = [p for p in os.listdir(folder) if p.endswith('_finmind_taiwan_stock_price.json')]
price_files.sort()
price_path = os.path.join(folder, price_files[-1]) if price_files else None

fin = json.load(open(fin_path, encoding='utf-8'))
price = json.load(open(price_path, encoding='utf-8')) if price_path else []
import pandas as pd

df_fin = pd.DataFrame(fin)
if 'type' in df_fin.columns and 'value' in df_fin.columns:
    df_fin_wide = df_fin.pivot_table(index='date', columns='type', values='value', aggfunc='first').reset_index()
else:
    df_fin_wide = df_fin.copy()

# find columns
eps_col = None
gross_col = None
for c in df_fin_wide.columns:
    lc = c.lower()
    if lc == 'eps' or lc.startswith('eps'):
        eps_col = c
    if lc in ('grossprofit','gross_profit','gross'):
        gross_col = c

if price:
    df_price = pd.DataFrame(price)
    df_price['date'] = pd.to_datetime(df_price['date'])
    idx = pd.Index(sorted(df_price['date'].unique()))
    df_fin_wide['date'] = pd.to_datetime(df_fin_wide['date'])
    df_fin_wide = df_fin_wide.set_index('date').sort_index()
    fin_daily = df_fin_wide.reindex(idx).bfill()

    daily = []
    for ddate, r in fin_daily.iterrows():
        daily.append({'date': ddate.strftime('%Y-%m-%d'), 'eps': float(r.get(eps_col)) if eps_col and not pd.isna(r.get(eps_col)) else None, 'gross_profit': float(r.get(gross_col)) if gross_col and not pd.isna(r.get(gross_col)) else None})
    html = generate_financial_chart(daily, '3034')
    print('generated_html_len=', len(html))
else:
    print('no price data')
