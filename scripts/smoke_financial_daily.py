import glob
import json
import pandas as pd
import os

folder = 'cache/聯詠-3034'
fin_path = os.path.join(folder, '2019-01-01_2025-09-30_finmind_financial_statement.json')
print('financial file exists?', os.path.exists(fin_path))
price_files = glob.glob(os.path.join(folder, '*_finmind_taiwan_stock_price.json'))
print('price files found:', len(price_files))
if price_files:
    price = json.load(open(price_files[-1], encoding='utf-8'))
    df_price = pd.DataFrame(price)
    print('price rows:', len(df_price))
else:
    df_price = None

fin = json.load(open(fin_path, encoding='utf-8'))
df_fin = pd.DataFrame(fin)
print('financial rows:', len(df_fin))

if 'type' in df_fin.columns and 'value' in df_fin.columns:
    df_fin_wide = df_fin.pivot_table(index='date', columns='type', values='value', aggfunc='first').reset_index()
else:
    df_fin_wide = df_fin.copy()

print('columns after pivot:', list(df_fin_wide.columns)[:20])

# find EPS and GrossProfit columns
eps_col = None
gross_col = None
for c in df_fin_wide.columns:
    lc = c.lower()
    if lc == 'eps' or lc.startswith('eps'):
        eps_col = c
    if lc in ('grossprofit','gross_profit','gross'):
        gross_col = c
print('eps_col:', eps_col, 'gross_col:', gross_col)

if df_price is not None and not df_price.empty:
    df_price['date'] = pd.to_datetime(df_price['date'])
    idx = pd.Index(sorted(df_price['date'].unique()))
    df_fin_wide['date'] = pd.to_datetime(df_fin_wide['date'])
    df_fin_wide = df_fin_wide.set_index('date').sort_index()
    fin_daily = df_fin_wide.reindex(idx).bfill()
    print('\nfin_daily head:\n', fin_daily[[eps_col, gross_col]].head(5))
    print('\nfin_daily tail:\n', fin_daily[[eps_col, gross_col]].tail(5))
else:
    print('No price data to align with.')
