import json
import os, sys
from pprint import pprint
# Ensure repo root is on sys.path so we can import app.finmind when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.finmind import _process_financial_data

p = 'cache/聯詠-3034/2019-01-01_2025-09-30_finmind_financial_statement.json'
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

latest_eps, latest_gross, series = _process_financial_data(data)
print('latest_eps, latest_gross =', latest_eps, latest_gross)
print('series length:', len(series))
print('recent rows:')
for i, r in enumerate(series[:8]):
    print(i, r)
