import os, json, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app as flask_app

# write helper

def write_dataset(folder_name, filename, data):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'cache', folder_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return path

start = '2025-01-01'
end = '2025-01-07'
stock = '9999'
price = [{'date':'2025-01-02','open':100,'high':110,'low':90,'close':105,'volume':1000}]
inst = [{'date':'2025-01-02','investor_type':'foreign','buy':100,'sell':50,'net':50}]
margin = [{'date':'2025-01-02','margin_balance':10000,'short_balance':500}]
revenue = [{'revenue_year':2025,'revenue_month':1,'revenue':1000000}]

import twstock
code_info = twstock.codes.get(stock)
company_name = code_info.name if code_info else stock
folder_name = f"{company_name}-{stock}"

write_dataset(folder_name, f'{start}_{end}_finmind_taiwan_stock_price.json', price)
write_dataset(folder_name, f'{start}_{end}_finmind_institutional.json', inst)
write_dataset(folder_name, f'{start}_{end}_finmind_margin.json', margin)
write_dataset(folder_name, f'{start}_{end}_finmind_revenue.json', revenue)

client = flask_app.test_client()
resp = client.post('/api/finmind/save_dashboard', json={'stock_id':stock,'start_date':start,'end_date':end})
print('STATUS', resp.status_code)
print(resp.get_json())

# cleanup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    shutil.rmtree(os.path.join(project_root, 'cache', folder_name))
except Exception:
    pass
try:
    shutil.rmtree(os.path.join(project_root, 'engine', 'datasets', f"{company_name}-{stock}"))
except Exception:
    pass
