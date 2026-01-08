import os
import json
import shutil
from app import app as flask_app


def write_dataset(folder_name, filename, data):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', folder_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def test_save_with_cached_data(monkeypatch, tmp_path):
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    start = '2025-01-01'
    end = '2025-01-07'
    stock = '9999'

    # Prepare sample datasets
    price = [
        {'date': '2025-01-02', 'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000}
    ]
    inst = [
        {'date': '2025-01-02', 'investor_type': 'foreign', 'buy': 100, 'sell': 50, 'net': 50}
    ]
    margin = [
        {'date': '2025-01-02', 'margin_balance': 10000, 'short_balance': 500}
    ]
    revenue = [
        {'revenue_year': 2025, 'revenue_month': 1, 'revenue': 1000000}
    ]

    import twstock
    code_info = twstock.codes.get(stock)
    company_name = code_info.name if code_info else stock
    folder_name = f"{company_name}-{stock}"

    write_dataset(folder_name, f'{start}_{end}_finmind_taiwan_stock_price.json', price)
    write_dataset(folder_name, f'{start}_{end}_finmind_institutional.json', inst)
    write_dataset(folder_name, f'{start}_{end}_finmind_margin.json', margin)
    write_dataset(folder_name, f'{start}_{end}_finmind_revenue.json', revenue)

    client = flask_app.test_client()
    resp = client.post('/api/finmind/save_dashboard', json={'stock_id': stock, 'start_date': start, 'end_date': end})
    assert resp.status_code == 200
    j = resp.get_json()
    assert j.get('saved') is True
    files = j.get('files')
    assert isinstance(files, list) and len(files) == 2

    # Files should exist
    for p in files:
        assert os.path.exists(p)

    # Combined JSON should contain the revenue dataset and derived field
    with open(files[0], 'r', encoding='utf-8') as f:
        combined = json.load(f)
    assert 'datasets' in combined
    assert 'finmind_revenue' in combined['datasets']
    assert 'derived' in combined['datasets']['finmind_revenue']

    # Cleanup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # remove created datasets folder
    shutil.rmtree(os.path.join(project_root, 'datasets', folder_name))
    # remove engine datasets_ml folder for this company-stock
    shutil.rmtree(os.path.join(project_root, 'engine', 'datasets_ml', f"{company_name}-{stock}"))


def test_missing_params_returns_400():
    client = flask_app.test_client()
    resp = client.post('/api/finmind/save_dashboard', json={'stock_id': '1234'})
    assert resp.status_code == 400


def test_force_refresh_without_key_returns_500(monkeypatch):
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)
    client = flask_app.test_client()
    resp = client.post('/api/finmind/save_dashboard', json={'stock_id': '1234', 'start_date': '2025-01-01', 'end_date': '2025-02-01', 'force_refresh': True})
    assert resp.status_code == 500
