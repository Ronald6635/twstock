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


def test_stock_info_cached(monkeypatch):
    # Ensure FINMIND_API_KEY is not required when cached
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    date = '2026-01-07'
    data = [{'symbol': 'TEST', 'name': 'Test Co.'}]
    write_dataset('all-all', f'{date}_finmind_stock_info.json', data)

    client = flask_app.test_client()
    resp = client.get('/api/finmind/stock_info')
    assert resp.status_code == 200
    assert resp.get_json() == data

    # Cleanup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(os.path.join(project_root, 'datasets', 'all-all'))


def test_daily_cached(monkeypatch):
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    start = '2024-01-01'
    end = '2024-01-05'
    stock = '9999'
    data = [
        {'date': '2024-01-01', 'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 1000},
        {'date': '2024-01-02', 'open': 106, 'high': 112, 'low': 102, 'close': 110, 'volume': 1200},
    ]
    # Ensure we write the file to the folder load_data_from_datasets expects
    import twstock
    code_info = twstock.codes.get(stock)
    company_name = code_info.name if code_info else stock
    folder_name = f"{company_name}-{stock}"
    path = write_dataset(folder_name, f'{start}_{end}_finmind_taiwan_stock_price.json', data)
    assert os.path.exists(path)

    from app.utils import load_data_from_datasets
    direct = load_data_from_datasets(stock, 'finmind_taiwan_stock_price', start, end)
    print('DIRECT LOAD:', direct)

    client = flask_app.test_client()
    resp = client.get(f'/api/finmind/{stock}?start_date={start}&end_date={end}')
    if resp.status_code != 200:
        print('BODY:', resp.get_data(as_text=True))
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'data' in j and isinstance(j['data'], list)
    assert j['data'] == data
    assert 'chart_image' in j

    # Cleanup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(os.path.join(project_root, 'datasets', f'{stock}-{stock}'))


def test_gold_cached(monkeypatch):
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    start = '2024-01-01'
    end = '2024-01-05'
    data = [{'date': start, 'price': 2000}, {'date': end, 'price': 2010}]
    write_dataset('gold-gold', f'{start}_{end}_finmind_gold_price.json', data)

    client = flask_app.test_client()
    resp = client.get(f'/api/finmind/gold_price?start_date={start}&end_date={end}')
    assert resp.status_code == 200
    assert resp.get_json() == data

    # Cleanup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(os.path.join(project_root, 'datasets', 'gold-gold'))


def test_crude_cached(monkeypatch):
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    data_id = 'WTI'
    start = '2024-01-01'
    end = '2024-01-05'
    data = [{'date': start, 'price': 70}, {'date': end, 'price': 71}]
    write_dataset(f'{data_id}-{data_id}', f'{start}_{end}_finmind_crude_oil_price.json', data)

    client = flask_app.test_client()
    resp = client.get(f'/api/finmind/crude_oil_price?data_id={data_id}&start_date={start}&end_date={end}')
    assert resp.status_code == 200
    assert resp.get_json() == data

    # Cleanup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(os.path.join(project_root, 'datasets', f'{data_id}-{data_id}'))


def test_translation_cached(monkeypatch):
    monkeypatch.delenv('FINMIND_API_KEY', raising=False)

    dataset = 'TaiwanStockDaily'
    date = '2026-01-07'
    data = {'date': '日期', 'open': '開盤價'}
    write_dataset(f'{dataset}-{dataset}', f'{date}_finmind_translation.json', data)

    client = flask_app.test_client()
    resp = client.get(f'/api/finmind/translation/{dataset}')
    assert resp.status_code == 200
    assert resp.get_json() == data

    # Cleanup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(os.path.join(project_root, 'datasets', f'{dataset}-{dataset}'))
