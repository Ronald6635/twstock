import os
import shutil
import json
import time
from app.utils import save_data_to_cache, load_data_from_cache


def test_save_data_prefix(tmp_path):
    stock_id = '9999'
    api_name = 'finmind_test'
    start_date = '2025-01-01'
    end_date = '2025-01-31'
    response_data = [{'sample': 1}]

    # Run the function (it writes to cache/<company>-<stock_id>)
    save_data_to_cache(stock_id, response_data, api_name, start_date, end_date)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'cache', f'{stock_id}-{stock_id}')
    expected_json = os.path.join(folder, f'{start_date}_{end_date}_{api_name}.json')
    expected_csv = os.path.join(folder, f'{start_date}_{end_date}_{api_name}.csv')

    assert os.path.exists(expected_json)
    # CSV may or may not exist depending on content; if it exists check content
    if os.path.exists(expected_csv):
        assert os.path.isfile(expected_csv)

    # Cleanup
    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_load_data_exact_match(tmp_path):
    stock_id = '9998'
    api_name = 'finmind_test'
    start_date = '2025-02-01'
    end_date = '2025-02-10'
    data = [{'a': 1}, {'b': 2}]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'cache', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    target = os.path.join(folder, f'{start_date}_{end_date}_{api_name}.json')
    with open(target, 'w', encoding='utf-8') as f:
        import json
        json.dump(data, f)

    loaded = load_data_from_cache(stock_id, api_name, start_date, end_date)
    assert loaded == data

    # Cleanup
    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_load_data_corrupt(tmp_path):
    stock_id = '9997'
    api_name = 'finmind_test'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'cache', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    target = os.path.join(folder, f'2025-03-01_{api_name}.json')
    with open(target, 'w', encoding='utf-8') as f:
        f.write('{ this is not valid json ')

    loaded = load_data_from_cache(stock_id, api_name)
    assert loaded is None

    # Cleanup
    try:
        shutil.rmtree(folder)
    except Exception:
        pass
