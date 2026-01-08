import os
import shutil
import json
import time
from app.utils import load_data_from_datasets


def test_load_data_covering_file_returns_filtered_subset(tmp_path):
    stock_id = '9986'
    api_name = 'finmind_test'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    # file covers a wide range
    data = [
        {'date': '2021-12-31', 'v': 0},
        {'date': '2022-03-05', 'v': 1},
        {'date': '2023-07-01', 'v': 2},
        {'date': '2024-11-01', 'v': 3},
        {'date': '2024-12-01', 'v': 4}
    ]
    with open(os.path.join(folder, f'2019-01-01_2026-01-07_{api_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f)

    loaded = load_data_from_datasets(stock_id, api_name, '2022-03-05', '2024-11-01')
    assert isinstance(loaded, list)
    # Expect only the three entries within range (inclusive)
    assert [r['v'] for r in loaded] == [1, 2, 3]

    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_multiple_covering_selects_smallest_span(tmp_path):
    stock_id = '9985'
    api_name = 'finmind_test'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    wide = [
        {'date': '2022-03-05', 'src': 'wide'},
        {'date': '2024-11-01', 'src': 'wide'}
    ]
    narrow = [
        {'date': '2022-03-05', 'src': 'narrow'},
        {'date': '2024-11-01', 'src': 'narrow'}
    ]
    with open(os.path.join(folder, f'2019-01-01_2026-01-07_{api_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(wide, f)
    with open(os.path.join(folder, f'2022-01-01_2024-12-31_{api_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(narrow, f)

    loaded = load_data_from_datasets(stock_id, api_name, '2022-03-05', '2024-11-01')
    assert isinstance(loaded, list)
    # narrow should be selected
    assert all(r.get('src') == 'narrow' for r in loaded)

    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_partial_overlap_returns_none(tmp_path):
    stock_id = '9984'
    api_name = 'finmind_test'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    # This file ends before the requested end
    partial = [
        {'date': '2022-03-05'},
        {'date': '2023-01-01'}
    ]
    with open(os.path.join(folder, f'2022-03-05_2023-01-01_{api_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(partial, f)

    loaded = load_data_from_datasets(stock_id, api_name, '2022-03-05', '2024-11-01')
    assert loaded is None

    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_ttl_applies_to_covering_file(tmp_path):
    stock_id = '9983'
    api_name = 'finmind_test'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    data = [
        {'date': '2022-03-05', 'v': 1},
        {'date': '2024-11-01', 'v': 2}
    ]
    fpath = os.path.join(folder, f'2019-01-01_2026-01-07_{api_name}.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    # Make file older than TTL
    old = time.time() - (86400 * 10)
    os.utime(fpath, (old, old))

    loaded = load_data_from_datasets(stock_id, api_name, '2022-03-05', '2024-11-01', max_age_days=1)
    assert loaded is None

    try:
        shutil.rmtree(folder)
    except Exception:
        pass


def test_corrupt_covering_file_returns_none(tmp_path):
    stock_id = '9982'
    api_name = 'finmind_test'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(project_root, 'datasets', f'{stock_id}-{stock_id}')
    os.makedirs(folder, exist_ok=True)

    fpath = os.path.join(folder, f'2019-01-01_2026-01-07_{api_name}.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('{ not valid json')

    loaded = load_data_from_datasets(stock_id, api_name, '2022-03-05', '2024-11-01')
    assert loaded is None

    try:
        shutil.rmtree(folder)
    except Exception:
        pass
