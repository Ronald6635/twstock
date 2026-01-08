import json
from pathlib import Path
from engine.datasets_ml import preprocessing


def _get_first_price_record(input_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    recs = data['datasets']['finmind_taiwan_stock_price']['records']
    return recs[0], recs


def _get_value_from_price(rec, keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def test_preprocess_merges_price_fields(tmp_path):
    input_fp = Path('engine/datasets_ml/聯詠-3034/combined_2025-01-01_2026-01-01_20260108T035618Z.json')
    assert input_fp.exists(), f"Input combined file not found: {input_fp}"

    out_json = tmp_path / 'preprocessed_test.json'
    out_csv = tmp_path / 'preprocessed_test.csv'

    # Run preprocessing
    preprocessing.preprocess_data(str(input_fp), str(out_json), str(out_csv))

    # Load original price records and preprocessed output
    first_price_rec, all_price_recs = _get_first_price_record(str(input_fp))

    with open(out_json, 'r', encoding='utf-8') as f:
        pre = json.load(f)
    pre_recs = pre.get('data', [])
    assert pre_recs, "No records in preprocessed output"

    # Build lookup by date for quick checks
    pre_by_date = {r['date']: r for r in pre_recs}

    # Check a sample of the first 10 price records (or fewer if not available)
    sample = all_price_recs[:10]
    for pr in sample:
        date = pr.get('date')
        assert date in pre_by_date, f"Date {date} missing in preprocessed output"
        out_rec = pre_by_date[date]

        # Keys to check and possible alternate names in source
        mappings = {
            'open': ['open', 'Open'],
            'high': ['high', 'High'],
            'low': ['low', 'Low'],
            'close': ['close', 'Close'],
            'volume': ['volume', 'Trading_turnover', 'Trading_Volume']
        }

        for dst_key, src_keys in mappings.items():
            src_val = _get_value_from_price(pr, src_keys)
            # If source has the value, preprocessed should have same value
            if src_val is not None:
                assert dst_key in out_rec, f"{dst_key} missing for date {date} in preprocessed output"
                assert out_rec[dst_key] == src_val, f"Value mismatch for {dst_key} on {date}: expected {src_val}, got {out_rec[dst_key]}"
            else:
                # If source didn't have value, it's okay for preprocessed to not have it
                pass

    # Also assert CSV was created and contains column headers for OHLC and volume
    assert out_csv.exists(), "Preprocessed CSV not created"
    csv_text = out_csv.read_text(encoding='utf-8')
    headers = csv_text.splitlines()[0]
    for h in ('open', 'high', 'low', 'close', 'volume'):
        assert h in headers, f"CSV missing column: {h}"