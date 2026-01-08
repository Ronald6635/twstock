import os
import pandas as pd

from engine.datasets_ml.data_visual import build_aux_fig, prepare_data_for_chart, load_json_data


def test_volume_values_match_price_source():
    js = load_json_data(os.path.join('engine', 'datasets_ml', 'preprocessed_3034.json'))
    # prepare_data_for_chart expects the preprocessed JSON shape
    price_data, inst, margin, rev = prepare_data_for_chart(js)
    assert len(price_data) > 0

    df_price = pd.DataFrame(price_data)
    df_price['date'] = pd.to_datetime(df_price['date'])
    # create a minimal df_rec to pass through
    records = js.get('data', []) if isinstance(js, dict) and 'data' in js else js
    df_rec = pd.DataFrame(records)
    if not df_rec.empty:
        df_rec['date'] = pd.to_datetime(df_rec['date'])

    fig = build_aux_fig(df_price, df_rec)
    vol_traces = [t for t in fig.data if getattr(t, 'name', '') == 'Volume']
    assert vol_traces, 'Volume trace not present in auxiliary figure'
    vol_y = list(vol_traces[0].y)
    # Compare first N (10) values to ensure they come from price source
    N = min(10, len(df_price))
    assert vol_y[:N] == list(df_price['volume'].iloc[:N]), 'Volume values do not match df_price["volume"]'


def test_volume_values_fallback_to_records():
    # df_price empty; df_rec has volume
    df_price = pd.DataFrame()
    df_rec = pd.DataFrame([
        {'date': '2026-01-01', 'volume': 101},
        {'date': '2026-01-02', 'volume': 202},
        {'date': '2026-01-03', 'volume': 303},
    ])
    df_rec['date'] = pd.to_datetime(df_rec['date'])

    fig = build_aux_fig(df_price, df_rec)
    vol_traces = [t for t in fig.data if getattr(t, 'name', '') == 'Volume']
    assert vol_traces, 'Volume trace not present in auxiliary figure (fallback)'
    vol_y = list(vol_traces[0].y)
    assert vol_y == list(df_rec['volume'])


def test_volume_not_same_as_price_close():
    # Sanity check: Volume bars should not contain the same values as price close data
    df_price = pd.DataFrame([
        {'date': '2026-01-01', 'open': 10, 'high': 12, 'low': 9, 'close': 11, 'volume': 100},
        {'date': '2026-01-02', 'open': 11, 'high': 13, 'low': 10, 'close': 12, 'volume': 200},
        {'date': '2026-01-03', 'open': 12, 'high': 14, 'low': 11, 'close': 13, 'volume': 300},
    ])
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_rec = pd.DataFrame()

    fig = build_aux_fig(df_price, df_rec)
    vol_traces = [t for t in fig.data if getattr(t, 'name', '') == 'Volume']
    assert vol_traces, 'Volume trace not present in auxiliary figure'
    vol_y = list(vol_traces[0].y)
    # Compare against close prices
    close_vals = list(df_price['close'])
    assert vol_y != close_vals, 'Volume values should not be identical to price close values'
