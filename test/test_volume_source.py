import os
import json

import pytest

from engine.datasets.data_visual import prepare_data_for_chart, load_json_data


def test_prepare_data_for_chart_prefers_price_volume():
    # Load the real preprocessed file and verify price_data volumes match the file
    here = os.path.dirname(__file__)
    sample_path = os.path.join(here, '..', 'engine', 'datasets', 'preprocessed_3034.json')
    sample_path = os.path.normpath(sample_path)
    raw = load_json_data(sample_path)

    price_data, inst, margin, rev = prepare_data_for_chart(raw)
    assert len(price_data) > 0
    # First record in the attached sample has volume 2423114
    assert price_data[0]['volume'] == 2423114


def test_volume_fallback_to_records():
    # When price entries are not provided separately, records' volume should be used
    small = {
        'data': [
            {'date': '2026-01-01', 'open': 10, 'high': 13, 'low': 9, 'close': 12, 'volume': 111},
            {'date': '2026-01-02', 'open': 12, 'high': 15, 'low': 11, 'close': 13, 'volume': 222},
        ]
    }

    price_data, *_ = prepare_data_for_chart(small)
    assert len(price_data) == 2
    assert [r['volume'] for r in price_data] == [111, 222]


def test_build_aux_fig_uses_records_volume():
    """Ensure the Volume bar in the auxiliary figure uses volumes directly from the records DataFrame."""
    import pandas as pd
    from engine.datasets.data_visual import build_aux_fig

    here = os.path.dirname(__file__)
    sample_path = os.path.join(here, '..', 'engine', 'datasets', 'preprocessed_3034.json')
    sample_path = os.path.normpath(sample_path)
    raw = load_json_data(sample_path)

    price_data, *_ = prepare_data_for_chart(raw)
    df_price = pd.DataFrame(price_data)
    if not df_price.empty:
        df_price['date'] = pd.to_datetime(df_price['date'])

    records = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw
    df_rec = pd.DataFrame(records)
    if not df_rec.empty:
        df_rec['date'] = pd.to_datetime(df_rec['date'])

    fig = build_aux_fig(df_price, df_rec)

    # Find the Bar trace named 'Volume' and assert y-values match the records' volume
    vol_y = None
    for t in fig.data:
        if getattr(t, 'type', '') == 'bar' and getattr(t, 'name', '') == 'Volume':
            vol_y = list(t.y)
            break

    assert vol_y is not None, "No Volume bar trace found in auxiliary figure"
    assert vol_y[:6] == df_rec['volume'].head(6).tolist()
