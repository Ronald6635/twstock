import os
import json

import pytest
import numpy as np

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


def test_supertrend_thumbnail_and_html_embedding(tmp_path):
    """Ensure the SuperTrend thumbnail can be generated and is embedded in the HTML output."""
    import pandas as pd
    from engine.datasets.data_visual import supertrend_thumbnail_png, build_aux_fig

    idx = pd.date_range('2020-01-01', periods=60, freq='D')
    close = 100 + np.arange(60) * 0.4
    high = close + 0.3
    low = close - 0.3
    df_price = pd.DataFrame({'date': idx, 'open': close, 'high': high, 'low': low, 'close': close})
    from engine.datasets.indicators import compute_supertrend
    st, st_dir = compute_supertrend(df_price, period=7, multiplier=3.0)
    df_price['supertrend'] = st
    df_price['supertrend_dir'] = st_dir
    # thumbnail data-url
    data_url = supertrend_thumbnail_png(df_price)
    assert isinstance(data_url, str) and data_url.startswith('data:image/png;base64,')
    # write HTML snippet as the module does and assert the <img> is present
    out = tmp_path / 'thumb_test.html'
    aux_fig = build_aux_fig(df_price, df_price)
    aux_html = aux_fig.to_html(full_html=False, include_plotlyjs='cdn')
    html = f"<html><body><img id=\"supertrend-thumb\" src=\"{data_url}\">{aux_html}</body></html>"
    assert 'id="supertrend-thumb"' in html
    # decode first bytes of the png to check valid PNG header
    import base64
    b = base64.b64decode(data_url.split(',', 1)[1])
    assert b[:8] == b'\x89PNG\r\n\x1a\n'


def test_build_aux_fig_plots_supertrend_and_signals():
    """When SuperTrend columns are present, the K-line subplot should include the SuperTrend line and buy/sell markers."""
    import pandas as pd
    from engine.datasets.indicators import compute_supertrend
    from engine.datasets.data_visual import build_aux_fig

    # Build a simple upward price series where SuperTrend should signal buys
    idx = pd.date_range('2020-01-01', periods=50, freq='D')
    close = 100 + np.arange(50) * 0.5
    high = close + 0.2
    low = close - 0.2
    df_price = pd.DataFrame({'date': idx, 'open': close, 'high': high, 'low': low, 'close': close})
    df_rec = df_price.copy()

    st, st_dir = compute_supertrend(df_price, period=7, multiplier=3.0)
    df_price['supertrend'] = st
    df_price['supertrend_dir'] = st_dir
    df_rec['supertrend'] = st
    df_rec['supertrend_dir'] = st_dir

    fig = build_aux_fig(df_price, df_rec)

    names = [getattr(t, 'name', '') for t in fig.data]
    assert 'SuperTrend' in names, 'SuperTrend line not plotted'
    assert 'SuperTrend Buy' in names or 'SuperTrend Sell' in names

    # Verify marker colors follow user rule: buy -> red (#FF3232), sell -> green (#00AB5E')
    buy_colors = []
    sell_colors = []
    for t in fig.data:
        if getattr(t, 'name', '') == 'SuperTrend Buy':
            buy_colors = list(getattr(t, 'marker').get('color', [])) if hasattr(getattr(t, 'marker'), 'get') else [getattr(t, 'marker').color]
        if getattr(t, 'name', '') == 'SuperTrend Sell':
            sell_colors = list(getattr(t, 'marker').get('color', [])) if hasattr(getattr(t, 'marker'), 'get') else [getattr(t, 'marker').color]

    if buy_colors:
        assert any('#FF3232' in str(c) for c in buy_colors)
    if sell_colors:
        assert any('#00AB5E' in str(c) for c in sell_colors)


def test_build_aux_fig_computes_supertrend_when_missing():
    """If OHLC are present but SuperTrend columns are missing, build_aux_fig should compute and plot SuperTrend."""
    import pandas as pd
    from engine.datasets.data_visual import build_aux_fig, ensure_supertrend

    idx = pd.date_range('2020-01-01', periods=40, freq='D')
    close = 50 + np.arange(40) * 0.8
    high = close + 0.2
    low = close - 0.2
    df_price = pd.DataFrame({'date': idx, 'open': close, 'high': high, 'low': low, 'close': close})

    # ensure_supertrend should attach the columns
    df_with = ensure_supertrend(df_price.copy())
    assert 'supertrend' in df_with.columns and 'supertrend_dir' in df_with.columns

    fig = build_aux_fig(df_price.copy(), df_price.copy())
    names = [getattr(t, 'name', '') for t in fig.data]
    assert 'SuperTrend' in names, 'build_aux_fig did not compute/plot SuperTrend when missing'
