import pandas as pd
from engine.datasets_ml.data_visual import build_aux_fig


def test_volume_trace_is_bar():
    # small synthetic price dataframe
    df_price = pd.DataFrame([
        {'date': '2026-01-01', 'open': 10, 'high': 12, 'low': 9, 'close': 11, 'volume': 100},
        {'date': '2026-01-02', 'open': 11, 'high': 13, 'low': 10, 'close': 12, 'volume': 200},
    ])
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_rec = pd.DataFrame()

    fig = build_aux_fig(df_price, df_rec)
    vol_traces = [t for t in fig.data if getattr(t, 'name', '') == 'Volume']
    assert vol_traces, 'Volume trace missing'
    assert vol_traces[0].type == 'bar'
