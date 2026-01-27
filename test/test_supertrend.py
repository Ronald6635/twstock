import numpy as np
import pandas as pd

from engine.datasets.indicators import compute_supertrend


def make_price_series(n=60, start=100.0, step=0.5):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = start + np.arange(n) * step
    high = close + 0.2
    low = close - 0.2
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def test_supertrend_basic_monotonic_up():
    df = make_price_series(n=60, step=1.0)
    st, dir = compute_supertrend(df, period=7, multiplier=3.0)

    assert isinstance(st, pd.Series) and isinstance(dir, pd.Series)
    assert len(st) == len(df) and len(dir) == len(df)

    # final direction for a steady uptrend should be bullish (1)
    assert dir.dropna().iloc[-1] == 1
    # supertrend should be below price in sustained uptrend
    # initial values can be seeded above/below price while ATR warms up —
    # assert bullish relationship on the latter half of valid samples
    valid_idx = np.flatnonzero(~st.isna())
    if len(valid_idx) >= 4:
        tail = valid_idx[len(valid_idx) // 2 :]
        assert (df['close'].iloc[tail].values > st.iloc[tail].values).all()


def test_supertrend_handles_short_series_and_nans():
    df = make_price_series(n=5)
    # inject a NaN row in high
    df.loc[df.index[2], 'high'] = np.nan
    st, dir = compute_supertrend(df, period=3, multiplier=2.0)

    # returns same-length series and does not raise
    assert st.shape[0] == df.shape[0]
    assert dir.shape[0] == df.shape[0]
    # direction uses integers in {-1, 0, 1}
    assert set(np.unique(dir.dropna())).issubset({-1, 0, 1})
