"""Technical indicators for the dataset pipeline.

This module implements the SuperTrend indicator used by the pipeline and
keeps the implementation small, well-documented and unit-testable.

Implementation details
- ATR: Wilder smoothing (EWMA with alpha=1/period, adjust=False)
- Bands: basic upper/lower bands from HL2 +/- multiplier * ATR
- Final bands: propagated according to standard SuperTrend rules
- Direction: 1 for bullish, -1 for bearish, 0 for unknown/insufficient data

References:
- Common SuperTrend implementations (public trading literature)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

__all__ = ["compute_supertrend"]


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> Tuple[pd.Series, pd.Series]:
    """Compute SuperTrend and direction for price series.

    Args:
        df: DataFrame containing high/low/close columns.
        period: ATR lookback period (must be >= 1).
        multiplier: ATR multiplier for band width.
        high_col/low_col/close_col: column names to use from ``df``.

    Returns:
        (supertrend, direction)
        - supertrend: float series (NaN where insufficient data)
        - direction: int series with values in {1, -1, 0}

    Raises:
        KeyError: if required columns are missing.
        ValueError: if period < 1 or df is empty.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
    if df.empty:
        raise ValueError("input DataFrame is empty")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder ATR (EWMA with alpha=1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    supertrend = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(0, index=df.index, dtype=int)

    first_idx = tr.first_valid_index()
    if first_idx is None:
        return supertrend, direction

    # Build ordered list of indices starting at first valid TR
    idxs = list(df.index[df.index.get_loc(first_idx):])
    i0 = idxs[0]
    final_ub.at[i0] = basic_ub.at[i0]
    final_lb.at[i0] = basic_lb.at[i0]
    # seed with bearish until price proves otherwise
    supertrend.at[i0] = final_ub.at[i0]
    direction.at[i0] = -1

    for i in idxs[1:]:
        prev_pos = df.index.get_loc(i) - 1
        prev_i = df.index[prev_pos]

        # propagate final upper band
        if (basic_ub.at[i] < final_ub.at[prev_i]) or (close.at[prev_i] > final_ub.at[prev_i]):
            final_ub.at[i] = basic_ub.at[i]
        else:
            final_ub.at[i] = final_ub.at[prev_i]

        # propagate final lower band
        if (basic_lb.at[i] > final_lb.at[prev_i]) or (close.at[prev_i] < final_lb.at[prev_i]):
            final_lb.at[i] = basic_lb.at[i]
        else:
            final_lb.at[i] = final_lb.at[prev_i]

        # determine trend
        if np.isfinite(supertrend.at[prev_i]) and supertrend.at[prev_i] == final_ub.at[prev_i]:
            # previously bearish
            if close.at[i] <= final_ub.at[i]:
                supertrend.at[i] = final_ub.at[i]
                direction.at[i] = -1
            else:
                supertrend.at[i] = final_lb.at[i]
                direction.at[i] = 1
        else:
            # previously bullish (or seeded bullish)
            if close.at[i] >= final_lb.at[i]:
                supertrend.at[i] = final_lb.at[i]
                direction.at[i] = 1
            else:
                supertrend.at[i] = final_ub.at[i]
                direction.at[i] = -1

    return supertrend, direction
