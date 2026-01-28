"""Technical indicators for the dataset pipeline.

This module implements various technical indicators used in financial analysis,
including SuperTrend, Moving Average, RSI, Stochastic Oscillator, MACD, and OBV.
It provides standalone functions for computing indicators and a main function
for demonstration and visualization.

Key Features:
- Modular indicator functions with comprehensive type hints
- Support for custom column names and parameters
- Interactive plotting with Plotly (fallback to Matplotlib)
- JSON data loading with date filtering
- Command-line interface for standalone execution

Implementation Details:
- ATR: Wilder smoothing (EWMA with alpha=1/period, adjust=False)
- Bands: Basic upper/lower bands from HL2 +/- multiplier * ATR
- Final bands: Propagated according to standard SuperTrend rules
- Direction: 1 for bullish, -1 for bearish, 0 for unknown/insufficient data

References:
- Common SuperTrend implementations (public trading literature)
- Standard technical analysis formulas for other indicators

See Also:
- data_visual.py: For additional visualization utilities
- unified_pipeline.py: For data preprocessing pipeline
"""
from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Optional, Any, Dict # Add Optional, Any, Dict for type hints
import argparse # Import argparse for command-line arguments
import json # Import json for loading nested JSON structures
from datetime import datetime # Import datetime for date parsing

import numpy as np
import pandas as pd
import logging

if TYPE_CHECKING:
    from pandas import DataFrame, Series

__all__ = ["compute_supertrend", "compute_ma", "compute_rsi", "compute_stochastic", "compute_macd", "compute_obv"]


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute SuperTrend indicator and trend direction for price series.

    This implementation uses position-based indexing internally to avoid label/position
    ambiguity (which caused ValueError when DataFrame indices are datetimes).

    Returns:
        Tuple of two pandas Series indexed like the input `df`:
        - supertrend: Trend line values (NaN where insufficient data)
        - direction: Trend direction (1=bullish, -1=bearish, 0=insufficient data)
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
    if df.empty:
        raise ValueError("input DataFrame is empty")

    n = len(df)
    if n < period + 1:
        # Not enough data to compute meaningful SuperTrend: return NaNs / zeros
        return pd.Series(np.nan, index=df.index, dtype=float), pd.Series(0, index=df.index, dtype=int)

    # Preserve original index to return results aligned to it
    orig_index = df.index

    # Work with position-indexed series to avoid label/position mixups
    high = df[high_col].astype(float).reset_index(drop=True)
    low = df[low_col].astype(float).reset_index(drop=True)
    close = df[close_col].astype(float).reset_index(drop=True)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    hl2 = (high + low) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    # Use position-indexed Series for internal mutability
    supertrend_pos = pd.Series(np.nan, index=range(n), dtype=float)
    direction_pos = pd.Series(0, index=range(n), dtype=int)

    # Find first valid TR index (position)
    first_pos = tr.first_valid_index()
    if first_pos is None:
        return pd.Series(np.nan, index=orig_index, dtype=float), pd.Series(0, index=orig_index, dtype=int)

    i0 = int(first_pos)
    final_ub.iat[i0] = basic_ub.iat[i0]
    final_lb.iat[i0] = basic_lb.iat[i0]
    supertrend_pos.iat[i0] = final_ub.iat[i0]
    direction_pos.iat[i0] = -1

    for i in range(i0 + 1, n):
        prev_i = i - 1

        # Propagate final upper band
        if (basic_ub.iat[i] < final_ub.iat[prev_i]) or (close.iat[prev_i] > final_ub.iat[prev_i]):
            final_ub.iat[i] = basic_ub.iat[i]
        else:
            final_ub.iat[i] = final_ub.iat[prev_i]

        # Propagate final lower band
        if (basic_lb.iat[i] > final_lb.iat[prev_i]) or (close.iat[prev_i] < final_lb.iat[prev_i]):
            final_lb.iat[i] = basic_lb.iat[i]
        else:
            final_lb.iat[i] = final_lb.iat[prev_i]

        # Determine trend direction based on previous trend and current price
        if np.isfinite(supertrend_pos.iat[prev_i]) and supertrend_pos.iat[prev_i] == final_ub.iat[prev_i]:
            # Previously bearish
            if close.iat[i] <= final_ub.iat[i]:
                supertrend_pos.iat[i] = final_ub.iat[i]
                direction_pos.iat[i] = -1
            else:
                supertrend_pos.iat[i] = final_lb.iat[i]
                direction_pos.iat[i] = 1
        else:
            # Previously bullish
            if close.iat[i] >= final_lb.iat[i]:
                supertrend_pos.iat[i] = final_lb.iat[i]
                direction_pos.iat[i] = 1
            else:
                supertrend_pos.iat[i] = final_ub.iat[i]
                direction_pos.iat[i] = -1

    # Map positional results back to original index
    supertrend = pd.Series(supertrend_pos.values, index=orig_index, dtype=float)
    direction = pd.Series(direction_pos.values, index=orig_index, dtype=int)

    return supertrend, direction


def compute_ma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """
    Compute Simple Moving Average (SMA) for a specified column.

    The Simple Moving Average calculates the arithmetic mean of a series
    over a specified number of periods, providing a smoothed trend line.

    Args:
        df: DataFrame containing the data.
        period: Number of periods for the moving average calculation.
        col: Column name in df to compute the moving average for.

    Returns:
        Pandas Series containing the moving average values (NaN for initial periods).

    Raises:
        KeyError: If the specified column does not exist in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> data = {'close': [100, 102, 101, 103, 105]}
        >>> df = pd.DataFrame(data)
        >>> ma = compute_ma(df, period=3, col='close')
        >>> print(ma.iloc[-1])  # Moving average of last 3 values

    Note:
        - Returns NaN for the first (period-1) values where insufficient data exists
        - Commonly used for trend identification and support/resistance levels
        - Can be applied to any numeric column (price, volume, etc.)
    """
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")
    return df[col].rolling(window=period).mean()


def compute_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    """
    Compute Relative Strength Index (RSI) for a price series.

    RSI is a momentum oscillator that measures the speed and change of price
    movements, oscillating between 0 and 100. Values above 70 indicate
    overbought conditions, below 30 indicate oversold conditions.

    Args:
        df: DataFrame containing price data.
        period: Lookback period for RSI calculation (typically 14).
        col: Column name for price data (usually 'close').

    Returns:
        Pandas Series containing RSI values (0-100, NaN for initial periods).

    Raises:
        KeyError: If the specified column does not exist in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> data = {'close': [100, 102, 101, 99, 98, 100, 102, 105]}
        >>> df = pd.DataFrame(data)
        >>> rsi = compute_rsi(df, period=14, col='close')
        >>> print(rsi.iloc[-1])  # Latest RSI value

    Note:
        - Uses Wilder's smoothing method for gain/loss calculations
        - Values above 70 suggest overbought (potential sell signal)
        - Values below 30 suggest oversold (potential buy signal)
        - Requires sufficient data points for accurate calculation
    """
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, high_col: str = "high", low_col: str = "low", close_col: str = "close") -> Tuple[pd.Series, pd.Series]:
    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
    lowest_low = df[low_col].rolling(window=k_period).min()
    highest_high = df[high_col].rolling(window=k_period).max()
    k = 100 * ((df[close_col] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, col: str = "close") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence) indicator.

    MACD shows the relationship between two moving averages of a price,
    revealing momentum changes. It consists of the MACD line, signal line,
    and histogram showing the difference between them.

    Args:
        df: DataFrame containing price data.
        fast_period: Period for fast EMA (typically 12).
        slow_period: Period for slow EMA (typically 26).
        signal_period: Period for signal line EMA (typically 9).
        col: Column name for price data (usually 'close').

    Returns:
        Tuple of three pandas Series:
        - macd: MACD line (fast EMA - slow EMA)
        - signal: Signal line (EMA of MACD)
        - histogram: MACD - signal (momentum histogram)

    Raises:
        KeyError: If the specified column does not exist in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> data = {'close': [100, 102, 101, 103, 105, 104, 106, 108]}
        >>> df = pd.DataFrame(data)
        >>> macd, signal, hist = compute_macd(df, 12, 26, 9)
        >>> print(macd.iloc[-1], signal.iloc[-1], hist.iloc[-1])

    Note:
        - Bullish signal when MACD crosses above signal line
        - Bearish signal when MACD crosses below signal line
        - Histogram shows momentum strength and direction
        - Zero crossings indicate potential trend changes
    """
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")
    fast_ema = df[col].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[col].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def compute_obv(df: pd.DataFrame, volume_col: str = "volume", close_col: str = "close") -> pd.Series:
    for c in (volume_col, close_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
    obv = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df[close_col].iloc[i] > df[close_col].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df[volume_col].iloc[i]
        elif df[close_col].iloc[i] < df[close_col].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df[volume_col].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

def generate_trading_analysis(results_df: pd.DataFrame, sample_df: pd.DataFrame, period: int, multiplier: float) -> None:
    """
    Generate comprehensive trading analysis summary for the last 2 weeks.

    Analyzes all technical indicators and provides buy/sell recommendations
    based on the most recent signals and trends.

    Args:
        results_df: DataFrame containing all computed indicators
        sample_df: Original price data DataFrame
        period: ATR period used for SuperTrend
        multiplier: ATR multiplier used for SuperTrend
    """
    print("\n" + "="*80)
    print("📊 技術指標專業分析報告 (最近兩周)")
    print("="*80)

    # Get last 14 trading days (approximately 2 weeks)
    recent_data = results_df.tail(14)
    recent_prices = sample_df.tail(14)
    if len(recent_data) < 7:
        print("⚠️  數據不足，無法進行完整分析")
        return

    buy_signals = 0
    sell_signals = 0
    neutral_signals = 0
    analysis_points = []
    section_points = []

    # ⚡ 趨勢反轉偵測
    trend_reversal_msgs = []
    # SuperTrend反轉
    st_dir = recent_data['direction']
    st_reversal = st_dir.diff().fillna(0).abs().sum() > 0
    if st_reversal:
        trend_reversal_msgs.append("SuperTrend出現趨勢反轉")
    # MACD零軸穿越
    macd_hist = recent_data['macd_histogram']
    macd_cross = ((macd_hist * macd_hist.shift(1)) < 0).any()
    if macd_cross:
        trend_reversal_msgs.append("MACD柱狀圖出現多空翻轉")
    # RSI極端反轉
    rsi = recent_data['rsi_14']
    if (rsi.iloc[-2] < 30 and rsi.iloc[-1] > 30) or (rsi.iloc[-2] > 70 and rsi.iloc[-1] < 70):
        trend_reversal_msgs.append("RSI由極端區間反轉")
    if trend_reversal_msgs:
        print("\n⚡ 趨勢反轉偵測:")
        for msg in trend_reversal_msgs:
            print(f"   • {msg}")

    # 🔀 訊號一致性/分歧
    signal_votes = []
    # SuperTrend
    latest_direction = st_dir.iloc[-1]
    signal_votes.append(latest_direction)
    # RSI
    latest_rsi = rsi.iloc[-1]
    if latest_rsi > 70:
        signal_votes.append(-1)
    elif latest_rsi < 30:
        signal_votes.append(1)
    else:
        signal_votes.append(0)
    # Stochastic
    stoch_k = recent_data['stoch_k']
    latest_stoch_k = stoch_k.iloc[-1]
    if latest_stoch_k > 80:
        signal_votes.append(-1)
    elif latest_stoch_k < 20:
        signal_votes.append(1)
    else:
        signal_votes.append(0)
    # MACD
    latest_histogram = macd_hist.iloc[-1]
    if latest_histogram > 0:
        signal_votes.append(1)
    elif latest_histogram < 0:
        signal_votes.append(-1)
    else:
        signal_votes.append(0)
    # 均線
    latest_price = recent_prices['close'].iloc[-1]
    latest_ma20 = recent_data['ma_20'].iloc[-1]
    latest_ma60 = recent_data['ma_60'].iloc[-1]
    latest_ma120 = recent_data['ma_120'].iloc[-1]
    ma_vote = 1 if (latest_price > latest_ma20 and latest_price > latest_ma60 and latest_price > latest_ma120) else -1 if (latest_price < latest_ma20 and latest_price < latest_ma60 and latest_price < latest_ma120) else 0
    signal_votes.append(ma_vote)
    # 一致性判斷
    if abs(sum(signal_votes)) == len(signal_votes):
        print("\n✅ 訊號高度一致: 所有指標同向")
    elif abs(sum(signal_votes)) >= len(signal_votes) - 1:
        print("\n☑️ 訊號大致一致: 多數指標同向")
    else:
        print("\n🔀 訊號分歧: 多空訊號交錯，建議保守")

    # 🔎 量價結構
    if 'obv' in results_df.columns:
        obv = recent_data['obv']
        obv_trend = obv.iloc[-1] > obv.iloc[0]
        price_trend = recent_prices['close'].iloc[-1] > recent_prices['close'].iloc[0]
        if obv_trend and price_trend:
            print("\n🔎 量價同步: OBV與價格同創新高，趨勢健康")
        elif not obv_trend and not price_trend:
            print("\n🔎 量價同步: OBV與價格同創新低，空方趨勢明顯")
        else:
            print("\n❗ 量價背離: OBV與價格走勢不同步，需留意趨勢反轉風險")

    # 📅 多時框觀點
    print("\n📅 多時框均線分析:")
    ma_signals = []
    if latest_price > latest_ma20:
        ma_signals.append("短期(>MA20)上漲")
        buy_signals += 0.5
    else:
        ma_signals.append("短期(<MA20)下跌")
        sell_signals += 0.5
    if latest_price > latest_ma60:
        ma_signals.append("中期(>MA60)上漲")
        buy_signals += 0.5
    else:
        ma_signals.append("中期(<MA60)下跌")
        sell_signals += 0.5
    if latest_price > latest_ma120:
        ma_signals.append("長期(>MA120)上漲")
        buy_signals += 1
    else:
        ma_signals.append("長期(<MA120)下跌")
        sell_signals += 1
    print("   • " + ", ".join(ma_signals))
    if ma_vote == 1:
        print("   → 三線同多，趨勢強勁")
    elif ma_vote == -1:
        print("   → 三線同空，空方壓力大")
    else:
        print("   → 多空交錯，需觀察方向明朗化")

    # 🛡️ 停損/停利參考
    recent_high = recent_prices['high'].max()
    recent_low = recent_prices['low'].min()
    print("\n🛡️ 停損/停利參考:")
    print(f"   • 近兩週最高價(參考停利): {recent_high:.2f}")
    print(f"   • 近兩週最低價(參考停損): {recent_low:.2f}")

    # 🎯 關鍵觀察點
    print("\n🎯 關鍵觀察價位:")
    print(f"   • MA20: {latest_ma20:.2f}  MA60: {latest_ma60:.2f}  MA120: {latest_ma120:.2f}")
    if 'supertrend' in results_df.columns:
        latest_st = recent_data['supertrend'].iloc[-1]
        print(f"   • SuperTrend線: {latest_st:.2f}")
    prev_high = recent_prices['high'].iloc[:-1].max()
    prev_low = recent_prices['low'].iloc[:-1].min()
    print(f"   • 前高: {prev_high:.2f}  前低: {prev_low:.2f}")

    # 🚨 事件風險提醒
    abnormal_vol = False
    if 'volume' in recent_prices.columns:
        vol = recent_prices['volume']
        mean_vol = vol.mean()
        std_vol = vol.std()
        if ((vol > mean_vol + 2*std_vol).sum() > 0):
            abnormal_vol = True
    abnormal_move = False
    close = recent_prices['close']
    mean_ret = close.pct_change().abs().mean()
    if (close.pct_change().abs() > 2*mean_ret).sum() > 0:
        abnormal_move = True
    if abnormal_vol or abnormal_move:
        print("\n🚨 事件風險:")
        if abnormal_vol:
            print("   • 發現異常大量，留意消息面或主力動向")
        if abnormal_move:
            print("   • 發現異常大波動，警惕突發事件或公告")

    # 指標分析摘要
    print("\n🔍 指標分析摘要:")
    # SuperTrend
    supertrend_signal = "買入" if latest_direction == 1 else "賣出"
    analysis_points.append(f"SuperTrend趨勢: {supertrend_signal}訊號 (Period={period}, Multiplier={multiplier})")
    if latest_direction == 1:
        buy_signals += 2
    else:
        sell_signals += 2
    # RSI
    rsi_overbought = (rsi > 70).sum()
    rsi_oversold = (rsi < 30).sum()
    if latest_rsi > 70:
        analysis_points.append(f"RSI指標: 超買區間 ({latest_rsi:.1f}) - 賣出訊號")
        sell_signals += 1
    elif latest_rsi < 30:
        analysis_points.append(f"RSI指標: 超賣區間 ({latest_rsi:.1f}) - 買入訊號")
        buy_signals += 1
    else:
        analysis_points.append(f"RSI指標: 正常區間 ({latest_rsi:.1f})")
        neutral_signals += 1
    # Stochastic
    latest_stoch_d = recent_data['stoch_d'].iloc[-1]
    if latest_stoch_k > 80:
        analysis_points.append(f"Stochastic: 超買區間 (K:{latest_stoch_k:.1f}, D:{latest_stoch_d:.1f}) - 賣出訊號")
        sell_signals += 1
    elif latest_stoch_k < 20:
        analysis_points.append(f"Stochastic: 超賣區間 (K:{latest_stoch_k:.1f}, D:{latest_stoch_d:.1f}) - 買入訊號")
        buy_signals += 1
    else:
        analysis_points.append(f"Stochastic: 正常區間 (K:{latest_stoch_k:.1f}, D:{latest_stoch_d:.1f})")
        neutral_signals += 1
    # MACD
    latest_macd = recent_data['macd'].iloc[-1]
    latest_signal = recent_data['macd_signal'].iloc[-1]
    macd_trend = "上漲動能" if latest_macd > 0 else "下跌動能"
    macd_signal_strength = "買入訊號" if latest_histogram > 0 else "賣出訊號"
    analysis_points.append(f"MACD: {macd_trend}, 柱狀圖顯示{macd_signal_strength}")
    if latest_histogram > 0:
        buy_signals += 1
    else:
        sell_signals += 1
    # OBV
    if 'obv' in results_df.columns:
        obv_trend = recent_data['obv'].iloc[-1] > recent_data['obv'].iloc[-7]
        if obv_trend:
            analysis_points.append("OBV成交量: 上升趨勢，確認價格上漲")
            buy_signals += 0.5
        else:
            analysis_points.append("OBV成交量: 下降趨勢，確認價格下跌")
            sell_signals += 0.5
    # 價格趨勢
    price_trend = recent_prices['close'].iloc[-1] > recent_prices['close'].iloc[0]
    price_change_pct = ((recent_prices['close'].iloc[-1] - recent_prices['close'].iloc[0]) / recent_prices['close'].iloc[0]) * 100
    if price_trend:
        analysis_points.append(f"價格趨勢: 上漲 {price_change_pct:.1f}%")
    else:
        analysis_points.append(f"價格趨勢: 下跌 {price_change_pct:.1f}%")
    for point in analysis_points:
        print(f"   • {point}")

    # 綜合評分
    total_score = buy_signals - sell_signals
    print(f"\n📈 訊號統計:")
    print(f"   • 買入訊號強度: {buy_signals:.1f}")
    print(f"   • 賣出訊號強度: {sell_signals:.1f}")
    print(f"   • 中性訊號: {neutral_signals}")
    print(f"   • 綜合評分: {total_score:.1f} ({'偏多' if total_score > 0 else '偏空' if total_score < 0 else '中性'})")

    # 專業投資建議
    print(f"\n💡 專業投資建議:")
    current_price = recent_prices['close'].iloc[-1]
    if total_score >= 2:
        print("   🟢 強烈買入建議")
        print(f"   💰 當前價格: {current_price:.2f}")
        print("   📝 理由: 多數指標顯示上漲訊號，建議積極買入")
    elif total_score >= 0.5:
        print("   🟡 謹慎買入建議")
        print(f"   💰 當前價格: {current_price:.2f}")
        print("   📝 理由: 部分指標偏向上漲，可考慮逐步建倉")
    elif total_score >= -0.5:
        print("   🟠 觀望建議")
        print(f"   💰 當前價格: {current_price:.2f}")
        print("   📝 理由: 指標訊號混亂，建議觀望或減倉")
    elif total_score >= -2:
        print("   🔴 謹慎賣出建議")
        print(f"   💰 當前價格: {current_price:.2f}")
        print("   📝 理由: 部分指標顯示下跌訊號，建議逐步減倉")
    else:
        print("   🔴 強烈賣出建議")
        print(f"   💰 當前價格: {current_price:.2f}")
        print("   📝 理由: 多數指標顯示下跌訊號，建議積極賣出")

    # 風險提醒
    print(f"\n⚠️  風險提醒:")
    print("   • 技術分析僅供參考，不保證未來表現")
    print("   • 請結合基本面分析和風險承受能力做決定")
    print("   • 投資有風險，入市需謹慎")
    print("\n" + "="*80)

def main() -> None:
    """
    Main function for standalone execution of technical indicators.

    This function serves as a command-line interface for computing and visualizing
    technical indicators. It can load data from JSON files or generate synthetic
    data, compute SuperTrend indicators, and display interactive plots.

    Command-line Arguments:
        data_path: Optional path to JSON file containing stock data.
                  If not provided, synthetic data is generated.
        --period: ATR lookback period for SuperTrend (default: 14).
        --multiplier: ATR multiplier for SuperTrend bands (default: 2.5).
        --start_date: Start date for data filtering/generation (YYYY-MM-DD, default: "2023-01-01").
        --num_days: Number of days for synthetic data (default: 100, ignored if data_path provided).

    Returns:
        None: Results are printed to console and plots are displayed.

    Raises:
        SystemExit: If argument parsing fails or required data is missing.

    Example:
        Generate synthetic data and compute SuperTrend:
        >>> python indicators.py --period 10 --multiplier 3.0

        Load data from JSON file:
        >>> python indicators.py "data/stock_data.json" --start_date 2024-1-1

    Note:
        - Requires plotly for interactive plots, falls back to matplotlib
        - JSON data should contain OHLC columns and optionally volume
        - Synthetic data uses random walk for demonstration purposes
        - Plots show candlestick charts with SuperTrend overlay and signals
    """
    print("--- Running indicators.py as a standalone script ---")
    
    parser = argparse.ArgumentParser(
        description="Compute SuperTrend indicator with sample data or from a JSON file."
    )
    # Changed --data_path to a positional argument 'data_path' with nargs='?'
    # 'nargs="?"' means 0 or 1 argument. If not present, its value will be None.
    parser.add_argument(
        "data_path", # No leading dashes for a positional argument
        type=str,
        nargs='?', # Makes it optional; it will be None if not provided
        help="Optional: Path to a JSON file containing stock data. If not provided, synthetic data is generated.",
        default=None # Explicitly set default to None for clarity
    )
    parser.add_argument(
        "--period",
        type=int,
        default=14,
        help="ATR lookback period (default: 14)",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=2.5,
        help="ATR multiplier for band width (default: 2.5)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2023-01-01",
        help="Start date for generating sample data (YYYY-MM-DD, default: 2023-01-01). Ignored if data_path is used.",
    )
    parser.add_argument(
        "--num_days",
        type=int,
        default=100,
        help="Number of days for synthetic data generation (default: 100). Ignored if data_path is used."
    )
    
    args = parser.parse_args()

    sample_df: pd.DataFrame

    if args.data_path: # Now `args.data_path` will directly hold the string path or None
        # Load data from JSON file with flexible structure handling
        try:
            # NOTE: If your JSON structure is complex (like the output of data_analysis.prepare_analysis_data),
            # you might need to adjust this parsing logic.
            # For example, if it's `{ "df_metric": { "index": [...], "columns": [...], "data": [[...]] } }`
            # you'd need to reconstruct the DataFrame.
            # A simpler case is a JSON array of objects or object of arrays.
            
            # Load JSON data using json.load for nested structures
            with open(args.data_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # Handle different JSON structures: direct data, nested 'data' key, or 'df_metric' key
            if isinstance(loaded_data, dict) and 'data' in loaded_data:
                sample_df = pd.DataFrame(loaded_data['data'])
            elif isinstance(loaded_data, dict) and 'df_metric' in loaded_data:
                # Fallback for unified_pipeline's specific output structure if needed
                sample_df = pd.DataFrame(loaded_data['df_metric'])
            else:
                # Assume it's already a flat DataFrame or list of dicts
                sample_df = pd.DataFrame(loaded_data)
            
            # Ensure index is datetime for consistency with synthetic data
            if not isinstance(sample_df.index, pd.DatetimeIndex):
                # Attempt to convert a common 'Date' or 'timestamp' column to index
                if 'date' in sample_df.columns:
                    sample_df['date'] = pd.to_datetime(sample_df['date'])
                    sample_df = sample_df.set_index('date')
                elif 'timestamp' in sample_df.columns:
                    sample_df['timestamp'] = pd.to_datetime(sample_df['timestamp'])
                    sample_df = sample_df.set_index('timestamp')
                else:
                    print("Warning: JSON data does not have a datetime index or 'date'/'timestamp' column. Using default integer index.")

            # Filter data by start_date if provided (for loaded data)
            if args.start_date and isinstance(sample_df.index, pd.DatetimeIndex):
                try:
                    filter_start_date = pd.to_datetime(args.start_date)
                    original_rows = len(sample_df)
                    sample_df = sample_df[sample_df.index >= filter_start_date]
                    if len(sample_df) < original_rows:
                        print(f"Filtered loaded data from {original_rows} rows to {len(sample_df)} rows starting from {args.start_date}.")
                    elif len(sample_df) == 0:
                        print(f"Warning: No data available from {args.start_date}. Please check the start date or data range.")
                except ValueError:
                    print(f"Warning: Invalid date format for --start_date ('{args.start_date}') when filtering loaded data. Expected YYYY-MM-DD. Skipping date filter.")

            print(f"\nLoaded data from: {args.data_path}")
            # Ensure the DataFrame has the required columns
            required_cols = {'high', 'low', 'close'}
            if not required_cols.issubset(sample_df.columns):
                raise ValueError(f"Loaded JSON data is missing required columns: {required_cols - set(sample_df.columns)}")

        except Exception as e:
            print(f"Error loading or processing JSON data from {args.data_path}: {e}")
            print("Falling back to synthetic data generation.")
            args.data_path = None # Force synthetic data generation

    if not args.data_path: # If data_path was not provided or failed to load
        # Use parsed start_date
        try:
            start_date_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: Invalid date format for --start_date. Please use YYYY-MM-DD. Using default '2023-01-01'.")
            start_date_dt = datetime.strptime("2023-01-01", "%Y-%m-%d").date()

        dates: pd.DatetimeIndex = pd.date_range(start=start_date_dt, periods=args.num_days, freq="D")
        
        # Simulate price movements
        np.random.seed(42) # for reproducibility
        open_prices: np.ndarray = 100 + np.cumsum(np.random.randn(args.num_days))
        high_prices: np.ndarray = open_prices + np.random.rand(args.num_days) * 2
        low_prices: np.ndarray = open_prices - np.random.rand(args.num_days) * 2
        close_prices: np.ndarray = open_prices + np.random.randn(args.num_days) * 0.5
        
        # Ensure high >= low and close is within high/low
        high_prices = np.maximum(high_prices, close_prices)
        low_prices = np.minimum(low_prices, close_prices)
        high_prices = np.maximum(high_prices, open_prices)
        low_prices = np.minimum(low_prices, open_prices)

        sample_df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        }, index=dates)

    print("\nDataFrame Head for SuperTrend computation:")
    print(sample_df.head())

    # Compute SuperTrend using parsed arguments
    supertrend_series, direction_series = compute_supertrend(
        sample_df,
        period=args.period,
        multiplier=args.multiplier
    )

    # Compute additional indicators
    ma_series_20 = compute_ma(sample_df, period=20, col="close")
    ma_series_60 = compute_ma(sample_df, period=60, col="close")
    ma_series_120 = compute_ma(sample_df, period=120, col="close")
    rsi_series = compute_rsi(sample_df, period=14, col="close")
    stoch_k, stoch_d = compute_stochastic(sample_df, k_period=14, d_period=3)
    macd_line, macd_signal, macd_histogram = compute_macd(sample_df, fast_period=12, slow_period=26, signal_period=9, col="close")
    obv_series = compute_obv(sample_df, volume_col="volume", close_col="close") if 'volume' in sample_df.columns else None

    # Combine into a single DataFrame for easier viewing
    results_df: pd.DataFrame = pd.DataFrame({
        "close": sample_df["close"],
        "supertrend": supertrend_series,
        "direction": direction_series,
        "ma_20": ma_series_20,
        "ma_60": ma_series_60,
        "ma_120": ma_series_120,
        "rsi_14": rsi_series,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
    })
    if obv_series is not None:
        results_df["obv"] = obv_series

    print(f"\nSuperTrend Results (Period={args.period}, Multiplier={args.multiplier}):")
    print(results_df.tail(20)) # Print last 20 rows for better visualization of trends

    # Interactive plot for visual inspection (optional, prefers plotly, falls back to matplotlib)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplot layout: Price chart + multiple indicators
        subplot_titles = ['K線圖與SuperTrend及MA', '成交量', 'RSI(14)', 'Stochastic', 'MACD', 'OBV']
        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=subplot_titles
        )

        # Row 1: Candlestick chart with SuperTrend
        if {'open', 'high', 'low', 'close'}.issubset(sample_df.columns):
            fig.add_trace(go.Candlestick(
                x=sample_df.index,
                open=sample_df['open'],
                high=sample_df['high'],
                low=sample_df['low'],
                close=sample_df['close'],
                name='K-Line',
                increasing=dict(line=dict(color='#FF3232', width=1.8), fillcolor='rgba(255,50,50,0.12)'),
                decreasing=dict(line=dict(color='#00AB5E', width=1.8), fillcolor='rgba(0,171,94,0.12)'),
                showlegend=False
            ), row=1, col=1)

        # Plot SuperTrend
        fig.add_trace(go.Scatter(
            x=supertrend_series.index,
            y=supertrend_series,
            mode='lines',
            name='SuperTrend',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='<b>SuperTrend</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         'Signal: %{customdata}<extra></extra>',
            customdata=['買入信號 (價格在趨勢線上方)' if direction == 1 else '賣出信號 (價格在趨勢線下方)' for direction in results_df['direction']]
        ), row=1, col=1)

        # Plot Moving Averages
        fig.add_trace(go.Scatter(
            x=ma_series_20.index,
            y=ma_series_20,
            mode='lines',
            name='MA(20)',
            line=dict(color='blue', width=2),
            hovertemplate='<b>MA(20) - 短期均線</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '解釋: 20日平均價格，反映短期趨勢<br>' +
                         '訊號: 價格>MA20=短期上漲，價格<MA20=短期下跌<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ma_series_60.index,
            y=ma_series_60,
            mode='lines',
            name='MA(60)',
            line=dict(color='green', width=2),
            hovertemplate='<b>MA(60) - 中期均線</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '解釋: 60日平均價格，反映中期趨勢<br>' +
                         '訊號: 價格>MA60=中期上漲，價格<MA60=中期下跌<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ma_series_120.index,
            y=ma_series_120,
            mode='lines',
            name='MA(120)',
            line=dict(color='orange', width=2),
            hovertemplate='<b>MA(120) - 長期均線</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '解釋: 120日平均價格，反映長期趨勢<br>' +
                         '訊號: 價格>MA120=長期上漲，價格<MA120=長期下跌<extra></extra>'
        ), row=1, col=1)

        # Plot direction changes as markers
        long_signals = results_df[results_df['direction'] == 1].index
        short_signals = results_df[results_df['direction'] == -1].index
        
        fig.add_trace(go.Scatter(
            x=long_signals,
            y=sample_df.loc[long_signals]['close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='red', size=10),
            hovertemplate='<b>買入信號</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: %{y:.2f}<br>' +
                         '建議: 價格突破SuperTrend線上方，趨勢轉為上漲<br>' +
                         '操作: 考慮買入<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=short_signals,
            y=sample_df.loc[short_signals]['close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='green', size=10),
            hovertemplate='<b>賣出信號</b><br>' +
                         'Date: %{x}<br>' +
                         'Price: %{y:.2f}<br>' +
                         '建議: 價格跌破SuperTrend線下方，趨勢轉為下跌<br>' +
                         '操作: 考慮賣出<extra></extra>'
        ), row=1, col=1)

        fig.update_yaxes(title_text='Price', row=1, col=1)

        # Row 2: Volume (if available)
        if 'volume' in sample_df.columns:
            # Color volume bars based on price movement
            if 'open' in sample_df.columns and 'close' in sample_df.columns:
                vol_colors = ['#FF3232' if close >= open else '#00AB5E' for close, open in zip(sample_df['close'], sample_df['open'])]
            else:
                vol_colors = ['#8884d8'] * len(sample_df)
            
            fig.add_trace(go.Bar(
                x=sample_df.index,
                y=sample_df['volume'],
                marker_color=vol_colors,
                name='Volume',
                opacity=0.7
            ), row=2, col=1)
            fig.update_yaxes(title_text='Volume', row=2, col=1)

        # Row 3: RSI
        fig.add_trace(go.Scatter(
            x=rsi_series.index,
            y=rsi_series,
            mode='lines',
            name='RSI(14)',
            line=dict(color='purple', width=2),
            hovertemplate='<b>RSI(14) - 相對強弱指標</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '解釋: 衡量價格變動速度和幅度<br>' +
                         '訊號: >70=超買(賣出訊號)，<30=超賣(買入訊號)<br>' +
                         '當前狀態: %{customdata}<extra></extra>',
            customdata=['超買區間 (賣出訊號)' if x > 70 else '超賣區間 (買入訊號)' if x < 30 else '正常區間' for x in rsi_series]
        ), row=3, col=1)
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.update_yaxes(title_text='RSI(14)', range=[0, 100], row=3, col=1)

        # Row 4: Stochastic
        fig.add_trace(go.Scatter(
            x=stoch_k.index,
            y=stoch_k,
            mode='lines',
            name='%K',
            line=dict(color='orange', width=2),
            hovertemplate='<b>%K線 - 隨機指標</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '解釋: 價格在近期高低點間的位置<br>' +
                         '訊號: >80=超買，<20=超賣<br>' +
                         '當前狀態: %{customdata}<extra></extra>',
            customdata=['超買區間' if x > 80 else '超賣區間' if x < 20 else '正常區間' for x in stoch_k]
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=stoch_d.index,
            y=stoch_d,
            mode='lines',
            name='%D',
            line=dict(color='brown', width=2),
            hovertemplate='<b>%D線 - 隨機指標均線</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '解釋: %K的3日移動平均<br>' +
                         '訊號: %K交叉%D產生買賣訊號<br>' +
                         '當前狀態: %{customdata}<extra></extra>',
            customdata=['超買區間' if x > 80 else '超賣區間' if x < 20 else '正常區間' for x in stoch_d]
        ), row=4, col=1)
        # Add Stochastic reference lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        fig.update_yaxes(title_text='Stochastic', range=[0, 100], row=4, col=1)

        # Row 5: MACD
        fig.add_trace(go.Scatter(
            x=macd_line.index,
            y=macd_line,
            mode='lines',
            name='MACD',
            line=dict(color='green', width=2),
            hovertemplate='<b>MACD線 - 趨勢動量指標</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.4f}<br>' +
                         '解釋: 短期EMA減去長期EMA<br>' +
                         '訊號: >0=上漲動能，<0=下跌動能<br>' +
                         '當前狀態: %{customdata}<extra></extra>',
            customdata=['上漲動能' if x > 0 else '下跌動能' for x in macd_line]
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=macd_signal.index,
            y=macd_signal,
            mode='lines',
            name='Signal',
            line=dict(color='red', width=2),
            hovertemplate='<b>Signal線 - MACD訊號線</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.4f}<br>' +
                         '解釋: MACD的9日EMA<br>' +
                         '訊號: MACD交叉Signal產生買賣訊號<br>' +
                         '當前狀態: %{customdata}<extra></extra>',
            customdata=['上漲訊號' if x > 0 else '下跌訊號' for x in macd_signal]
        ), row=5, col=1)
        fig.add_trace(go.Bar(
            x=macd_histogram.index,
            y=macd_histogram,
            name='Histogram',
            marker_color=['green' if x >= 0 else 'red' for x in macd_histogram],
            opacity=0.7,
            hovertemplate='<b>MACD柱狀圖</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.4f}<br>' +
                         '解釋: MACD與Signal的差值<br>' +
                         '訊號: 綠色=買入訊號，紅色=賣出訊號<br>' +
                         '當前狀態: %{customdata}<extra></extra>',
            customdata=['買入訊號' if x >= 0 else '賣出訊號' for x in macd_histogram]
        ), row=5, col=1)
        fig.update_yaxes(title_text='MACD', row=5, col=1)

        # Row 6: OBV (if available)
        if obv_series is not None:
            fig.add_trace(go.Scatter(
                x=obv_series.index,
                y=obv_series,
                mode='lines',
                name='OBV',
                line=dict(color='cyan', width=2),
                hovertemplate='<b>OBV - 能量潮指標</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: %{y:,.0f}<br>' +
                             '解釋: 根據成交量確認價格趨勢<br>' +
                             '訊號: OBV上升=買入訊號，OBV下降=賣出訊號<br>' +
                             '當前狀態: %{customdata}<extra></extra>',
                customdata=['成交量確認上漲趨勢' if i > 0 and (obv_series.iloc[i] > obv_series.iloc[i-1] if i > 0 else True) else '成交量確認下跌趨勢' for i in range(len(obv_series))]
            ), row=6, col=1)
            fig.update_yaxes(title_text='OBV', row=6, col=1)
        else:
            # Hide OBV subplot if no volume data
            fig.update_yaxes(visible=False, row=6, col=1)
            fig.update_xaxes(visible=False, row=6, col=1)

        fig.update_layout(
            title=f"Technical Indicators Dashboard (SuperTrend Period={args.period}, Multiplier={args.multiplier})",
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            showlegend=True,
            height=1200  # Increase height for multiple subplots
        )
        
        # Respect robot automation flag: do not open interactive plots when running in automated mode
        import os as _os
        if not _os.environ.get('ROBOT_NO_PLOT'):
            fig.show()
        else:
            # In non-interactive mode we skip showing plots to avoid opening browsers or GUI
            logging.info('ROBOT_NO_PLOT set, skipping interactive Plotly show()')

    except ImportError:
        # Fallback to matplotlib if plotly not available
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True, gridspec_kw={'height_ratios': [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]})

            # Row 1: Candlestick chart with SuperTrend
            ax1 = axes[0]
            if {'open', 'high', 'low', 'close'}.issubset(sample_df.columns):
                # Plot candlestick manually
                for idx, row in sample_df.iterrows():
                    color = 'red' if row['close'] >= row['open'] else 'green'
                    # Body
                    ax1.bar(idx, row['close'] - row['open'], bottom=row['open'], width=0.6, color=color, alpha=0.7)
                    # Wick
                    ax1.vlines(idx, row['low'], row['high'], color=color, linewidth=1)
            else:
                # Fallback to line chart if no OHLC
                ax1.plot(sample_df.index, sample_df["close"], label="Close Price", color='blue', alpha=0.7)

            # Plot SuperTrend
            ax1.plot(supertrend_series.index, supertrend_series, label="SuperTrend", color='red', linestyle='--', alpha=0.8, linewidth=2)

            # Plot direction changes as markers
            long_signals = results_df[results_df['direction'] == 1].index
            short_signals = results_df[results_df['direction'] == -1].index
            
            ax1.scatter(long_signals, sample_df.loc[long_signals]['close'], marker='^', color='red', s=100, label='Buy Signal', alpha=1, zorder=5)
            ax1.scatter(short_signals, sample_df.loc[short_signals]['close'], marker='v', color='green', s=100, label='Sell Signal', alpha=1, zorder=5)

            ax1.set_title(f"Technical Indicators Dashboard (SuperTrend Period={args.period}, Multiplier={args.multiplier})")
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)

            # Row 2: Volume
            ax2 = axes[1]
            if 'volume' in sample_df.columns:
                # Color volume bars based on price movement
                if 'open' in sample_df.columns and 'close' in sample_df.columns:
                    vol_colors = ['#FF3232' if close >= open else '#00AB5E' for close, open in zip(sample_df['close'], sample_df['open'])]
                else:
                    vol_colors = ['#8884d8'] * len(sample_df)
                ax2.bar(sample_df.index, sample_df['volume'], color=vol_colors, alpha=0.7, width=0.6)
                ax2.set_ylabel("Volume")
                ax2.grid(True, linestyle='--', alpha=0.6)
            else:
                ax2.set_visible(False)

            # Row 3: RSI
            ax3 = axes[2]
            ax3.plot(rsi_series.index, rsi_series, label="RSI(14)", color='purple', linewidth=2)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax3.set_ylabel("RSI(14)")
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.6)

            # Row 4: Stochastic
            ax4 = axes[3]
            ax4.plot(stoch_k.index, stoch_k, label="%K", color='orange', linewidth=2)
            ax4.plot(stoch_d.index, stoch_d, label="%D", color='brown', linewidth=2)
            ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7)
            ax4.set_ylabel("Stochastic")
            ax4.set_ylim(0, 100)
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.6)

            # Row 5: MACD
            ax5 = axes[4]
            ax5.plot(macd_line.index, macd_line, label="MACD", color='green', linewidth=2)
            ax5.plot(macd_signal.index, macd_signal, label="Signal", color='red', linewidth=2)
            # Plot histogram
            colors = ['green' if x >= 0 else 'red' for x in macd_histogram]
            ax5.bar(macd_histogram.index, macd_histogram, color=colors, alpha=0.7, width=0.6)
            ax5.set_ylabel("MACD")
            ax5.legend()
            ax5.grid(True, linestyle='--', alpha=0.6)

            # Row 6: OBV
            ax6 = axes[5]
            if obv_series is not None:
                ax6.plot(obv_series.index, obv_series, label="OBV", color='cyan', linewidth=2)
                ax6.set_ylabel("OBV")
                ax6.legend()
                ax6.grid(True, linestyle='--', alpha=0.6)
            else:
                ax6.set_visible(False)

            plt.xlabel("Date")
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            # Respect robot automation flag: avoid opening GUI when automated
            import os as _os
            if not _os.environ.get('ROBOT_NO_PLOT'):
                plt.show()
            else:
                logging.info('ROBOT_NO_PLOT set, skipping matplotlib plt.show()')

        except ImportError:
            print("\nNeither plotly nor matplotlib installed. Skipping plot generation.")
        except Exception as e:
            print(f"\nError generating matplotlib plot: {e}")

    # Generate trading analysis summary for the last 2 weeks
    generate_trading_analysis(results_df, sample_df, args.period, args.multiplier)

if __name__ == "__main__":
    main()
