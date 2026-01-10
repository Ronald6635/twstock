"""
Data Analysis Preparation Module

This module reuses the data parsing logic from `data_visual.py` to prepare
pandas DataFrames and summary statistics for downstream analysis. It does
NOT produce any charts; its goal is to provide clean, numeric-ready
structures (dataframes, monthly aggregations, stats, and correlations)
that other analysis or visualization modules can consume.

Key features:
- Loads and parses preprocessed JSON data
- Computes basic statistics for financial metrics
- Generates correlation matrices
- Prepares monthly aggregations

Architecture notes:
- Depends on data_visual.py for parsing logic
- Returns structured results for analysis
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try importing parsing helpers from data_visual.py with a resilient fallback
try:
    from engine.datasets.data_visual import load_json_data, prepare_data_for_chart
except Exception:  # pragma: no cover - tolerant import fallback
    # Add the project root (two levels up from this file) so 'engine' package can be imported
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from engine.datasets.data_visual import load_json_data, prepare_data_for_chart
    except Exception as err:
        raise ModuleNotFoundError(
            "Failed to import parsing helpers from 'engine.datasets.data_visual'. "
            "Ensure you're running from project root or that PYTHONPATH includes the project." 
        ) from err


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric, coercing errors to NaN.

    Args:
        s: Input pandas Series to convert.

    Returns:
        Series with numeric values, NaN for invalid entries.
    """
    return pd.to_numeric(s, errors='coerce')


def compute_basic_stats(s: pd.Series) -> Dict[str, Optional[float]]:
    """
    Compute basic statistics for a numeric pandas Series.

    Args:
        s: Input pandas Series (numeric).

    Returns:
        Dictionary with count, mean, median, std, min, max, and sign counts.
    """
    if s is None or s.dropna().empty:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'positive_count': 0,
            'negative_count': 0,
            'zero_count': 0
        }
    s = _safe_to_numeric(s).dropna()
    return {
        'count': int(s.size),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'std': float(s.std(ddof=0)) if s.size > 0 else None,
        'min': float(s.min()),
        'max': float(s.max()),
        'positive_count': int((s > 0).sum()),
        'negative_count': int((s < 0).sum()),
        'zero_count': int((s == 0).sum())
    }


def prepare_analysis_data(
    raw: Any,
) -> Dict[str, Any]:
    """
    Prepare dataframes and summary values from preprocessed FinMind-like data.

    This function loads and processes financial data from various sources,
    creating structured DataFrames for analysis and computing basic statistics.
    It provides correlation analysis.

    Args:
        raw: Either a path to a JSON file (str), a list of records, or a dict
             (as loaded from JSON). Supports wrapper format {"data": [...]}.

    Returns:
        A dictionary with the following keys:
            - df_price: DataFrame with columns ['date','open','high','low','close','volume']
              (dates parsed as datetime)
            - df_rec: DataFrame of original records with numeric financial columns normalized
            - df_institutional: DataFrame for institutional entries (if any)
            - df_margin: DataFrame for margin entries (if any)
            - df_revenue: DataFrame for revenue entries (if any)
            - monthly_daily_revenue_mean: Series indexed by Period (monthly avg daily_revenue)
            - eps_stats: dict of basic EPS statistics (count, mean, median, etc.)
            - gross_profit_stats: dict of basic gross profit statistics
            - correlation: correlation matrix DataFrame among available numeric columns

    Raises:
        FileNotFoundError: If raw is a string path and the file doesn't exist

    Example:
        Basic usage with a JSON file path:

        >>> results = prepare_analysis_data('data/preprocessed_1727.json')
        >>> print(f"Records: {len(results['df_rec'])}")
        >>> print(f"EPS mean: {results['eps_stats']['mean']}")

        Using with a list of records:

        >>> records = [{'date': '2023-01-01', 'close': 100.0, 'eps': 1.5}]
        >>> results = prepare_analysis_data(records)
        >>> print(results['df_price'].head())

    Note:
        The function normalizes numeric columns.
        Correlation matrix is computed only for available numeric columns.
    """
    # Load JSON if `raw` is a path
    records: List[dict]
    if isinstance(raw, str):
        if not os.path.exists(raw):
            raise FileNotFoundError(f"JSON file not found: {raw}")
        raw_loaded = load_json_data(raw)
        # support wrapper shape {"data": [...]}
        if isinstance(raw_loaded, dict) and isinstance(raw_loaded.get('data'), list):
            records = raw_loaded['data']
        elif isinstance(raw_loaded, list):
            records = raw_loaded
        else:
            records = []
    elif isinstance(raw, dict) and isinstance(raw.get('data'), list):
        records = raw.get('data')
    elif isinstance(raw, list):
        records = raw
    else:
        records = []

    # Use prepare_data_for_chart to defensively extract parts
    price_data, institutional_data, margin_data, revenue_data = prepare_data_for_chart(records)

    df_price = pd.DataFrame(price_data)
    if not df_price.empty:
        df_price['date'] = pd.to_datetime(df_price['date'])
        # ensure numeric types
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in df_price.columns:
                df_price[col] = _safe_to_numeric(df_price[col])
        df_price = df_price.sort_values('date')

    df_rec = pd.DataFrame(records)
    if not df_rec.empty and 'date' in df_rec.columns:
        df_rec['date'] = pd.to_datetime(df_rec['date'], errors='coerce')
        df_rec = df_rec.sort_values('date')

    df_institutional = pd.DataFrame(institutional_data)
    df_margin = pd.DataFrame(margin_data)
    df_revenue = pd.DataFrame(revenue_data)

    # Normalize common numeric columns in df_rec
    numeric_cols = ['close', 'eps', 'gross_profit', 'daily_revenue', 'MarginPurchaseBalanceChange', 'ShortSaleBalanceChange']
    for col in numeric_cols:
        if col in df_rec.columns:
            df_rec[col] = _safe_to_numeric(df_rec[col])

    # Monthly average daily_revenue (group by YYYY-MM)
    monthly_daily_revenue_mean = pd.Series(dtype=float)
    if 'daily_revenue' in df_rec.columns and not df_rec['daily_revenue'].dropna().empty:
        df_rev = df_rec[['date', 'daily_revenue']].copy()
        df_rev = df_rev.dropna(subset=['date'])
        df_rev['month'] = df_rev['date'].dt.to_period('M')
        monthly_daily_daily = df_rev.groupby('month')['daily_revenue'].mean()
        monthly_daily_revenue_mean = monthly_daily_daily

    # Basic stats
    eps_stats = compute_basic_stats(df_rec['eps']) if 'eps' in df_rec.columns else compute_basic_stats(pd.Series(dtype=float))
    gross_profit_stats = compute_basic_stats(df_rec['gross_profit']) if 'gross_profit' in df_rec.columns else compute_basic_stats(pd.Series(dtype=float))

    # Correlation among numeric columns of interest
    correlation = pd.DataFrame()
    corr_cols = [c for c in ('close', 'volume', 'foreign_investor_net', 'investment_trust_net', 'dealer_net', 'MarginPurchaseBalanceChange', 'ShortSaleBalanceChange', 'eps', 'gross_profit') if c in df_rec.columns]
    if corr_cols:
        corr_df = df_rec[corr_cols].apply(pd.to_numeric, errors='coerce').dropna()
        if not corr_df.empty:
            correlation = corr_df.corr()

    return {
        'df_price': df_price,
        'df_rec': df_rec,
        'df_institutional': df_institutional,
        'df_margin': df_margin,
        'df_revenue': df_revenue,
        'monthly_daily_revenue_mean': monthly_daily_revenue_mean,
        'eps_stats': eps_stats,
        'gross_profit_stats': gross_profit_stats,
        'correlation': correlation,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataframes and stats from a preprocessed JSON file (no charts).')
    parser.add_argument('json_path', nargs='?', default=os.path.join(os.path.dirname(__file__), 'preprocessed_1727.json'), help='Path to preprocessed JSON file')
    parser.add_argument('--start-date', dest='start_date', help='Filter start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', dest='end_date', help='Filter end date (YYYY-MM-DD)', default=None)
    args = parser.parse_args()

    try:
        results = prepare_analysis_data(args.json_path)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    df_rec = results['df_rec']
    df_price = results['df_price']

    # If user provided date filters, apply them (inclusive)
    if args.start_date or args.end_date:
        from datetime import datetime
        try:
            s = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
            e = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None
        except ValueError:
            print('Invalid date format for --start-date or --end-date. Use YYYY-MM-DD.')
            raise SystemExit(1)

        # Default missing endpoints to available data bounds
        if s and not e:
            e = df_rec['date'].max().date() if not df_rec.empty else s
        if e and not s:
            s = df_rec['date'].min().date() if not df_rec.empty else e

        # Filter dataframes
        if not df_rec.empty and 'date' in df_rec.columns:
            df_rec = df_rec[(df_rec['date'].dt.date >= s) & (df_rec['date'].dt.date <= e)]
        if not df_price.empty and 'date' in df_price.columns:
            df_price = df_price[(df_price['date'].dt.date >= s) & (df_price['date'].dt.date <= e)]

        # Recompute basic derived stats from filtered data
        monthly_daily_revenue_mean = pd.Series(dtype=float)
        if 'daily_revenue' in df_rec.columns and not df_rec['daily_revenue'].dropna().empty:
            df_rev = df_rec[['date', 'daily_revenue']].dropna(subset=['date']).copy()
            df_rev['month'] = df_rev['date'].dt.to_period('M')
            monthly_daily_revenue_mean = df_rev.groupby('month')['daily_revenue'].mean()

        eps_stats = compute_basic_stats(df_rec['eps']) if 'eps' in df_rec.columns else compute_basic_stats(pd.Series(dtype=float))
        gross_profit_stats = compute_basic_stats(df_rec['gross_profit']) if 'gross_profit' in df_rec.columns else compute_basic_stats(pd.Series(dtype=float))

        corr_cols = [c for c in ('close', 'volume', 'foreign_investor_net', 'investment_trust_net', 'dealer_net', 'MarginPurchaseBalanceChange', 'ShortSaleBalanceChange', 'eps', 'gross_profit') if c in df_rec.columns]
        correlation = pd.DataFrame()
        if corr_cols:
            corr_df = df_rec[corr_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if not corr_df.empty:
                correlation = corr_df.corr()

        # Update results dict for printing
        results['monthly_daily_revenue_mean'] = monthly_daily_revenue_mean
        results['eps_stats'] = eps_stats
        results['gross_profit_stats'] = gross_profit_stats
        results['correlation'] = correlation

    stock_id = df_rec['stock_id'].iloc[0] if not df_rec.empty and 'stock_id' in df_rec.columns else ''
    start_date = df_rec['date'].min().date() if not df_rec.empty and 'date' in df_rec.columns else None
    end_date = df_rec['date'].max().date() if not df_rec.empty and 'date' in df_rec.columns else None

    print('\n'+"="*40)
    print('Data Analysis Preparation Results')
    print(f"Prepared data for stock: {stock_id}")
    print(f"Filtered date range: {start_date} -> {end_date}")
    print(f"Records: {len(df_rec)}, Price rows: {len(df_price)}")
    
    print('\nEPS stats:')
    print(results['eps_stats'])

    print('\nGross Profit stats:')
    print(results['gross_profit_stats'])

    if not results['correlation'].empty:
        print('\nCorrelation matrix:')
        print(results['correlation'])


    print(f"\n\nFiltered date range: {start_date} -> {end_date}")