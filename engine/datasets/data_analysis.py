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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ml_model
import dl_model

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
    start_date: Optional[Union[str, datetime.date]] = None,
    end_date: Optional[Union[str, datetime.date]] = None,
) -> Dict[str, Any]:
    """
    Prepare dataframes and summary values from preprocessed FinMind-like data.

    This function loads and processes financial data from various sources,
    creating structured DataFrames for analysis and computing basic statistics.
    It provides correlation analysis. Optionally filters data by date range.

    Args:
        raw: Either a path to a JSON file (str), a list of records, or a dict
             (as loaded from JSON). Supports wrapper format {"data": [...]}.
        start_date: Optional start date for filtering (inclusive). Accepts str 'YYYY-MM-DD' or date object.
        end_date: Optional end date for filtering (inclusive). Accepts str 'YYYY-MM-DD' or date object.

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
        ValueError: If start_date or end_date are invalid date strings

    Example:
        Basic usage with a JSON file path:

        >>> results = prepare_analysis_data('data/preprocessed_1727.json')
        >>> print(f"Records: {len(results['df_rec'])}")
        >>> print(f"EPS mean: {results['eps_stats']['mean']}")

        Using with a list of records:

        >>> records = [{'date': '2023-01-01', 'close': 100.0, 'eps': 1.5}]
        >>> results = prepare_analysis_data(records)
        >>> print(results['df_price'].head())

        Filtering by date range:

        >>> results = prepare_analysis_data('data.json', start_date='2023-01-01', end_date='2023-12-31')
        >>> print(f"Filtered records: {len(results['df_rec'])}")

    Note:
        The function normalizes numeric columns.
        Correlation matrix is computed only for available numeric columns.
        If start_date or end_date are provided, all computations are based on filtered data.
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

    # Apply date filtering if provided
    if start_date or end_date:
        s = None
        e = None
        if start_date:
            if isinstance(start_date, str):
                try:
                    s = datetime.strptime(start_date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError(f"Invalid start_date format: {start_date}. Use YYYY-MM-DD.")
            elif isinstance(start_date, datetime.date):
                s = start_date
            else:
                raise ValueError("start_date must be str or date object")
        if end_date:
            if isinstance(end_date, str):
                try:
                    e = datetime.strptime(end_date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Use YYYY-MM-DD.")
            elif isinstance(end_date, datetime.date):
                e = end_date
            else:
                raise ValueError("end_date must be str or date object")

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
    corr_cols = [c for c in ('close', 'volume', 'daily_revenue', 'foreign_investor_net', 'investment_trust_net', 'dealer_net', 'MarginPurchaseBalanceChange', 'ShortSaleBalanceChange', 'eps', 'gross_profit') if c in df_rec.columns]
    if corr_cols:
        corr_df = df_rec[corr_cols].apply(pd.to_numeric, errors='coerce')
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
        'df_metric': corr_df,
    }

def data_analyzer(metric_df: pd.DataFrame) -> None:
    """
    Analyze the correlation DataFrame and print insights.

    Args:
        metric_df: DataFrame containing numeric columns for correlation analysis.
    """
    algorithms = {"regression": True, "classification": False, "deep_learning": True} # configurable options for analysis types

    print(f"Shape of input Metric DataFrame: {metric_df.shape}")
    print(f"Type of input Metric DataFrame: {type(metric_df)}")
    if metric_df.empty:
        print("Metric DataFrame is empty. No analysis performed.")
        return
    # Remove rows with NaN values
    metric_df = metric_df.dropna() 

    y = metric_df['close'].to_numpy() # Target variable
    plt.figure(figsize=(10, 8))
    plt.plot(y, label='Close Prices', color='blue')
    plt.title('Close Prices Over Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.hist(y, bins=30, color='green', alpha=0.7)
    plt.title('Distribution of Close Prices')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Example analysis: Print correlation of each feature with the target variable  
    sel_feature = None
    for i, col in enumerate(metric_df.columns):
        if col == 'close':
            continue
        feature = metric_df[col].to_numpy()
        if feature.size != y.size:
            print(f"Skipping correlation for {col} due to size mismatch.")
            continue
        corr = np.corrcoef(feature, y)[0, 1]
        print(f"Correlation between 'close' and '{col}': {corr:.4f}")
        if abs(corr) > 0.1:  # Threshold for feature selection
            # Use append in-place if list exists, otherwise create a new list
            if sel_feature:
                sel_feature.append(col)
            else:
                sel_feature = [col]
    
    X = metric_df[sel_feature].to_numpy() if sel_feature else metric_df.drop(columns=['close']).to_numpy()
    
    print(f"Converted Metric DataFrame to numpy arrays: X shape {X.shape}, y shape {y.shape}")
    print(f"Selected features for model fitting: {sel_feature}")

    samples = X.shape[0]
    train_ratio = 0.9
    X_train = X[:round(samples*train_ratio)]
    y_train = y[:round(samples*train_ratio)]   
    X_test = X[round(samples*train_ratio):]
    y_test = y[round(samples*train_ratio):]
    print(f"Split data into training and testing sets: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 0], y, alpha=0.5)
    # plt.title(f'Scatter Plot of {sel_feature[0] if sel_feature else "Feature 0"} vs Close Price')
    # plt.xlabel(sel_feature[0] if sel_feature else "Feature 0")
    # plt.ylabel('Close Price')
    # plt.grid(True, alpha=0.3)
    # plt.show()

    if algorithms.get("regression"):
        print("\nStarting Regression Model Fitting...")
        ml_model.regression_models(X_train, y_train, X_test, y_test)

    if algorithms.get("classification"):
        print("\nStarting Classification Model Fitting...")

        # Semi-supervised Learning: Generate cluster labels to use as targets
        # This captures hidden structures (e.g., market regimes) to assist classification
        results_cluster = ml_model.clustering_model(X_train, y_train, X_test, y_test, n_clusters=30, show_plots=False)
        y_train_cluster = results_cluster["train_clusters"]
        y_test_cluster = results_cluster["test_clusters"] 
        y_representative_target = results_cluster["representative_targets"]

        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_train_cluster, c=y_train_cluster, cmap='viridis', alpha=0.5)
        plt.title('Cluster Assignments vs Close Price (Training Set)')
        plt.xlabel('Close Price')
        plt.ylabel('Cluster Label')
        plt.colorbar(label='Cluster Label')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Classification model fitting
        ml_model.classification_models(X_train, y_train_cluster, X_test, y_test_cluster, y_representative_target, y_test, show_plots=True)

    if algorithms.get("deep_learning"):
        print("\nStarting Deep Learning Model Fitting...")
        dl_model.models(X_train, y_train, X_test, y_test)
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataframes and stats from a preprocessed JSON file (no charts).')
    parser.add_argument('json_path', nargs='?', default=os.path.join(os.path.dirname(__file__), 'preprocessed_1727.json'), help='Path to preprocessed JSON file')
    parser.add_argument('--start-date', dest='start_date', help='Filter start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', dest='end_date', help='Filter end date (YYYY-MM-DD)', default=None)
    args = parser.parse_args()

    try:
        results = prepare_analysis_data(args.json_path, start_date=args.start_date, end_date=args.end_date)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)
    except ValueError as e:
        print(e)
        raise SystemExit(1)

    df_rec = results['df_rec']
    df_price = results['df_price']

    stock_id = df_rec['stock_id'].iloc[0] if not df_rec.empty and 'stock_id' in df_rec.columns else ''
    start_date = df_rec['date'].min().date() if not df_rec.empty and 'date' in df_rec.columns else None
    end_date = df_rec['date'].max().date() if not df_rec.empty and 'date' in df_rec.columns else None

    print('\n'+"="*35)
    print('Data Analysis Preparation Results')
    print(f"Prepared data for stock: {stock_id}")
    print(f"Filtered date range: {start_date} -> {end_date}")
    print(f"Records: {len(df_rec)}, Price rows: {len(df_price)}")
    
    print('\nEPS stats:')
    print(results['eps_stats'])

    print('\nGross Profit stats:')
    print(results['gross_profit_stats'])

    # # Correlation matrix
    # if not results['correlation'].empty:
    #     print('\nCorrelation matrix:')
    #     print(results['correlation'])

    print('\n'+'='*35)
    print("Start Data Analysis")
    data_analyzer(results['df_metric'])

    print(f"\n\nFiltered date range: {start_date} -> {end_date}")