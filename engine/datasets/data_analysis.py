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

# Prefer package-qualified imports; fall back to adding project root to sys.path for direct execution
try:
    from engine.datasets import ml_model, dl_model
except Exception:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from engine.datasets import ml_model, dl_model

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

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators for stock data and prepare for ML models.

    Args:
        df (pd.DataFrame): Dataframe containing at least 'close' column.

    Returns:
        pd.DataFrame: Dataframe with technical indicators and NaNs removed.
    """
    df = df.copy()
    
    # Simple Moving Averages (use min_periods=1 to avoid trimming the start of the dataset)
    df['SMA_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Daily Return (fillna(0) to recover the very first row)
    df['Daily_Return'] = df['close'].pct_change().fillna(0)
    
    # Volatility (Standard Deviation of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20, min_periods=1).std().fillna(0)

    # Lagged features: Tree-based models (RF/GB) benefit from seeing previous time steps
    # as they cannot inherently understand time-series order like linear models might.
    for lag in range(1, 11):
        df[f'Close_Lag{lag}'] = df['close'].shift(lag).ffill().bfill()
        df[f'Return_Lag{lag}'] = df['Daily_Return'].shift(lag).fillna(0)

    # Rolling statistics for multiple windows
    for w in (5, 10, 20, 60):
        df[f'roll_mean_{w}'] = df['close'].rolling(window=w, min_periods=1).mean()
        df[f'roll_std_{w}'] = df['close'].rolling(window=w, min_periods=1).std().fillna(0)

    # Ratios and momentum features
    if 'SMA_20' in df.columns:
        df['close_over_SMA20'] = df['close'] / df['SMA_20']
        df['momentum_SMA20'] = df['close'] - df['SMA_20']

    # Imputation Strategy to minimize "Data Trimming" (head/tail row loss):
    # 1. Financial metrics (EPS, Revenue) are periodic. 
    # Forward-fill carries the last report forward.
    # Back-fill is used here ONLY to recover rows at the start before the first report.
    periodic_cols = ['eps', 'gross_profit', 'revenue', 'daily_revenue', 'revenue_year', 'revenue_month']
    for c in periodic_cols:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    # 2. Institutional/Margin changes are often NaN at the start of a period. Fill with 0.
    change_cols = [
        'MarginPurchaseBalanceChange', 'ShortSaleBalanceChange',
        'foreign_investor_net', 'investment_trust_net', 'dealer_net'
    ]
    for c in change_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 3. Handle any remaining intermittent NaNs in the price data (rare)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    # Remove rows with NaN values created by rolling windows (e.g., SMA_60) and shifts.
    # RF/GB models in scikit-learn cannot handle NaNs.
    df.dropna(inplace=True)
    
    return df

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
    day_shift: int = -1,
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
        day_shift: Integer, number of days to shift for target_close (default: -1, i.e., next day's close).

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

        >>> results = prepare_analysis_data('data/preprocessed_1727.json', day_shift=-1)
        >>> print(f"Records: {len(results['df_rec'])}")
        >>> print(f"EPS mean: {results['eps_stats']['mean']}")

        Using with a list of records:

        >>> records = [{'date': '2023-01-01', 'close': 100.0, 'eps': 1.5}]
        >>> results = prepare_analysis_data(records, day_shift=-1)
        >>> print(results['df_price'].head())

        Filtering by date range:

        >>> results = prepare_analysis_data('data.json', start_date='2023-01-01', end_date='2023-12-31', day_shift=-1)
        >>> print(f"Filtered records: {len(results['df_rec'])}")

    Note:
        The function normalizes numeric columns.
        Correlation matrix is computed only for available numeric columns.
        If start_date or end_date are provided, all computations are based on filtered data.
        The day_shift parameter controls the target_close shift (e.g., -1 for next day, -2 for two days ahead).
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
    numeric_cols = ['close', 'open', 'high', 'low', 'volume', 'eps', 'gross_profit', 'daily_revenue', 'MarginPurchaseBalanceChange', 'ShortSaleBalanceChange']
    for col in numeric_cols:
        if col in df_rec.columns:
            df_rec[col] = _safe_to_numeric(df_rec[col])

    # Calculate Technical Indicators
    if not df_rec.empty and 'close' in df_rec.columns:
        df_rec = calculate_technical_indicators(df_rec)
        # Create Target: Shifted Close Price (controlled by day_shift)
        df_rec['target_close'] = df_rec['close'].shift(day_shift)
        # Drop rows with NaN (from indicators and shifted target)
        df_rec = df_rec.dropna(subset=['SMA_20', 'RSI', 'target_close'])

        # Diagnostic: report which columns caused trimming of head/tail
        def _diagnose_date_trimming(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
            orig_start = original_df['date'].min()
            orig_end = original_df['date'].max()
            proc_start = processed_df['date'].min()
            proc_end = processed_df['date'].max()

            # Find first valid index where all required columns are non-NaN
            required = [c for c in processed_df.columns if c not in ('date',)]
            first_valid_idx = None
            last_valid_idx = None
            for idx, row in processed_df.iterrows():
                if not row[required].isna().any():
                    first_valid_idx = row['date']
                    break
            for idx in range(len(processed_df)-1, -1, -1):
                row = processed_df.iloc[idx]
                if not row[required].isna().any():
                    last_valid_idx = row['date']
                    break

            # Check which columns have NaNs at the top or bottom of original df
            top_trim_cols = []
            bottom_trim_cols = []
            for col in required:
                series = original_df[col] if col in original_df.columns else processed_df[col]
                # top NaN stretch length
                top_na = series.isna().cummax().sum() if hasattr(series, 'isna') else 0
                if pd.isna(series.iloc[0]) or series.isna().iloc[:10].any():
                    top_trim_cols.append(col)
                if pd.isna(series.iloc[-1]) or series.isna().iloc[-10:].any():
                    bottom_trim_cols.append(col)

            print('\nDiagnostic: Data trimming summary:')
            print(f"original range: {orig_start} -> {orig_end}")
            print(f"processed range: {proc_start} -> {proc_end}")
            print(f"first fully-valid row in processed data: {first_valid_idx}")
            print(f"last fully-valid row in processed data: {last_valid_idx}")
            print(f"Columns with NaNs near start (first 10 rows): {top_trim_cols}")
            print(f"Columns with NaNs near end (last 10 rows): {bottom_trim_cols}\n")

        # Call the diagnostic using copies so it checks the preprocessed (before dropna) and postprocessed frames
        try:
            small_orig = pd.DataFrame(records)
            small_orig['date'] = pd.to_datetime(small_orig['date'], errors='coerce')
            _diagnose_date_trimming(small_orig, df_rec)
        except Exception as e:
            print(f"Date trimming diagnostic failed: {e}")

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
    corr_df = pd.DataFrame()  # Initialize to avoid unbound error
    corr_cols = [c for c in (
        'close', 'volume', 'daily_revenue', 'foreign_investor_net', 
        'investment_trust_net', 'dealer_net', 'MarginPurchaseBalanceChange', 
        'ShortSaleBalanceChange', 'eps', 'gross_profit',
        'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Daily_Return', 'Volatility', 'target_close'
    ) if c in df_rec.columns]
    if corr_cols:
        # Preserve date index for time-series plotting when 'date' exists.
        if 'date' in df_rec.columns:
            # Ensure the date column is datetime and set as index for corr_df
            corr_df = df_rec.set_index('date')[corr_cols].apply(pd.to_numeric, errors='coerce')
            # Convert index to DatetimeIndex if not already
            if not isinstance(corr_df.index, pd.DatetimeIndex):
                corr_df.index = pd.to_datetime(corr_df.index, errors='coerce')
        else:
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

def data_analyzer(metric_df: pd.DataFrame, run_trees: bool = False) -> None:
    """
    Analyze the correlation DataFrame and print insights.

    Args:
        metric_df: DataFrame containing numeric columns for correlation analysis.
        run_trees: If True, execute time-series tree training and plot predictions with date x-axis.
    """
    algorithms = {"regression": True, 
                  "trees": run_trees, 
                  "classification": False,
                  "deep_learning": True} # configurable options for analysis types

    # Import the time-series tree training module
    try:
        from engine.datasets.train_trees import train_tree_models
    except Exception:
        from train_trees import train_tree_models  # fallback for local runs


    print(f"Shape of input Metric DataFrame: {metric_df.shape}")
    print(f"Type of input Metric DataFrame: {type(metric_df)}")
    if metric_df.empty:
        print("Metric DataFrame is empty. No analysis performed.")
        return
    # Remove rows with NaN values
    metric_df = metric_df.dropna()

    # Choose target series (keep pandas Series to preserve index)
    if 'target_close' in metric_df.columns:
        y_series = metric_df['target_close']  # Target variable: Tomorrow's Close
        target_col = 'target_close'
    else:
        y_series = metric_df['close']  # Fallback
        target_col = 'close'

    # Determine x-axis values: prefer DatetimeIndex, otherwise look for a 'date' column, otherwise fallback to integer index
    if isinstance(y_series.index, pd.DatetimeIndex):
        x_values = y_series.index
        xlabel = 'Date'
    elif 'date' in metric_df.columns:
        x_values = pd.to_datetime(metric_df['date'])
        xlabel = 'Date'
    else:
        x_values = range(len(y_series))
        xlabel = 'Sample Index'

    plt.figure(figsize=(10, 8))
    plt.plot(x_values, y_series.values, label=f'Target Close Prices ({ "Tomorrow" if target_col=="target_close" else "Close"})', color='blue')
    plt.title('Target Close Prices Over Time')
    plt.xlabel(xlabel)
    plt.ylabel('Close Price')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.hist(y_series.values, bins=30, color='green', alpha=0.7)
    plt.title('Distribution of Target Close Prices')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Example analysis: Print correlation of each feature with the target variable  
    sel_feature = []
    for i, col in enumerate(metric_df.columns):
        if col == target_col or col == 'close': # Don't use future close or current close as feature directly if we want a real prediction
            continue
        feature = metric_df[col].to_numpy()
        if feature.size != y_series.size:
            print(f"Skipping correlation for {col} due to size mismatch.")
            continue
        corr = np.corrcoef(feature, y_series.values)[0, 1]
        print(f"Correlation between '{target_col}' and '{col}': {corr:.4f}")
        if abs(corr) > 0.05:  # Lowered threshold for feature selection
            sel_feature.append(col)
    
    X = metric_df[sel_feature].to_numpy() if sel_feature else metric_df.drop(columns=[target_col]).to_numpy()
    
    print(f"Converted Metric DataFrame to numpy arrays: X shape {X.shape}, y shape {y_series.size}")
    print(f"Selected features for model fitting: {sel_feature}")

    samples = X.shape[0]
    train_ratio = 0.9
    split_idx = round(samples * train_ratio)
    X_train = X[:split_idx]
    y_train = y_series.values[:split_idx]   
    X_test = X[split_idx:]
    y_test = y_series.values[split_idx:]

    # Prepare x-axis indices for plots: prefer datetime index when available
    if isinstance(x_values, pd.DatetimeIndex):
        x_values_arr = x_values
    else:
        x_values_arr = list(x_values)
    x_train_idx = x_values_arr[:split_idx]
    x_test_idx = x_values_arr[split_idx:]

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
        # Capture regression results so we can run the ensemble backtest on the test set
        results_reg = ml_model.regression_models(
            X_train, y_train, X_test, y_test,
            show_plots=True, x_train_idx=x_train_idx, x_test_idx=x_test_idx
        )

        # If regression returned predictions, run example ensemble backtest (price-based)
        try:
            preds = {k: results_reg[f"y_pred_{k}"] for k in ("lr", "svr") if f"y_pred_{k}" in results_reg}
            if preds:
                # current_prices: current close aligned with preds (same slice used for y_test which is target_close)
                current_prices = metric_df['close'].values[split_idx:]
                # y_test are target next-day closes; convert to next-period returns
                y_returns = (y_test - current_prices) / current_prices
                res = ml_model.example_run_ensemble_from_preds(
                    preds, y_returns,
                    score_type='price',
                    current_prices=current_prices,
                    threshold=0.002,            # 0.2% threshold
                    min_models_agree=1,
                    transaction_cost=0.0005,
                    show_plots=True,
                    x_idx=x_test_idx
                )
                print("Ensemble metrics:", res.get('metrics'))
        except Exception as e:
            print(f"Ensemble demo skipped (error): {e}")

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
        ml_model.classification_models(
            X_train,
            y_train_cluster,
            X_test,
            y_test_cluster,
            y_representative_target,
            y_test,
            show_plots=True,
            x_train_idx=x_train_idx,
            x_test_idx=x_test_idx,
        )

    if algorithms.get("deep_learning"):
        print("\nStarting Deep Learning Model Fitting...")
        # Capture DL results and run ensemble demo (handles LSTM sequence offset)
        dl_results = dl_model.models(
            X_train, y_train, X_test, y_test,
            x_train_idx=x_train_idx, x_test_idx=x_test_idx
        )
        try:
            preds_dl = {}
            if 'predictions' in dl_results:
                preds_dl['dl'] = dl_results['predictions']
            if preds_dl:
                preds_arr = preds_dl['dl']
                m = len(preds_arr)
                test_closes = metric_df['close'].values[split_idx:]
                # If predictions are shorter, compute sequence offset
                if m <= len(test_closes):
                    seq_len = len(test_closes) - m
                    # current_prices aligned to the prediction (current price is the day before the predicted next-day close)
                    start_idx = split_idx + seq_len - 1 if seq_len > 0 else split_idx
                    current_prices = metric_df['close'].values[start_idx : start_idx + m]
                    # y_test aligned with predictions
                    y_true_actual = y_test[seq_len:] if seq_len > 0 else y_test[:m]
                    # compute returns (next-day return aligned with predictions)
                    y_returns_dl = (y_true_actual - current_prices) / current_prices
                    # align x_idx for plotting if available
                    x_idx_dl = x_test_idx[seq_len:] if (seq_len > 0 and x_test_idx is not None) else x_test_idx
                    res_dl = ml_model.example_run_ensemble_from_preds(
                        preds_dl, y_returns_dl,
                        score_type='price',
                        current_prices=current_prices,
                        threshold=0.002,
                        min_models_agree=1,
                        transaction_cost=0.0005,
                        show_plots=True,
                        x_idx=x_idx_dl
                    )
                    print("DL Ensemble metrics:", res_dl.get('metrics'))
        except Exception as e:
            print(f"DL Ensemble demo skipped (error): {e}")
    
    if algorithms.get("trees"):
        tasks: List[Literal['regression', 'classification']] = ['regression', 'classification']
        for task in tasks:
            print(f"\nStarting Time-Series Tree Model Fitting ({task.upper()})...")
            try:
                results_trees = train_tree_models(metric_df, target_col='Daily_Return', task=task, n_splits=5, n_iter=20, test_ratio=0.1)
                
                if task == 'regression':
                    print(f"RF Test R²: {results_trees.get('rf_test_r2', 0):.4f}")
                    print(f"GB Test R²: {results_trees.get('gb_test_r2', 0):.4f}")
                    # Plot predictions vs actual using date index when available
                    if 'test_index' in results_trees and 'y_test' in results_trees:
                        try:
                            ti = results_trees['test_index']
                            y_test_vals = results_trees['y_test']
                            y_pred_rf = results_trees['y_pred_rf']
                            y_pred_gb = results_trees['y_pred_gb']
                            plt.figure(figsize=(12, 6))
                            plt.plot(ti, y_test_vals, '.-', label='Actual', color='black', alpha=0.8)
                            plt.plot(ti, y_pred_rf, label='RF Predicted', color='tab:blue', alpha=0.8)
                            plt.plot(ti, y_pred_gb, label='GB Predicted', color='tab:orange', alpha=0.8)
                            plt.title(f'Tree Model Predictions vs Actual ({task.upper()})')
                            plt.xlabel('Date')
                            plt.ylabel('Target')
                            plt.legend()
                            plt.gcf().autofmt_xdate()
                            plt.grid(True, alpha=0.3)
                            plt.show()
                        except Exception as e:
                            print(f"Could not plot tree predictions: {e}")
                else:
                    print(f"RF Test Acc: {results_trees.get('rf_test_acc', 0):.4f}")
                    print(f"GB Test Acc: {results_trees.get('gb_test_acc', 0):.4f}")
                    if 'rf_report' in results_trees:
                        print("RF Classification Report (Directional):")
                        print(results_trees['rf_report'])
                    # Plot classification predictions over time
                    if 'test_index' in results_trees and 'y_test' in results_trees:
                        try:
                            ti = results_trees['test_index']
                            y_test_vals = results_trees['y_test']
                            y_pred_rf = results_trees['y_pred_rf']
                            y_pred_gb = results_trees['y_pred_gb']
                            plt.figure(figsize=(12, 4))
                            plt.scatter(ti, y_test_vals, label='Actual', c='black', s=10)
                            plt.scatter(ti, y_pred_rf, label='RF Pred', c='tab:blue', s=6, alpha=0.7)
                            plt.scatter(ti, y_pred_gb, label='GB Pred', c='tab:orange', s=6, alpha=0.7)
                            plt.title(f'Tree Model Classification Predictions ({task.upper()})')
                            plt.xlabel('Date')
                            plt.ylabel('Class Label')
                            plt.legend()
                            plt.gcf().autofmt_xdate()
                            plt.grid(True, alpha=0.3)
                            plt.show()
                        except Exception as e:
                            print(f"Could not plot classification predictions: {e}")

                # Display top features for the first model (RF)
                if 'rf_importances' in results_trees:
                    top_features = sorted(results_trees['rf_importances'].items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"Top 5 Features ({task}): {top_features}")

            except Exception as e:
                print(f"Error running tree training pipeline for {task}: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataframes and stats from a preprocessed JSON file (no charts).')
    parser.add_argument('json_path', nargs='?', default=os.path.join(os.path.dirname(__file__), 'preprocessed_1727.json'), help='Path to preprocessed JSON file')
    parser.add_argument('--start-date', dest='start_date', help='Filter start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', dest='end_date', help='Filter end date (YYYY-MM-DD)', default=None)
    parser.add_argument('--day-shift', dest='day_shift', type=int, default=-1, help='Number of days to shift for target_close (default: -1, next day)')
    args = parser.parse_args()

    try:
        results = prepare_analysis_data(
            args.json_path,
            start_date=args.start_date,
            end_date=args.end_date,
            day_shift=args.day_shift
        )
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
    data_analyzer(results['df_metric'], run_trees=False)

    print(f"\n\nFiltered date range: {start_date} -> {end_date}")