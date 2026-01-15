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

import os

from sklearn.model_selection import TimeSeriesSplit
os.environ["KERAS_BACKEND"] = "torch"
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union, Tuple

# Prefer package-qualified imports; fall back to local imports for direct execution
try:
    from engine.datasets import ml_model, dl_model
except (ImportError, ModuleNotFoundError):
    try:
        import ml_model
        import dl_model
    except ImportError:
        # Fallback to adding project root to sys.path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from engine.datasets import ml_model, dl_model

# Try importing parsing helpers from data_visual.py with a resilient fallback
try:
    from engine.datasets.data_visual import load_json_data, prepare_data_for_chart
except (ImportError, ModuleNotFoundError):
    try:
        from data_visual import load_json_data, prepare_data_for_chart
    except ImportError:
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
            
    # Metadata and categorical info (like country) should also be forward filled
    for c in ['country', 'stock_id']:
        if c in df.columns:
            df[c] = df[c].ffill().bfill()

    # Drop only if critical features are still missing after imputation.
    # We want to keep recent data points even if some non-critical columns are delayed.
    # The downstream ML pipeline in unified_pipeline.py will perform final cleaning.
    critical_cols = ['close', 'volume', 'SMA_20', 'RSI']
    df.dropna(subset=[c for c in critical_cols if c in df.columns], inplace=True)
    
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
    data_path: str, 
    start_date: Optional[Union[str, date, datetime]] = None, 
    end_date: Optional[Union[str, date, datetime]] = None, 
    day_shift: int = -1,
    pred_date: Optional[Union[str, date, datetime]] = None
) -> Dict[str, Any]:
    """
    Prepare and clean stock data for analysis and model training.

    Args:
        data_path: Path to the raw JSON data file.
        start_date: Optional filter for the start date (str, date, or datetime).
        end_date: Optional filter for the end date (str, date, or datetime).
        day_shift: Lag/shift applied to the target close price.
        pred_date: Optional date to start prediction (for filtering or metadata).

    Returns:
        A dictionary containing processed DataFrames and generated metrics.
    """
    # Load JSON if `raw` is a path
    records: Union[List[dict], Dict[str, Any]] = []
    if isinstance(data_path, str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"JSON file not found: {data_path}")
        raw_loaded = load_json_data(data_path)
        
        # Support various JSON structures:
        # 1. {"data": [...]}
        # 2. {"stock_id": {"data": [...]}} or {"stock_id": [...]}
        # 3. List of records directly [...]
        # 4. Dict of lists {"price": [...], "revenue": [...]}
        if isinstance(raw_loaded, dict):
            # Check if it's keyed by stock ID (especially for crawler outputs or combined files)
            first_key = next(iter(raw_loaded.keys())) if raw_loaded else None
            # Heuristic for stock ID key: numeric or has a dash (e.g. 2330 or "京元電子-2449")
            if first_key and (first_key.isdigit() or '-' in first_key):
                inner_data = raw_loaded[first_key]
                if isinstance(inner_data, dict):
                    if 'data' in inner_data and isinstance(inner_data['data'], list):
                        records = inner_data['data']
                    else:
                        # Could be the dict of lists (price, revenue, etc.)
                        records = inner_data
                elif isinstance(inner_data, list):
                    records = inner_data
            elif isinstance(raw_loaded.get('data'), list):
                records = raw_loaded['data']
            else:
                records = raw_loaded
        elif isinstance(raw_loaded, list):
            records = raw_loaded
    elif isinstance(data_path, dict):
        records = data_path.get('data') if isinstance(data_path.get('data'), list) else data_path
    elif isinstance(data_path, list):
        records = data_path

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
        
        # Drop rows with NaN from critical technical indicators (head trimming)
        # We keep the rows where only 'target_close' is NaN (usually the last row)
        # to ensure the most recent data is available for prediction or inference.
        indicator_cols = ['SMA_20', 'RSI']
        available_indicators = [c for c in indicator_cols if c in df_rec.columns]
        df_rec = df_rec.dropna(subset=available_indicators)

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
            elif isinstance(start_date, date):
                s = start_date
            else:
                raise ValueError("start_date must be str or date object")
        if end_date:
            if isinstance(end_date, str):
                try:
                    e = datetime.strptime(end_date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError(f"Invalid end_date format: {end_date}. Use YYYY-MM-DD.")
            elif isinstance(end_date, date):
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
        'pred_date': pred_date
    }

def data_analyzer(
    df: pd.DataFrame,
    run_trees: bool = False,
    show_plots: bool = True,
    pred_date: Optional[Union[str, date, datetime]] = None
) -> Dict[str, Any]:
    """
    Perform statistical analysis and feature correlation checks on preprocessed stock data.

    Args:
        df: DataFrame containing numeric columns for correlation analysis.
        run_trees: If True, execute time-series tree training and plot predictions with date x-axis.
        show_plots: Whether to display charts.
        pred_date: Optional date to split train and test sets for visualization.

    Returns:
        Dict with analysis artifacts (keys include 'results_reg', 'results_cluster',
        'dl_results', 'results_trees'). Empty dict is returned for empty input.
    """
    algorithms = {
        "regression": True,
        "trees": run_trees,
        "classification": False,
        "deep_learning": True
    }

    try:
        from engine.datasets.train_trees import train_tree_models
    except Exception:
        from train_trees import train_tree_models  # fallback for local runs

    print(f"Shape of input Metric DataFrame: {df.shape}")
    print(f"Type of input Metric DataFrame: {type(df)}")
    if df.empty:
        print("Metric DataFrame is empty. No analysis performed.")
        return {}  # return empty dict to respect typed return

    # Drop rows with NaN for the analysis (already defensive elsewhere)
    df = df.dropna()

    # Choose target series
    if 'target_close' in df.columns:
        y_series = df['target_close']
        target_col = 'target_close'
    else:
        y_series = df['close']
        target_col = 'close'

    # Determine x_values (keep robust type)
    if isinstance(y_series.index, pd.DatetimeIndex):
        x_values = y_series.index
        xlabel = 'Date'
    elif 'date' in df.columns:
        x_values = pd.to_datetime(df['date'], errors='coerce')
        xlabel = 'Date'
    else:
        x_values = pd.Index(range(len(y_series)))
        xlabel = 'Sample Index'

    # Plots (unchanged behavior)
    if show_plots:
        plt.figure(figsize=(10, 8))
        plt.plot(x_values, y_series.values, label=f'Target Close Prices ({"Tomorrow" if target_col=="target_close" else "Close"})', color='blue')
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

    # Feature selection via correlation
    sel_feature: List[str] = []
    for col in df.columns:
        if col in (target_col, 'close', 'date', 'stock_id'):
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        feature = df[col].to_numpy()
        if feature.size != y_series.values.size:
            print(f"Skipping correlation for {col} due to size mismatch.")
            continue
        corr = np.corrcoef(feature, y_series.values)[0, 1]
        print(f"Correlation between '{target_col}' and '{col}': {corr:.4f}")
        if abs(corr) > 0.1:
            sel_feature.append(col)

    excluded = ['SMA_5']
    sel_feature = [f for f in sel_feature if f not in excluded]

    # Ensure X is numeric; if no selected features, use numeric dtypes excluding target
    if sel_feature:
        X_df = df[sel_feature]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        X_df = df[numeric_cols]

    X = X_df.to_numpy()
    print(f"Converted Metric DataFrame to numpy arrays: X shape {X.shape}, y shape {y_series.size}")
    print(f"Selected features for model fitting: {list(X_df.columns)}")

    samples = X.shape[0]

    # Robust pred_date splitting: coerce x_values to datetime if possible
    split_idx = round(samples * 0.9)
    if pred_date is not None:
        try:
            p_date = pd.to_datetime(pred_date)
            xv = pd.to_datetime(pd.Index(x_values), errors='coerce')
            # produce boolean mask, safe for non-datetime entries
            future_mask = xv >= p_date
            if future_mask.any():
                split_idx = int(np.where(future_mask)[0][0])
                print(f"Analysis split at date: {pred_date} (index {split_idx})")
            else:
                print(f"Warning: pred_date {pred_date} after range. Falling back to {split_idx}/{samples} split.")
        except Exception as e:
            print(f"Error splitting analysis by pred_date ({e}). Falling back to 90% train.")

    X_train = X[:split_idx]
    y_train = y_series.values[:split_idx]
    X_test = X[split_idx:]
    y_test = y_series.values[split_idx:]

    # Prepare x indices for plotting
    x_values_arr = x_values if isinstance(x_values, pd.DatetimeIndex) else list(x_values)
    x_train_idx = x_values_arr[:split_idx]
    x_test_idx = x_values_arr[split_idx:]

    print(f"Split data into training and testing sets: X_train {X_train.shape}, X_test {X_test.shape}")

    results: Dict[str, Any] = {
        'results_reg': None,
        'results_cluster': None,
        'dl_results': None,
        'results_trees': None
    }

    # Regression
    if algorithms.get("regression"):
        print("\nStarting Regression Model Fitting...")
        results_reg = ml_model.regression_models(
            X_train, y_train, X_test, y_test,
            show_plots=show_plots, x_train_idx=x_train_idx, x_test_idx=x_test_idx,
            cv=TimeSeriesSplit(n_splits=30)
        )
        results['results_reg'] = results_reg

        try:
            preds = {k: results_reg[f"y_pred_{k}"] for k in ("lr", "svr") if f"y_pred_{k}" in results_reg}
            if preds:
                current_prices = df['close'].values[split_idx:]
                y_returns = (y_test - current_prices) / current_prices
                res = ml_model.example_run_ensemble_from_preds(
                    preds, y_returns,
                    score_type='price',
                    current_prices=current_prices,
                    threshold=0.002,
                    min_models_agree=1,
                    transaction_cost=0.0005,
                    show_plots=True,
                    x_idx=x_test_idx
                )
                print("Ensemble metrics:", res.get('metrics'))
        except Exception as e:
            print(f"Ensemble demo skipped (error): {e}")

    # Classification
    if algorithms.get("classification"):
        print("\nStarting Classification Model Fitting...")
        results_cluster = ml_model.clustering_model(X_train, y_train, X_test, y_test, n_clusters=30, show_plots=False)
        results['results_cluster'] = results_cluster
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

    # Deep Learning
    if algorithms.get("deep_learning"):
        print("\nStarting Deep Learning Model Fitting...")
        dl_results = dl_model.models(
            X_train, y_train, X_test, y_test,
            x_train_idx=x_train_idx, x_test_idx=x_test_idx
        )
        results['dl_results'] = dl_results
        try:
            preds_dl = {}
            if 'predictions' in dl_results:
                preds_dl['dl'] = dl_results['predictions']
            if preds_dl:
                preds_arr = preds_dl['dl']
                m = len(preds_arr)
                test_closes = df['close'].values[split_idx:]
                if m <= len(test_closes):
                    seq_len = len(test_closes) - m
                    start_idx = split_idx + seq_len - 1 if seq_len > 0 else split_idx
                    current_prices = df['close'].values[start_idx : start_idx + m]
                    y_true_actual = y_test[seq_len:] if seq_len > 0 else y_test[:m]
                    y_returns_dl = (y_true_actual - current_prices) / current_prices
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

    # Time-series trees
    if algorithms.get("trees"):
        tasks: List[Literal['regression', 'classification']] = ['regression', 'classification']
        for task in tasks:
            print(f"\nStarting Time-Series Tree Model Fitting ({task.upper()})...")
            try:
                results_trees = train_tree_models(df, target_col='Daily_Return', task=task, n_splits=30, n_iter=20, test_ratio=0.1, pred_date=pred_date, show_plots=show_plots)
                results['results_trees'] = results_trees

                if task == 'regression':
                    print(f"RF Test R²: {results_trees.get('rf_test_r2', 0):.4f}")
                    print(f"GB Test R²: {results_trees.get('gb_test_r2', 0):.4f}")
                    if 'test_index' in results_trees and 'y_test' in results_trees and show_plots:
                        try:
                            ti = results_trees['test_index']
                            y_test_vals = results_trees['y_test']
                            y_pred_rf = results_trees.get('y_pred_rf')
                            y_pred_gb = results_trees.get('y_pred_gb')
                            plt.figure(figsize=(12, 6))
                            plt.plot(ti, y_test_vals, '.-', label='Actual', color='black', alpha=0.8)
                            if y_pred_rf is not None:
                                plt.plot(ti, y_pred_rf, label='RF Predicted', color='tab:blue', alpha=0.8)
                            if y_pred_gb is not None:
                                plt.plot(ti, y_pred_gb, label='GB Predicted', color='tab:orange', alpha=0.8)
                            plt.title(f'Tree Model Predictions vs Actual ({task.upper()})')
                            is_date = isinstance(ti, pd.DatetimeIndex) or 'datetime' in str(getattr(ti, 'dtype', '')).lower()
                            if not is_date and ti is not None and len(ti) > 0:
                                is_date = hasattr(ti[0], 'year') or 'datetime' in str(type(ti[0])).lower()
                            plt.xlabel('Date' if is_date else 'Sample Index')
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
                    if 'test_index' in results_trees and 'y_test' in results_trees and show_plots:
                        try:
                            ti = results_trees['test_index']
                            y_test_vals = results_trees['y_test']
                            y_pred_rf = results_trees.get('y_pred_rf')
                            y_pred_gb = results_trees.get('y_pred_gb')
                            plt.figure(figsize=(12, 4))
                            plt.scatter(ti, y_test_vals, label='Actual', c='black', s=10)
                            if y_pred_rf is not None:
                                plt.scatter(ti, y_pred_rf, label='RF Pred', c='tab:blue', s=6, alpha=0.7)
                            if y_pred_gb is not None:
                                plt.scatter(ti, y_pred_gb, label='GB Pred', c='tab:orange', s=6, alpha=0.7)
                            plt.title(f'Tree Model Classification Predictions ({task.upper()})')
                            is_date = isinstance(ti, pd.DatetimeIndex) or 'datetime' in str(getattr(ti, 'dtype', '')).lower()
                            if not is_date and ti is not None and len(ti) > 0:
                                is_date = hasattr(ti[0], 'year') or 'datetime' in str(type(ti[0])).lower()
                            plt.xlabel('Date' if is_date else 'Sample Index')
                            plt.ylabel('Class Label')
                            plt.legend()
                            plt.gcf().autofmt_xdate()
                            plt.grid(True, alpha=0.3)
                            plt.show()
                        except Exception as e:
                            print(f"Could not plot classification predictions: {e}")

                if 'rf_importances' in results_trees:
                    top_features = sorted(results_trees['rf_importances'].items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"Top 5 Features ({task}): {top_features}")

            except Exception as e:
                print(f"Error running tree training pipeline for {task}: {e}")

    return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataframes and stats from a preprocessed JSON file (no charts).')
    parser.add_argument('json_path', nargs='?', default=os.path.join(os.path.dirname(__file__), 'preprocessed_1727.json'), help='Path to preprocessed JSON file')
    parser.add_argument('--start_date', dest='start_date', help='Filter start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end_date', dest='end_date', help='Filter end date (YYYY-MM-DD)', default=None)
    parser.add_argument('--pred_date', dest='pred_date', help='Date to start prediction splitting (YYYY-MM-DD)', default=None)
    parser.add_argument('--day_shift', dest='day_shift', type=int, default=-1, help='Number of days to shift for target_close (default: -1, next day)')
    parser.add_argument('--show_plots', '--show-plots', action='store_true', dest='show_plots', help='Display plots (default: True)')
    parser.add_argument('--no_plots', '--no-plots', action='store_false', dest='show_plots', help='Suppress all plots')
    parser.set_defaults(show_plots=True)
    args = parser.parse_args()

    try:
        results = prepare_analysis_data(
            args.json_path,
            start_date=args.start_date,
            end_date=args.end_date,
            day_shift=args.day_shift,
            pred_date=args.pred_date
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
    data_analyzer(results['df_metric'], run_trees=False, show_plots=args.show_plots, pred_date=args.pred_date)
    print(f"\n\nFiltered date range: {start_date} -> {end_date}")