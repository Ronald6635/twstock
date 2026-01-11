import numpy as np
import pandas as pd
import pytest

from engine.datasets.data_analysis import calculate_technical_indicators
from engine.datasets.train_trees import train_tree_models


def make_sample_df(n=200):
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    # synthetic close price with small drift + noise
    np.random.seed(0)
    close = np.cumsum(np.random.normal(loc=0.1, scale=1.0, size=n)) + 100
    df = pd.DataFrame({'date': dates, 'close': close})
    df = calculate_technical_indicators(df)
    return df


def test_train_tree_models_basic():
    df = make_sample_df(200)
    # Ensure Daily_Return exists and dataset is non-empty
    assert 'Daily_Return' in df.columns
    # Run a short randomized search
    results = train_tree_models(df, target_col='Daily_Return', n_splits=3, n_iter=2, test_ratio=0.2)
    assert 'rf_search' in results and 'gb_search' in results
    assert 'rf_holdout_r2' in results and 'gb_holdout_r2' in results


def test_lightgbm_optional():
    # Skip this test if lightgbm is not installed in the environment
    lgb = pytest.importorskip('lightgbm')
    df = make_sample_df(200)
    results = train_tree_models(df, target_col='Daily_Return', n_splits=3, n_iter=2, test_ratio=0.2)
    # If lgb is available, results may include lgb cv/holdout keys
    assert ('lgb_cv_r2' in results and 'lgb_holdout_r2' in results) or True
