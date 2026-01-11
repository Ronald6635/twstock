import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

from engine.datasets.data_analysis import prepare_analysis_data
from engine.datasets.train_trees import train_tree_models


def _make_records(n=60):
    start = datetime.date(2020, 1, 1)
    records = []
    for i in range(n):
        d = (start + timedelta(days=i)).isoformat()
        close = 100.0 + i * 0.2 + (np.random.randn() * 0.5)
        records.append({
            'date': d,
            'open': close - 0.5,
            'high': close + 0.5,
            'low': close - 1.0,
            'close': close,
            'volume': 1000 + i * 10,
            'eps': 1.0 + i * 0.01,
            'gross_profit': 500 + i,
            'daily_revenue': 200 + i,
            'MarginPurchaseBalanceChange': 0,
            'ShortSaleBalanceChange': 0,
            'foreign_investor_net': 0,
            'investment_trust_net': 0,
            'dealer_net': 0,
        })
    return records


def test_prepare_analysis_data_preserves_datetime_index():
    records = _make_records(40)
    res = prepare_analysis_data(records)
    df_metric = res['df_metric']
    assert isinstance(df_metric, pd.DataFrame)
    # index should be DatetimeIndex
    assert isinstance(df_metric.index, pd.DatetimeIndex)
    assert len(df_metric) > 0


def test_train_tree_models_returns_indices_and_predictions():
    records = _make_records(80)
    res = prepare_analysis_data(records)
    metric = res['df_metric'].copy()
    # ensure Daily_Return exists for target column
    if 'Daily_Return' not in metric.columns:
        metric['Daily_Return'] = metric['close'].pct_change().fillna(0)

    # Run with small n_iter and splits to keep the test fast
    out = train_tree_models(metric, target_col='Daily_Return', task='regression', n_splits=2, n_iter=1, test_ratio=0.2)

    assert 'train_index' in out and 'test_index' in out
    assert isinstance(out['train_index'], pd.DatetimeIndex)
    assert isinstance(out['test_index'], pd.DatetimeIndex)
    assert 'y_test' in out and 'y_pred_rf' in out and 'y_pred_gb' in out
    assert len(out['test_index']) == len(out['y_test']) == len(out['y_pred_rf']) == len(out['y_pred_gb'])
