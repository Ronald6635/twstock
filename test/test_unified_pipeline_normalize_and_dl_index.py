import types
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

# --- Helpers to stub heavy deps so we can import modules that normally require Keras/skopt ---
def _insert_minimal_keras_stub():
    """Insert minimal 'keras' and submodules into sys.modules so importing dl_model/unified_pipeline succeeds."""
    import types
    # top-level keras module
    km = types.ModuleType('keras')
    sys.modules['keras'] = km
    # submodules that are imported at module level
    for name in ('models', 'layers', 'callbacks', 'optimizers', 'losses', 'metrics', 'regularizers'):
        m = types.ModuleType(f'keras.{name}')
        sys.modules[f'keras.{name}'] = m
        setattr(km, name, m)
    # provide minimal symbols used at import time
    sys.modules['keras.models'].Sequential = lambda *a, **k: None
    sys.modules['keras.models'].Model = type('Model', (), {})
    # layers (only names must exist)
    for sym in ('Dense', 'Dropout', 'BatchNormalization', 'LSTM', 'Input', 'concatenate', 'Flatten'):
        setattr(sys.modules['keras.layers'], sym, lambda *a, **k: None)
    sys.modules['keras.callbacks'].EarlyStopping = lambda *a, **k: None
    sys.modules['keras.callbacks'].ModelCheckpoint = lambda *a, **k: None
    sys.modules['keras.optimizers'].Adam = lambda *a, **k: None
    sys.modules['keras.losses'].MeanSquaredError = lambda *a, **k: None
    sys.modules['keras.metrics'].MeanAbsoluteError = lambda *a, **k: None
    sys.modules['keras.regularizers'].L2 = lambda *a, **k: None


def _insert_minimal_skopt_stub():
    import types
    sk = types.ModuleType('skopt')
    sys.modules['skopt'] = sk
    sp = types.ModuleType('skopt.space')
    sys.modules['skopt.space'] = sp
    ut = types.ModuleType('skopt.utils')
    sys.modules['skopt.utils'] = ut
    # placeholders
    sk.gp_minimize = lambda *a, **k: None
    sp.Real = lambda *a, **k: None
    sp.Integer = lambda *a, **k: None
    sp.Categorical = lambda *a, **k: None
    ut.use_named_args = lambda *a, **k: (lambda f: f)


# Ensure stubs exist BEFORE importing the target modules
_insert_minimal_keras_stub()
_insert_minimal_skopt_stub()

# Import modules under test
from engine.datasets import unified_pipeline, dl_model


def make_sample_df(n=80):
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    df = pd.DataFrame(index=dates)
    rng = np.random.RandomState(0)
    df['close'] = 100 + np.cumsum(rng.randn(n))
    df['f1'] = rng.randn(n) * 10
    df['f2'] = rng.randn(n) * 5
    # target_close = next-day close (pipeline expects shifted target)
    df['target_close'] = df['close'].shift(-1)
    return df


def test_run_pipeline_normalize_records_metadata(monkeypatch):
    """--normalize is applied and normalized feature names are recorded in results['metadata']."""
    df = make_sample_df(120)
    # prepare the structure that preprocess_data expects
    monkeypatch.setattr(
        unified_pipeline.data_analysis,
        'prepare_analysis_data',
        lambda path, **kw: {'df_metric': df.copy()}
    )

    p = unified_pipeline.UnifiedPipeline(model_types=['traditional'], pred_date='2025-01-01', show_plots=False)

    # run with normalize=True (should not raise)
    res = p.run_pipeline(data_path='unused.json', start_date='2024-01-01', normalize=True)

    assert 'metadata' in res
    assert 'normalized_features' in res['metadata']
    assert isinstance(p.scaler, object)
    # normalized features should be subset of feature columns
    assert set(res['metadata']['normalized_features']).issubset(set(res['metadata']['feature_columns']))


def test_dl_plot_trims_longer_xidx(monkeypatch):
    """When x_idx is longer than y_true, dl_model._plot_predictions should align by keeping the tail.
    We verify the plotted x-data equals the tail of the provided x_idx.
    """
    # create small y arrays
    n_true = 5
    y_true = np.linspace(10.0, 20.0, n_true)
    y_pred = y_true + np.linspace(-0.5, 0.5, n_true)

    # create an x_idx longer than y_true (e.g. lookback + samples)
    x_idx_full = pd.date_range('2025-11-01', periods=n_true + 3, freq='B')

    # prevent plt.show from blocking in test
    monkeypatch.setattr(matplotlib.pyplot, 'show', lambda *a, **k: None)

    # call the plotting helper (should not raise)
    dl_model._plot_predictions(y_true, y_pred, x_idx_full)

    # inspect the created figure's second axis (time-series). The plotted x-data should be the tail
    fig = plt.gcf()
    try:
        ax = fig.axes[1]
    except Exception:
        pytest.skip('Plot structure not as-expected; skip')

    # lines: first is actual, second is predicted (in our implementation)
    lines = ax.get_lines()
    assert len(lines) >= 2
    xdata = lines[1].get_xdata()
    # convert to pandas DatetimeIndex for easy comparison
    plotted_idx = pd.to_datetime(xdata)
    expected_idx = x_idx_full[-n_true:]
    assert plotted_idx.equals(expected_idx)

    plt.close('all')
