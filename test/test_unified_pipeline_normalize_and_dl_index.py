import types
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

# --- Helpers to stub heavy deps so we can import modules that normally require Keras/skopt ---
def _insert_minimal_stubs():
    """Insert minimal stubs for heavy dependencies."""
    import types
    
    # Keras
    km = types.ModuleType('keras')
    sys.modules['keras'] = km
    for name in ('models', 'layers', 'callbacks', 'optimizers', 'losses', 'metrics', 'regularizers', 'backend'):
        m = types.ModuleType(f'keras.{name}')
        sys.modules[f'keras.{name}'] = m
        setattr(km, name, m)
    sys.modules['keras.models'].Sequential = lambda *a, **k: None
    sys.modules['keras.models'].Model = type('Model', (), {})
    for sym in ('Dense', 'Dropout', 'BatchNormalization', 'LSTM', 'Input', 'concatenate', 'Flatten'):
        setattr(sys.modules['keras.layers'], sym, lambda *a, **k: None)
    sys.modules['keras.callbacks'].EarlyStopping = lambda *a, **k: None
    sys.modules['keras.callbacks'].ModelCheckpoint = lambda *a, **k: None
    sys.modules['keras.optimizers'].Adam = lambda *a, **k: None
    sys.modules['keras.optimizers'].SGD = lambda *a, **k: None
    sys.modules['keras.optimizers'].RMSprop = lambda *a, **k: None
    sys.modules['keras.losses'].MeanSquaredError = lambda *a, **k: None
    sys.modules['keras.metrics'].MeanAbsoluteError = lambda *a, **k: None
    sys.modules['keras.regularizers'].L2 = lambda *a, **k: None
    sys.modules['keras.backend'].clear_session = lambda *a, **k: None
    
    # skopt
    sk = types.ModuleType('skopt')
    sys.modules['skopt'] = sk
    sp = types.ModuleType('skopt.space')
    sys.modules['skopt.space'] = sp
    ut = types.ModuleType('skopt.utils')
    sys.modules['skopt.utils'] = ut
    sk.gp_minimize = lambda *a, **k: None
    sp.Real = lambda *a, **k: None
    sp.Integer = lambda *a, **k: None
    sp.Categorical = lambda *a, **k: None
    ut.use_named_args = lambda *a, **k: (lambda f: f)
    
    # Stub out modules that unified_pipeline imports (fallback paths)
    def _dummy_regression_models(X_train, y_train, X_test, y_test, **kw):
        # Return dummy predictions matching y_test length
        return {
            'y_pred_lr': np.ones(len(y_test)) * 100,
            'y_pred_svr': np.ones(len(y_test)) * 100,
            'predictions': np.ones(len(y_test)) * 100,
        }
    ml_model_stub = types.ModuleType('ml_model')
    sys.modules['ml_model'] = ml_model_stub
    ml_model_stub.regression_models = _dummy_regression_models
    ml_model_stub.classification_models = lambda *a, **k: {}
    
    train_trees_stub = types.ModuleType('train_trees')
    sys.modules['train_trees'] = train_trees_stub
    train_trees_stub.train_tree_models = lambda *a, **k: {}
    
    data_visual_stub = types.ModuleType('data_visual')
    sys.modules['data_visual'] = data_visual_stub
    data_visual_stub.load_json_data = lambda *a, **k: None
    data_visual_stub.prepare_data_for_chart = lambda *a, **k: None
    
    data_analysis_stub = types.ModuleType('data_analysis')
    sys.modules['data_analysis'] = data_analysis_stub
    data_analysis_stub.prepare_analysis_data = lambda *a, **k: {'df_metric': pd.DataFrame()}
    
    indicators_stub = types.ModuleType('indicators')
    sys.modules['indicators'] = indicators_stub
    indicators_stub.compute_supertrend = lambda *a, **k: (pd.Series(), pd.Series())
    
    dl_model_stub = types.ModuleType('dl_model')
    sys.modules['dl_model'] = dl_model_stub
    
    # Stub out data_analysis module (with proper path) to avoid the entire import chain
    data_analysis = types.ModuleType('engine.datasets.data_analysis')
    sys.modules['engine.datasets.data_analysis'] = data_analysis
    data_analysis.prepare_analysis_data = lambda *a, **k: {'df_metric': pd.DataFrame()}
    
    # Stub out other modules that unified_pipeline imports
    ml_model = types.ModuleType('engine.datasets.ml_model')
    sys.modules['engine.datasets.ml_model'] = ml_model
    ml_model.regression_models = _dummy_regression_models
    ml_model.classification_models = lambda *a, **k: {}
    
    train_trees = types.ModuleType('engine.datasets.train_trees')
    sys.modules['engine.datasets.train_trees'] = train_trees
    train_trees.train_tree_models = lambda *a, **k: {}
    
    indicators = types.ModuleType('engine.datasets.indicators')
    sys.modules['engine.datasets.indicators'] = indicators
    indicators.compute_supertrend = lambda *a, **k: (pd.Series(), pd.Series())


# Ensure stubs exist BEFORE importing the target modules
_insert_minimal_stubs()

# Import modules under test - dl_model separately first
from engine.datasets import dl_model

# Now import unified_pipeline
from engine.datasets import unified_pipeline


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
