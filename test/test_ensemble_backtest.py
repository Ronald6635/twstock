import numpy as np
from engine.datasets.ml_model import ensemble_backtest_signals

def test_ensemble_basic_score_backtest():
    rng = np.random.RandomState(0)
    n = 200
    # synthetic daily returns (~0.1% mean)
    y = rng.normal(loc=0.001, scale=0.01, size=n)
    # two simple predictive "models" with signal correlated to y
    m1 = y + rng.normal(scale=0.005, size=n)
    m2 = 0.8 * y + rng.normal(scale=0.006, size=n)
    preds = {"m1": m1, "m2": m2}
    res = ensemble_backtest_signals(y, preds, score_type="score", threshold=0.0, min_models_agree=1, transaction_cost=0.0)
    assert "metrics" in res
    assert res["signals"].shape[0] == n
    assert res["strategy_returns"].shape[0] == n
    # directional accuracy should be > 0.5 for correlated synthetic preds
    assert res["metrics"]["directional_accuracy"] >= 0.5