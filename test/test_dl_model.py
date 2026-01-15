import numpy as np

from engine.datasets.dl_model import models


def test_models_dense_builds_and_predicts():
    np.random.seed(0)
    X_train = np.random.rand(50, 5)
    y_train = np.random.rand(50)
    X_test = np.random.rand(10, 5)
    y_test = np.random.rand(10)

    # Keep training short to make the test fast
    config = {
        'epochs': 1,
        'patience': 1,
        'validation_split': 0.1,
        'final_dense_units': 8,
        'l2_reg': 1e-5
    }

    res = models(
        X_train, y_train, X_test, y_test,
        model_config=config,
        use_lstm=False,
        tune_hyperparams=False,
        show_plots=False
    )

    assert res['model'].output_shape == (None, 1)
    assert len(res['predictions']) == len(y_test)


def test_models_lstm_builds_and_predicts():
    np.random.seed(1)
    X_train = np.random.rand(30, 4)
    y_train = np.random.rand(30)
    X_test = np.random.rand(10, 4)
    y_test = np.random.rand(10)

    seq_len = 2
    config = {
        'sequence_length': seq_len,
        'epochs': 1,
        'patience': 1,
        'validation_split': 0.1,
        'final_dense_units': 8,
        'l2_reg': 1e-5
    }

    res = models(
        X_train, y_train, X_test, y_test,
        model_config=config,
        use_lstm=True,
        tune_hyperparams=False,
        show_plots=False
    )

    assert res['model'].output_shape == (None, 1)
    # For LSTM, predictions are made on sequences, length = len(X_test) - seq_len
    assert len(res['predictions']) == max(0, len(y_test) - seq_len)
