import types
from types import SimpleNamespace

import numpy as np

import engine.datasets.dl_model as dl


def test_build_wide_deep_activation_applies():
    model = dl._build_wide_deep_model(
        input_shape=(10,), hidden_layers=(32, 16), dropout_rate=0.1, final_dense_units=12, l2_reg=1e-4, use_lstm=False, activation='elu'
    )
    # Check final_dense activation is ELU
    act_name = getattr(model.get_layer('final_dense'), 'activation', None)
    assert act_name is not None and getattr(act_name, '__name__', '') == 'elu'


def test_hyperparameter_search_includes_new_params(monkeypatch):
    # Create tiny synthetic data
    rng = np.random.RandomState(0)
    X_train = rng.rand(20, 5)
    y_train = rng.rand(20)
    X_test = rng.rand(10, 5)
    y_test = rng.rand(10)

    # Monkeypatch gp_minimize to return a predictable result
    def fake_gp_minimize(objective, space, n_calls, random_state):
        # Return sample x that matches the extended space ordering
        # [learning_rate, dropout_rate, batch_size, hidden_layers, final_dense_units, l2_reg, activation, optimizer]
        return SimpleNamespace(x=[1e-3, 0.2, 32, '64,32', 128, 1e-5, 'elu', 'rmsprop'])

    monkeypatch.setattr(dl, 'gp_minimize', fake_gp_minimize)

    res = dl.hyperparameter_search(X_train, y_train, X_test, y_test, use_lstm=False, sequence_length=5, n_calls=1)

    assert 'final_dense_units' in res and res['final_dense_units'] == 128
    assert 'l2_reg' in res and abs(res['l2_reg'] - 1e-5) < 1e-12
    assert res.get('activation') == 'elu'
    assert res.get('optimizer') == 'rmsprop'
