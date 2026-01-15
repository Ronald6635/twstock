import numpy as np
from engine.datasets.dl_model import _build_wide_deep_model


def test_build_dense_model_shape_and_final_units():
    input_shape = (10,)  # 10 features
    hidden_layers = (64, 32)
    model = _build_wide_deep_model(
        input_shape=input_shape,
        hidden_layers=hidden_layers,
        dropout_rate=0.1,
        final_dense_units=16,
        l2_reg=1e-4,
        use_lstm=False
    )
    assert model.output_shape == (None, 1)
    assert model.get_layer('final_dense').units == 16


def test_build_lstm_model_shape_and_final_units():
    input_shape = (5, 8)  # sequence length 5, 8 features
    hidden_layers = (32, 16)
    model = _build_wide_deep_model(
        input_shape=input_shape,
        hidden_layers=hidden_layers,
        dropout_rate=0.1,
        final_dense_units=12,
        l2_reg=1e-4,
        use_lstm=True
    )
    assert model.output_shape == (None, 1)
    assert model.get_layer('final_dense').units == 12
