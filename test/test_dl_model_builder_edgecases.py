import numpy as np
from engine.datasets.dl_model import _build_wide_deep_model


def test_builder_accepts_numpy_int_for_final_dense_units():
    model = _build_wide_deep_model(
        input_shape=(10,),
        hidden_layers=(16, 8),
        dropout_rate=0.1,
        final_dense_units=np.int64(123),
        l2_reg=1e-4,
        use_lstm=False,
        activation='relu'
    )
    assert model.get_layer('final_dense').units == 123


def test_builder_activation_arg_used_not_params():
    # Ensure explicitly passed activation is applied and no NameError occurs
    model = _build_wide_deep_model(
        input_shape=(10,),
        hidden_layers=(16, 8),
        dropout_rate=0.1,
        final_dense_units=32,
        l2_reg=1e-4,
        use_lstm=False,
        activation='elu'
    )
    act = getattr(model.get_layer('final_dense'), 'activation', None)
    assert act is not None and getattr(act, '__name__', '') == 'elu'
