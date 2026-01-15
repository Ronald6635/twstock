"""
Deep Learning Model Module

This module implements deep learning models for stock price prediction using Keras with PyTorch backend.
It provides functions to train, evaluate, and visualize neural network models on financial data.

Key features:
- Simple feedforward neural network for regression tasks
- Model training with early stopping and validation
- Performance evaluation using R² score and MSE
- Visualization of predictions vs actual values

Architecture notes:
- Uses Keras 3 with PyTorch for backend execution
- Designed for time series prediction with tabular financial features
- Includes hyperparameter tuning and model persistence options
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import Keras with multi-backend support
try:
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Input, concatenate, Flatten
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.optimizers import Adam
    from keras.losses import MeanSquaredError
    from keras.metrics import MeanAbsoluteError
except ImportError as e:
    raise ImportError(
        "Keras is required for deep learning models. "
        "Install it with: pip install keras"
    ) from e

# Import scikit-optimize for hyperparameter tuning
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
except ImportError as e:
    raise ImportError(
        "scikit-optimize is required for hyperparameter tuning. "
        "Install it with: pip install scikit-optimize"
    ) from e


def create_sequences(data, target, sequence_length=60):
    """
    Create sequences for LSTM model.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


def _build_wide_deep_model(
    input_shape: Union[Tuple[int, ...], Tuple[int, int]],
    hidden_layers: Sequence[int],
    dropout_rate: float,
    final_dense_units: int = 64,
    l2_reg: float = 1e-4,
    use_lstm: bool = True,
    activation: str = 'relu'
) -> Model:
    """
    Build a Wide & Deep functional Keras model matching the topology used in `models()`.

    Args:
        input_shape: Input shape tuple. For LSTM this is (seq_len, n_features),
            for Dense this is (n_features,).
        hidden_layers: Sequence of integers for hidden/dense sizes.
        dropout_rate: Dropout rate applied after layers.
        final_dense_units: Units in the merged dense layer.
        l2_reg: L2 regularization strength for final dense layer.
        use_lstm: Whether to build LSTM (True) or Dense (False) deep path.

    Returns:
        Keras Model (uncompiled)
    """
    inputs = Input(shape=input_shape)
    if use_lstm:
        # Use up to two layers for LSTM stack; if only one provided, reuse it
        if len(hidden_layers) >= 2:
            lstm_0, lstm_1 = hidden_layers[0], hidden_layers[1]
        else:
            lstm_0 = lstm_1 = hidden_layers[0]
        x = LSTM(lstm_0, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(lstm_1, return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)
        deep = x
        wide = Flatten()(inputs)
    else:
        x = Dense(hidden_layers[0], activation=activation)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        for units in hidden_layers[1:]:
            x = Dense(units, activation=activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        deep = x
        wide = inputs

    merged = concatenate([deep, wide])
    # ensure final_dense_units is a native Python int and positive
    final_dense_units = int(final_dense_units)
    if final_dense_units <= 0:
        raise ValueError(f"final_dense_units must be a positive integer, got {final_dense_units!r}")

    merged_hidden = Dense(
        final_dense_units,
        activation=activation,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.L2(l2_reg),
        name='final_dense'
    )(merged)
    merged_hidden = BatchNormalization(name='final_bn')(merged_hidden)
    merged_hidden = Dropout(dropout_rate, name='final_dropout')(merged_hidden)
    output = Dense(1, activation='linear', name='output')(merged_hidden)
    model = Model(inputs=inputs, outputs=output)
    return model


def _select_optimizer(name: Optional[str], lr: float):
    """
    Return a Keras optimizer instance given a name and learning rate.

    Args:
        name: Optimizer name (case-insensitive): 'adam', 'sgd', or 'rmsprop'.
        lr: Learning rate float.

    Returns:
        An instantiated Keras optimizer.
    """
    name = (name or 'adam').lower()
    if name == 'adam':
        return Adam(learning_rate=lr)
    if name == 'sgd':
        return keras.optimizers.SGD(learning_rate=lr)
    if name in ('rmsprop', 'rms'):
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name!r}")


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_lstm: bool,
    sequence_length: int = 30,
    n_calls: int = 20
) -> Dict[str, Any]:
    """
    Perform Bayesian optimization for hyperparameter tuning using scikit-optimize.
    
    Optimizes learning_rate, dropout_rate, batch_size, and hidden_layers for Dense and LSTM models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        use_lstm: Whether to use LSTM model
        sequence_length: Sequence length for LSTM
        n_calls: Number of optimization calls
        
    Returns:
        Dict with best hyperparameters
    """
    # Define mapping to avoid passing complex objects through skopt's internal NumPy logic
    layer_map = {
        "64": (64,),
        "64,32": (64, 32),
        "64,32,16": (64, 32, 16),
        "128": (128,),
        "128,64": (128, 64),
        "128,64,32": (128, 64, 32),
        "256": (256,),
        "256,128": (256, 128),
        "256,128,64": (256, 128, 64)
    }

    # Define search space using string labels and additional parameters
    space = [
        Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(0.1, 0.5, name='dropout_rate'),
        Integer(16, 64, name='batch_size'),
        Categorical(list(layer_map.keys()), name='hidden_layers'),
        # Additional tunables
        Integer(16, 256, name='final_dense_units'),
        Real(1e-6, 1e-2, prior='log-uniform', name='l2_reg'),
        Categorical(['relu', 'elu'], name='activation'),
        Categorical(['adam', 'rmsprop', 'sgd'], name='optimizer')
    ]
    
    @use_named_args(space)
    def objective(**params: Any) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            **params: Hyperparameters sampled by gp_minimize.
            
        Returns:
            float: The value to minimize (e.g., negative R² score).
        """
        # Clear the Keras session to prevent memory accumulation in the backend
        keras.backend.clear_session()

        # Map the string label back to the tuple configuration
        actual_layers = layer_map[params['hidden_layers']]

        # Build and evaluate model with given params
        config = {
            'hidden_layers': actual_layers,
            'dropout_rate': params['dropout_rate'],
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'epochs': 100,  # Reduced for tuning
            'patience': 10,
            'validation_split': 0.1,
            'sequence_length': sequence_length
        }
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # Prepare data
        if use_lstm:
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_s, sequence_length)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_s, sequence_length)
            if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
                return 1.0  # Invalid configuration
            X_train_final, y_train_final = X_train_seq, y_train_seq
            X_test_final, y_test_final = X_test_seq, y_test_seq
            y_test_actual = y_test[sequence_length:]
        else:
            X_train_final, y_train_final = X_train_scaled, y_train_s
            X_test_final, y_test_final = X_test_scaled, y_test_s
            y_test_actual = y_test
        
        # Build Wide & Deep functional model (match models() topology)
        # Ensure defaults for final merged dense layer and regularization during tuning
        config.setdefault('final_dense_units', 64)
        config.setdefault('l2_reg', 1e-4)
        # Build model using sampled hyperparameters
        input_shape = (X_train_final.shape[1], X_train_final.shape[2]) if use_lstm else (X_train_final.shape[1],)
        model = _build_wide_deep_model(
            input_shape=input_shape,
            hidden_layers=actual_layers,
            dropout_rate=params['dropout_rate'],
            final_dense_units=params.get('final_dense_units', config['final_dense_units']),
            l2_reg=params.get('l2_reg', config['l2_reg']),
            use_lstm=use_lstm,
            activation=params.get('activation', 'relu')
        )
        # Select optimizer
        optimizer = _select_optimizer(params.get('optimizer', 'adam'), params['learning_rate'])
        model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True, verbose=0)
        
        try:
            model.fit(X_train_final, y_train_final, validation_split=config['validation_split'], 
                     epochs=config['epochs'], batch_size=params['batch_size'], callbacks=[early_stopping], verbose=0)
            pred_s = model.predict(X_test_final, verbose=0).ravel()
            predictions = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
            r2 = r2_score(y_test_actual, predictions)
            return -r2  # Minimize negative R²
        except Exception:
            return 1.0  # Return high loss for failed configurations
    
    # Run optimization
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42, n_jobs=-1)
    
    best_params = {
        'learning_rate': res.x[0],
        'dropout_rate': res.x[1],
        'batch_size': res.x[2],
        # map the best string result back to a list
        'hidden_layers': list(layer_map[res.x[3]]),
        # Additional params
        'final_dense_units': res.x[4],
        'l2_reg': res.x[5],
        'activation': res.x[6],
        'optimizer': res.x[7]
    }
    
    return best_params

def models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_config: Optional[Dict[str, Any]] = None,
    use_lstm: bool = True,
    tune_hyperparams: bool = False, # Whether to perform hyperparameter tuning
    x_train_idx: Optional[Sequence] = None,
    x_test_idx: Optional[Sequence] = None,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Train and evaluate a deep learning model for stock price prediction.
    
    This function supports both feedforward (Dense) and Recurrent (LSTM) neural networks.
    LSTM is generally much better for time series price data.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        y_test: Test target vector
        model_config: Optional model configuration overrides
        use_lstm: Whether to use LSTM (True) or Dense (False) model
        tune_hyperparams: If True, perform Bayesian optimization for hyperparameter tuning
        x_train_idx: Optional training indices
        x_test_idx: Optional test indices
        show_plots: Whether to display plots
        
    Returns:
        Dict containing model, history, predictions, and metrics
    """
    # Validate input shapes
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train must have same number of samples. "
            f"Got X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"
        )
    
    # Set default model configuration
    default_config = {
        'hidden_layers': [64, 32, 16],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 200,
        'patience': 20,
        'validation_split': 0.1,
        'sequence_length': 30,
        # Number of units in the final merged dense layer
        'final_dense_units': 64,
        # Optional L2 regularization strength for final dense layer
        'l2_reg': 1e-4,
        'optimizer': 'adam'
    }
    config = {**default_config, **(model_config or {})}
    
    # Perform hyperparameter tuning if requested
    if tune_hyperparams:
        print("Performing hyperparameter tuning with Bayesian optimization...")
        best_config = hyperparameter_search(X_train, y_train, X_test, y_test, use_lstm, config['sequence_length'])
        config.update(best_config)
        
        # Ensure integer hyperparameters are cast to standard Python ints
        int_params: List[str] = ['batch_size', 'final_dense_units', 'epochs', 'patience']
        for param in int_params:
            if param in config:
                config[param] = int(config[param])
        
        # Handle hidden_layers list if present
        if 'hidden_layers' in config:
            config['hidden_layers'] = [int(layer) for layer in config['hidden_layers']]

        print(f"Best hyperparameters found: {best_config}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale target variable
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Prepare holder for adjusted test index when sequences are used
    x_test_idx_actual: Optional[Sequence] = None

    if use_lstm:
        # Prepare sequences for LSTM
        seq_len = config['sequence_length']
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_s, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_s, seq_len)
        
        if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
            print("Not enough data for the specified sequence length. Falling back to Dense model.")
            use_lstm = False
        else:
            X_train_final, y_train_final = X_train_seq, y_train_seq
            X_test_final, y_test_final = X_test_seq, y_test_seq
            y_test_actual = y_test[seq_len:]
            if x_test_idx is not None:
                try:
                    x_test_idx_actual = x_test_idx[seq_len:]
                except Exception:
                    x_test_idx_actual = None
    
    if not use_lstm:
        X_train_final, y_train_final = X_train_scaled, y_train_s
        X_test_final, y_test_final = X_test_scaled, y_test_s
        y_test_actual = y_test
        x_test_idx_actual = x_test_idx

    # Build model using helper that implements the Wide & Deep topology
    input_shape = (X_train_final.shape[1], X_train_final.shape[2]) if use_lstm else (X_train_final.shape[1],)
    model = _build_wide_deep_model(
        input_shape=input_shape,
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate'],
        final_dense_units=config['final_dense_units'],
        l2_reg=config['l2_reg'],
        use_lstm=use_lstm
    )
    
    # Compile model
    optimizer = _select_optimizer(config.get('optimizer', 'adam'), config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print(f"Training {'LSTM' if use_lstm else 'Dense'} deep learning model...")
    try:
        history = model.fit(
            X_train_final, y_train_final,
            validation_split=config['validation_split'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e
    
    # Make predictions
    pred_s = model.predict(X_test_final).ravel()
    predictions = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    r2 = r2_score(y_test_actual, predictions)
    mse = mean_squared_error(y_test_actual, predictions)
    mae = np.mean(np.abs(y_test_actual - predictions))
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Create visualizations (pass date indices when available)
    if show_plots:
        _plot_predictions(y_test_actual, predictions, x_idx=x_test_idx_actual)
        _plot_training_history(history)
    
    return {
        'model': model,
        'history': history,
        'predictions': predictions,
        'r2_score': r2,
        'mse': mse,
        'mae': mae,
        'scaler': scaler
    }


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, x_idx: Optional[Sequence] = None) -> None:
    """
    Create scatter plot and time series comparison of predictions vs actual values.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        x_idx: Optional x-axis indices (dates or sequence) matching y_true/y_pred
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: No data available to plot predictions.")
        return

    # NOTE: Validate that x_idx length matches scientific data to prevent plotting crashes
    if x_idx is not None and len(x_idx) != len(y_true):
        # Identify 'gotcha': LSTM lookback often causes indices to be longer than predictions if not sliced
        print(f"Warning: Index length ({len(x_idx)}) mismatch with data ({len(y_true)}). Disabling x_idx.")
        x_idx = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Close Prices')
    ax1.set_ylabel('Predicted Close Prices')
    ax1.set_title('Actual vs Predicted Close Prices (Deep Learning)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time series plot: use x_idx if provided
    if x_idx is not None:
        try:
            ax2.plot(x_idx, y_true, '.-', label='Actual', color='blue', alpha=0.7)
            ax2.plot(x_idx, y_pred, '--', label='Predicted', color='red', alpha=0.7)
            
            # Robust check for datetime type to determine axis label
            is_date = isinstance(x_idx, pd.DatetimeIndex) or 'datetime' in str(getattr(x_idx, 'dtype', '')).lower()
            if not is_date and x_idx is not None and len(x_idx) > 0:
                is_date = hasattr(x_idx[0], 'year') or 'datetime' in str(type(x_idx[0])).lower()
                
            ax2.set_xlabel('Date' if is_date else 'Sample Index')
            fig.autofmt_xdate()
        except Exception as e:
            # HACK: Fallback to sample index if provided x_idx is incompatible with matplotlib plot
            print(f"Time series plot error: {e}")
            ax2.plot(y_true, '.-', label='Actual', color='blue', alpha=0.7)
            ax2.plot(y_pred, '--', label='Predicted', color='red', alpha=0.7)
            ax2.set_xlabel('Sample Index')
    else:
        ax2.plot(y_true, '.-', label='Actual', color='blue', alpha=0.7)
        ax2.plot(y_pred, '--', label='Predicted', color='red', alpha=0.7)
        ax2.set_xlabel('Sample Index')

    ax2.set_ylabel('Close Price')
    ax2.set_title('Predicted vs Actual Close Prices Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _plot_training_history(history: keras.callbacks.History) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Keras training history object
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage and testing
if __name__ == '__main__':
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Simulate stock features and target
    X = np.random.randn(n_samples, n_features)
    # Create some correlation with target
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5 + 100
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    results = models(X_train, y_train, X_test, y_test)
    
    print("\nModel training completed successfully!")
    print(f"Final R² Score: {results['r2_score']:.4f}")