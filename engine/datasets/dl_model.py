"""
Deep Learning Model Module

This module implements deep learning models for stock price prediction using TensorFlow/Keras.
It provides functions to train, evaluate, and visualize neural network models on financial data.

Key features:
- Simple feedforward neural network for regression tasks
- Model training with early stopping and validation
- Performance evaluation using R² score and MSE
- Visualization of predictions vs actual values

Architecture notes:
- Uses TensorFlow/Keras for neural network implementation
- Designed for time series prediction with tabular financial features
- Includes hyperparameter tuning and model persistence options
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import TensorFlow with fallback for environments without GPU support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for deep learning models. "
        "Install it with: pip install tensorflow"
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

def models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_config: Optional[Dict[str, Any]] = None,
    use_lstm: bool = True,
    x_train_idx: Optional[Sequence] = None,
    x_test_idx: Optional[Sequence] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a deep learning model for stock price prediction.
    
    This function supports both feedforward (Dense) and Recurrent (LSTM) neural networks.
    LSTM is generally much better for time series price data.
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
        'sequence_length': 30
    }
    config = {**default_config, **(model_config or {})}
    
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

    # Build model
    model = Sequential()
    
    if use_lstm:
        # LSTM model
        model.add(LSTM(
            config['hidden_layers'][0], 
            return_sequences=True,
            input_shape=(X_train_final.shape[1], X_train_final.shape[2])
        ))
        model.add(Dropout(config['dropout_rate']))
        model.add(LSTM(config['hidden_layers'][1], return_sequences=False))
        model.add(Dropout(config['dropout_rate']))
    else:
        # Input layer (Dense)
        model.add(Dense(
            config['hidden_layers'][0], 
            activation='relu', 
            input_shape=(X_train_final.shape[1],)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout_rate']))
        
        # Hidden layers
        for units in config['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(config['dropout_rate']))
    
    # Output layer (regression)
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    optimizer = Adam(learning_rate=config['learning_rate'])
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
            ax2.plot(x_idx, y_pred, label='Predicted', color='red', alpha=0.7)
            ax2.set_xlabel('Date' if (hasattr(x_idx, 'dtype') and 'datetime' in str(x_idx.dtype)) else 'Sample Index')
            fig.autofmt_xdate()
        except Exception:
            ax2.plot(y_true, '.-', label='Actual', color='blue', alpha=0.7)
            ax2.plot(y_pred, label='Predicted', color='red', alpha=0.7)
            ax2.set_xlabel('Sample Index')
    else:
        ax2.plot(y_true, '.-', label='Actual', color='blue', alpha=0.7)
        ax2.plot(y_pred, label='Predicted', color='red', alpha=0.7)
        ax2.set_xlabel('Sample Index')

    ax2.set_ylabel('Close Price')
    ax2.set_title('Predicted vs Actual Close Prices Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _plot_training_history(history: tf.keras.callbacks.History) -> None:
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