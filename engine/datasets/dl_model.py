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
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import TensorFlow with fallback for environments without GPU support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for deep learning models. "
        "Install it with: pip install tensorflow"
    ) from e


def models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train and evaluate a deep learning model for stock price prediction.
    
    This function creates a feedforward neural network using TensorFlow/Keras,
    trains it on the provided training data, evaluates performance on test data,
    and generates visualizations comparing predictions to actual values.
    
    The model architecture consists of multiple dense layers with dropout and
    batch normalization for regularization. Early stopping is used to prevent
    overfitting, and the best model weights are saved during training.
    
    Args:
        X_train: Training feature matrix of shape (n_train_samples, n_features).
        y_train: Training target vector of shape (n_train_samples,).
        X_test: Test feature matrix of shape (n_test_samples, n_features).
        y_test: Test target vector of shape (n_test_samples,).
        model_config: Optional dictionary with model hyperparameters. Defaults to:
            {
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10,
                'validation_split': 0.2
            }
    
    Returns:
        Dictionary containing:
            - 'model': Trained Keras model instance
            - 'history': Training history object with loss curves
            - 'predictions': Predicted values on test set
            - 'r2_score': R² coefficient of determination
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
    
    Raises:
        ValueError: If input arrays have incompatible shapes or insufficient data
        RuntimeError: If model training fails
    
    Example:
        Basic usage with default configuration:
        
        >>> results = dl_model(X_train, y_train, X_test, y_test)
        >>> print(f"R² Score: {results['r2_score']:.4f}")
        >>> print(f"MSE: {results['mse']:.4f}")
        
        Custom model configuration:
        
        >>> config = {
        ...     'hidden_layers': [128, 64, 32],
        ...     'dropout_rate': 0.3,
        ...     'learning_rate': 0.0005,
        ...     'epochs': 200
        ... }
        >>> results = dl_model(X_train, y_train, X_test, y_test, config)
        
    Note:
        - Input features are automatically scaled using StandardScaler
        - Model weights are saved to 'best_model.h5' during training  # Updated filename
        - Early stopping monitors validation loss with patience parameter
        - For time series data, consider using LSTM layers instead of dense layers
    """
    # Validate input shapes
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train must have same number of samples. "
            f"Got X_train: {X_train.shape[0]}, y_train: {y_train.shape[0]}"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_test and y_test must have same number of samples. "
            f"Got X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}"
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have same number of features. "
            f"Got X_train: {X_train.shape[1]}, X_test: {X_test.shape[1]}"
        )
    
    # Set default model configuration
    default_config = {
        'hidden_layers': [64, 32, 16],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 1000,
        'patience': 100,
        'validation_split': 0.1
    }
    config = {**default_config, **(model_config or {})}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale target variable
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    
    # Build model
    model = Sequential()
    
    # Input layer
    model.add(Dense(
        config['hidden_layers'][0], 
        activation='relu', 
        input_shape=(X_train.shape[1],)
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
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',  # Changed from 'best_model.keras' to avoid Windows filename issues
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("Training deep learning model...")
    try:
        history = model.fit(
            X_train_scaled, y_train_s,
            validation_split=config['validation_split'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e
    
    # Make predictions
    pred_s = model.predict(X_test_scaled).ravel()
    predictions = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = np.mean(np.abs(y_test - predictions))
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Create visualizations
    _plot_predictions(y_test, predictions)
    _plot_training_history(history)
    
    return {
        'model': model,
        'history': history,
        'predictions': predictions,
        'r2_score': r2,
        'mse': mse,
        'mae': mae,
        'scaler': scaler  # Include scaler for future predictions
    }


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Create scatter plot and time series comparison of predictions vs actual values.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
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
    
    # Time series plot
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