"""
Time-series training pipeline for tree-based models (Random Forest, Gradient Boosting, LightGBM).

This module supports both Regression (predicting returns) and Classification (predicting direction).
It uses TimeSeriesSplit and RandomizedSearchCV to optimize models while respecting temporal order.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Literal, Sequence
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.metrics import r2_score, accuracy_score, f1_score, classification_report


def train_tree_models(
    metric_df: pd.DataFrame,
    target_col: str = 'Daily_Return',
    task: Literal['regression', 'classification'] = 'regression',
    n_splits: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
    test_ratio: float = 0.1,
    save: bool = False,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train tree-based models on time-series data.

    Args:
        metric_df: DataFrame with numeric features.
        target_col: Column to base target on (e.g., 'Daily_Return'). We predict next day: shift(-1).
        task: 'regression' to predict raw values, 'classification' to predict binary direction (Up/Down).
        n_splits: TimeSeriesSplit folds for CV.
        n_iter: Number of parameter samples for RandomizedSearchCV.
        random_state: Seed for reproducibility.
        test_ratio: Fraction to reserve for final holdout test.
        save: Whether to save the learned estimators to disk.
        save_dir: Directory to save models.

    Returns:
        Dict with best estimators, metrics, and feature importances.
    """
    if metric_df.empty:
        raise ValueError("Empty metric_df supplied")

    df = metric_df.copy()

    # Sort by date to ensure time-series integrity
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        # set date as index so downstream splits can preserve date indices
        df = df.set_index('date')

    # Target: next-day value/direction
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in metric_df")

    if task == 'regression':
        df['target_next'] = df[target_col].shift(-1)
    else:  # classification
        # 1 if next day return > 0, else 0
        df['target_next'] = (df[target_col].shift(-1) > 0).astype(int)
    
    df = df.dropna()  # drop the last row (no target) and any other NaNs

    # Drop non-feature columns
    excluded = {'target_next', target_col, 'close', 'target_close', 'date', 'stock_id'}
    feat_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    if not feat_cols:
        raise ValueError("No numeric feature columns found for tree training")

    X = df[feat_cols].to_numpy()
    y = df['target_next'].to_numpy()

    # Time-ordered train/test split
    n = X.shape[0]
    split_idx = int(n * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Capture train/test date indices (DatetimeIndex)
    train_index = df.index[:split_idx]
    test_index = df.index[split_idx:]
    tss = TimeSeriesSplit(n_splits=n_splits)
    
    # Define scoring and models based on task
    if task == 'regression':
        scoring = 'r2'
        rf_base = RandomForestRegressor(random_state=random_state)
        gb_base = GradientBoostingRegressor(random_state=random_state)
        rf_param = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        scoring = 'accuracy'
        rf_base = RandomForestClassifier(random_state=random_state)
        gb_base = GradientBoostingClassifier(random_state=random_state)
        rf_param = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [2, 4, 8],
            'class_weight': ['balanced', None]
        }

    gb_param = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9]
    }

    print(f"Starting {task} tree model training...")
    
    # 1. Random Forest
    rf_search = RandomizedSearchCV(
        rf_base, rf_param, n_iter=min(n_iter, 50),
        cv=tss, scoring=scoring, random_state=random_state, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    
    # 2. Gradient Boosting
    gb_search = RandomizedSearchCV(
        gb_base, gb_param, n_iter=min(n_iter, 50),
        cv=tss, scoring=scoring, random_state=random_state, n_jobs=-1
    )
    gb_search.fit(X_train, y_train)

    # Evaluation
    rf_best = rf_search.best_estimator_
    gb_best = gb_search.best_estimator_
    y_pred_rf = rf_best.predict(X_test)
    y_pred_gb = gb_best.predict(X_test)

    results = {
        'rf_model': rf_best,
        'gb_model': gb_best,
        'rf_best_params': rf_search.best_params_,
        'gb_best_params': gb_search.best_params_,
        'feature_columns': feat_cols,
        'rf_importances': dict(zip(feat_cols, rf_best.feature_importances_)),
        'gb_importances': dict(zip(feat_cols, gb_best.feature_importances_)),
        'train_index': train_index,
        'test_index': test_index,
        # Include test arrays for easy plotting/analysis
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_pred_gb': y_pred_gb,
    }

    if task == 'regression':
        results['rf_test_r2'] = r2_score(y_test, y_pred_rf)
        results['gb_test_r2'] = r2_score(y_test, y_pred_gb)
        print(f"RF CV R²: {rf_search.best_score_:.4f}, Test R²: {results['rf_test_r2']:.4f}")
        print(f"GB CV R²: {gb_search.best_score_:.4f}, Test R²: {results['gb_test_r2']:.4f}")
        # Print a quick sample of true vs predicted
        print("Sample test (true vs RF pred vs GB pred):")
        for a, b, c in list(zip(y_test[:5], y_pred_rf[:5], y_pred_gb[:5])):
            print(f"{a:.6f} | {b:.6f} | {c:.6f}")
    else:
        results['rf_test_acc'] = accuracy_score(y_test, y_pred_rf)
        results['gb_test_acc'] = accuracy_score(y_test, y_pred_gb)
        results['rf_report'] = classification_report(y_test, y_pred_rf, zero_division=0)
        results['gb_report'] = classification_report(y_test, y_pred_gb, zero_division=0)
        print(f"RF CV Acc: {rf_search.best_score_:.4f}, Test Acc: {results['rf_test_acc']:.4f}")
        print(f"GB CV Acc: {gb_search.best_score_:.4f}, Test Acc: {results['gb_test_acc']:.4f}")
        print("Sample test (true vs RF pred vs GB pred):")
        for a, b, c in list(zip(y_test[:5], y_pred_rf[:5], y_pred_gb[:5])):
            print(f"{a} | {b} | {c}")

    # Optional LightGBM Integration
    try:
        import lightgbm as lgb
        def train_lgbm():
            tss_lgb = TimeSeriesSplit(n_splits=n_splits)
            lgb_params = {
                'objective': 'regression' if task == 'regression' else 'binary',
                'metric': 'rmse' if task == 'regression' else 'binary_logloss',
                'verbosity': -1,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5
            }
            
            # Use last fold for simple "early stopping" simulation
            train_idx, val_idx = list(tss_lgb.split(X_train))[-1]
            dtrain = lgb.Dataset(X_train[train_idx], label=y_train[train_idx])
            dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx], reference=dtrain)
            
            try:
                model = lgb.train(
                    lgb_params, dtrain, num_boost_round=500,
                    valid_sets=[dval], early_stopping_rounds=50, verbose_eval=False
                )
            except TypeError:
                # Older/lightweight LightGBM builds may not accept early_stopping_rounds;
                # use callback API instead for compatibility.
                callbacks = [lgb.callback.early_stopping(50), lgb.callback.log_evaluation(period=0)]
                model = lgb.train(
                    lgb_params, dtrain, num_boost_round=500,
                    valid_sets=[dval], callbacks=callbacks
                )
            
            p_test = model.predict(X_test, num_iteration=model.best_iteration)
            if task == 'classification':
                p_test = (p_test > 0.5).astype(int)
                score = accuracy_score(y_test, p_test)
                print(f"LightGBM Test Accuracy: {score:.4f}")
                results['lgbm_test_acc'] = score
            else:
                score = r2_score(y_test, p_test)
                print(f"LightGBM Test R²: {score:.4f}")
                results['lgbm_test_r2'] = score
            results['lgbm_model'] = model
            
        train_lgbm()
    except Exception as e:
        print(f"LightGBM skipped: {e}")

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import joblib
        joblib.dump(rf_best, os.path.join(save_dir, f'rf_{task}.joblib'))
        joblib.dump(gb_best, os.path.join(save_dir, f'gb_{task}.joblib'))

    return results

