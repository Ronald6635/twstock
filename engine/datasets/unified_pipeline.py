"""
Unified Pipeline Module

This module integrates all machine learning models from ml_model.py, train_trees.py, and dl_model.py
into a single pipeline using TimeSeriesSplit as the default cross-validation. It allows selecting
model types (traditional ML, tree-based, deep learning) and handles both regression and classification tasks.
Includes data preprocessing from data_analysis.py. Provides a main function to run the pipeline on prepared data.

Key features:
- Unified interface for model selection and training
- TimeSeriesSplit cross-validation for temporal data
- Support for regression and classification tasks
- Ensemble backtesting for signal generation
- Data preprocessing with technical indicators

Architecture notes:
- Integrates preprocessing, model training, and evaluation
- Uses scikit-learn's TimeSeriesSplit for CV
- Supports multiple model types in a single run
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Union, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score, classification_report

# Import local modules
try:
    from . import ml_model, train_trees, dl_model, data_analysis
except ImportError:
    # Fallback for direct execution
    import ml_model
    import train_trees
    import dl_model
    import data_analysis

__all__ = ["UnifiedPipeline"]
class UnifiedPipeline:
    """
    Unified pipeline for stock price prediction integrating multiple ML models.
    
    Supports traditional ML (LR, SVR), tree-based (RF, GB), and deep learning (Dense, LSTM) models
    with TimeSeriesSplit cross-validation and ensemble backtesting.
    """
    
    def __init__(
        self,
        model_types: List[Literal['traditional', 'trees', 'deep_learning']] = ['traditional', 'trees', 'deep_learning'],
        task: Literal['regression', 'classification'] = 'regression',
        n_splits: int = 5,
        test_ratio: float = 0.2,
        random_state: int = 42,
        use_ensemble: bool = True,
        show_plots: bool = True
    ):
        """
        Initialize the unified pipeline.
        
        Args:
            model_types: List of model types to include ('traditional', 'trees', 'deep_learning')
            task: Task type ('regression' or 'classification')
            n_splits: Number of TimeSeriesSplit folds
            test_ratio: Fraction of data for final holdout test
            random_state: Random seed for reproducibility
            use_ensemble: Whether to run ensemble backtesting
            show_plots: Whether to display plots
        """
        self.model_types = model_types
        self.task = task
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.use_ensemble = use_ensemble
        self.show_plots = show_plots
        self.lookback = 30  # Default lookback for sequence models (aligned with dl_model default)
        self.results = {}
        
    def preprocess_data(
        self,
        data_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        day_shift: int = -1,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Preprocess raw data into an ML-ready format.
        
        This method delegates initial data preparation to the data_analysis module,
        identifies numeric feature columns, and applies rigorous filters for
        finite values (no inf/NaN).
        
        Args:
            data_path: Path to the raw JSON data file.
            start_date: Start date for filtering in YYYY-MM-DD format (optional).
            end_date: End date for filtering in YYYY-MM-DD format (optional).
            day_shift: Lag/shift days for the target variable (default: -1).
            **kwargs: Additional ignored keyword arguments for forward compatibility.
            
        Returns:
            A cleaned pandas DataFrame containing features and targets.
            
        Raises:
            ValueError: If 'df_metric' is missing or no finite data remains after cleaning.
        """
        print("Preprocessing data and applying ML filters...")
        raw_data: Dict[str, Any] = data_analysis.prepare_analysis_data(
            data_path, start_date=start_date, end_date=end_date, day_shift=day_shift
        )
        self.data: Dict[str, Any] = raw_data

        if 'df_metric' not in raw_data:
            raise ValueError("The analysis module did not return 'df_metric'.")
            
        df: pd.DataFrame = raw_data['df_metric'].copy()
        
        # Identify possible target columns
        target_cols = ['target_close', 'Daily_Return']
        available_targets = [c for c in target_cols if c in df.columns]
        
        # Identify numeric feature columns (exclude targets, dates, metadata)
        excluded = set(target_cols) | {'close', 'open', 'high', 'low', 'date', 'stock_id', 'SMA_5', 'SMA_20'}
        self.feat_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
        
        if not self.feat_cols:
            raise ValueError("No numeric feature columns found for training.")
            
        # Clean data: Replace any inf with NaN and drop rows missing features or targets
        cols_to_clean = self.feat_cols + available_targets
        df[cols_to_clean] = df[cols_to_clean].replace([np.inf, -np.inf], np.nan)
        df_cleaned = df.dropna(subset=cols_to_clean)
        
        if df_cleaned.empty:
            raise ValueError("No finite data available after cleaning (all rows contained NaNs or Inf).")
            
        # Ensure temporal sorting
        if 'date' in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values('date')
            
        return df_cleaned

    def run_pipeline(
        self, 
        data_path: str, 
        test_ratio: float = 0.2, 
        use_ensemble: bool = True,
        include_raw: bool = False,
        **preprocess_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute the full ML pipeline from data loading to model evaluation.
        
        Args:
            data_path: Path to source data.
            test_ratio: Ratio of historical data to hold out for testing.
            use_ensemble: Boolean flag to enable ensemble backtesting.
            include_raw: Whether to include large raw X/y arrays in return dict.
            **preprocess_kwargs: Arbitrary keyword arguments passed to preprocess_data.
            
        Returns:
            Comprehensive results dictionary indexed by model type.
        """
        # 1. Cleaning & Preprocessing (Consolidated logic)
        df: pd.DataFrame = self.preprocess_data(data_path, **preprocess_kwargs)
        feat_cols = self.feat_cols
               
        # 2. Target Selection
        if self.task == 'regression':
            target_col = 'target_close' if 'target_close' in df.columns else 'close'
        else:
            target_col = 'Daily_Return'
            
        # 3. Temporal Split
        if 'date' in df.columns:
            df = df.set_index('date')
            
        n = len(df)
        split_idx = int(n * (1 - self.test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[feat_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feat_cols].values
        y_test = test_df[target_col].values
        
        # 4. Results Metadata
        train_idx = train_df.index if isinstance(train_df.index, pd.DatetimeIndex) else None
        test_idx = test_df.index if isinstance(test_df.index, pd.DatetimeIndex) else None
        
        print(f"Dataset Split: train={len(X_train)}, test={len(X_test)}")
        print(f"Feature set size: {len(feat_cols)}")
        
        results = {
            'metadata': {
                'task': self.task,
                'feature_columns': feat_cols,
                'target_column': target_col,
                'samples_train': len(X_train),
                'samples_test': len(X_test)
            },
            'test_idx': test_idx,
            'y_test': y_test
        }
        
        # Memory Optimization: Optional raw data inclusion
        if include_raw:
            results.update({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'train_idx': train_idx
            })
        
        # 5. Run selected model types
        if 'traditional' in self.model_types:
            results['traditional'] = self._run_traditional_models(X_train, y_train, X_test, y_test, train_idx, test_idx)
            
        if 'trees' in self.model_types:
            results['trees'] = self._run_tree_models(df, target_col)
            
        if 'deep_learning' in self.model_types:
            results['deep_learning'] = self._run_deep_learning_models(X_train, y_train, X_test, y_test, train_idx, test_idx)
            
        # Normalize model outputs to include a unified 'predictions' key where possible
        # This adapter helps the ensemble method find per-model predictions even when
        # individual training functions return different keys (e.g., 'y_pred_lr', 'y_pred_svr').
        def _collect_and_avg(arrays: list, length_target: Optional[int] = None):
            """Return a 1D numpy array averaged across input arrays, aligned to length_target if provided."""
            arrays = [np.asarray(a).astype(float) for a in arrays if a is not None]
            if not arrays:
                return None
            if length_target is not None:
                arrays = [a[-length_target:] if len(a) > length_target else a for a in arrays]
            if len(arrays) == 1:
                return arrays[0]
            # Broadcast to same length (trim to shortest)
            min_len = min(a.shape[0] for a in arrays)
            arrays = [a[-min_len:] if a.shape[0] > min_len else a for a in arrays]
            return np.mean(arrays, axis=0)

        # Top-level y_test length to align predictions if needed
        y_test_len = results.get('y_test').shape[0] if 'y_test' in results and results.get('y_test') is not None else None

        # Traditional models: look for common prediction keys and average them
        if 'traditional' in results and isinstance(results['traditional'], dict):
            trad = results['traditional']
            trad_preds = []
            for k in ('y_pred_lr', 'y_pred_svr', 'y_pred'):
                if k in trad:
                    trad_preds.append(trad[k])
            pred_arr = _collect_and_avg(trad_preds, length_target=y_test_len)
            if pred_arr is not None:
                results['traditional']['predictions'] = pred_arr

        # Tree models: aggregate RF/GB predictions when present
        if 'trees' in results and isinstance(results['trees'], dict):
            trees = results['trees']
            tree_preds = []
            for k in ('y_pred_rf', 'y_pred_gb', 'y_pred'):
                if k in trees:
                    tree_preds.append(trees[k])
            pred_arr = _collect_and_avg(tree_preds, length_target=y_test_len)
            if pred_arr is not None:
                results['trees']['predictions'] = pred_arr

        # Deep learning models generally return 'predictions' already; ensure alignment if needed
        if 'deep_learning' in results and isinstance(results['deep_learning'], dict):
            dl = results['deep_learning']
            if 'predictions' not in dl:
                for k in ('predictions', 'y_pred', 'y_pred_dl'):
                    if k in dl:
                        dl_preds = np.asarray(dl[k]).astype(float)
                        if y_test_len is not None and dl_preds.shape[0] > y_test_len:
                            dl_preds = dl_preds[-y_test_len:]
                        results['deep_learning']['predictions'] = dl_preds
                        break

        # Ensemble if requested and we have predictions
        if self.use_ensemble:
            results['ensemble'] = self._run_ensemble(results)
            
        self.results = results
        return results
    def _run_traditional_models(self, X_train, y_train, X_test, y_test, train_idx, test_idx):
        """Run traditional ML models (LR, SVR for regression; LR, RF, SVC for classification)."""
        print("\nRunning traditional ML models...")
        
        if self.task == 'regression':
            return ml_model.regression_models(
                X_train, y_train, X_test, y_test,
                show_plots=self.show_plots,
                x_train_idx=train_idx,
                x_test_idx=test_idx
            )
        else:
            # For classification, convert to binary direction
            y_train_cls = (y_train > 0).astype(int)
            y_test_cls = (y_test > 0).astype(int)
            return ml_model.classification_models(
                X_train, y_train_cls, X_test, y_test_cls, y_train, y_test,
                show_plots=self.show_plots,
                x_train_idx=train_idx,
                x_test_idx=test_idx
            )
    
    def _run_tree_models(self, data, target_col):
        """Run tree-based models using TimeSeriesSplit."""
        print("\nRunning tree-based models...")
        results = train_trees.train_tree_models(
            data,
            target_col=target_col,
            task=self.task,
            n_splits=self.n_splits,
            n_iter=20,
            random_state=self.random_state,
            test_ratio=self.test_ratio,
            save=False
        )

        if self.show_plots and 'test_index' in results and 'y_test' in results:
            import matplotlib.pyplot as plt
            try:
                ti = results['test_index']
                y_test_vals = results['y_test']
                y_pred_rf = results['y_pred_rf']
                y_pred_gb = results['y_pred_gb']
                
                plt.figure(figsize=(12, 6))
                plt.plot(ti, y_test_vals, '.-', label='Actual', color='black', alpha=0.8)
                plt.plot(ti, y_pred_rf, label='RF Predicted', color='tab:blue', alpha=0.8)
                plt.plot(ti, y_pred_gb, label='GB Predicted', color='tab:orange', alpha=0.8)
                
                plt.title(f'Tree Model Predictions vs Actual ({self.task.upper()})')
                plt.xlabel('Date' if isinstance(ti, pd.DatetimeIndex) else 'Sample Index')
                plt.ylabel('Target')
                plt.legend()
                if isinstance(ti, pd.DatetimeIndex):
                    plt.gcf().autofmt_xdate()
                plt.grid(True, alpha=0.3)
                plt.show()
            except Exception as e:
                print(f"Could not plot tree predictions: {e}")

        return results
    
    def _run_deep_learning_models(self, X_train, y_train, X_test, y_test, train_idx, test_idx):
        """Run deep learning models."""
        print("\nRunning deep learning models...")
        return dl_model.models(
            X_train, y_train, X_test, y_test,
            use_lstm=True,  # Use LSTM for time series
            x_train_idx=train_idx,
            x_test_idx=test_idx,
            show_plots=self.show_plots
        )
    
    # =============================================================================
    # ENSEMBLE METHODS
    # =============================================================================

    def _run_ensemble(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ensemble model using predictions from individual models.
        
        This method aligns predictions from different models (e.g., traditional, trees, deep_learning)
        by taking the shortest available prediction sequence and aligning y_test accordingly.
        
        Args:
            results (Dict[str, Any]): Dictionary containing model results with 'predictions' and 'y_test'.
                Expected keys: 'traditional', 'trees', 'deep_learning', 'y_test'.
                
        Returns:
            Dict[str, Any]: Dictionary with ensemble results, including aligned predictions and test targets.
                Keys: 'ensemble_predictions' (np.ndarray), 'aligned_y_test' (np.ndarray).
                
        Raises:
            ValueError: If no predictions are available for ensemble evaluation.
        """
        # Collect predictions from each model and ensure they are float arrays
        model_preds = {}
        for model_type in ['traditional', 'trees', 'deep_learning']:
            if model_type in results and isinstance(results[model_type], dict) and 'predictions' in results[model_type]:
                preds = np.asarray(results[model_type]['predictions']).astype(float)
                if preds.size > 0:
                    model_preds[model_type] = preds
        
        if not model_preds:
            raise ValueError("No predictions available for ensemble evaluation.")
            
        # Determine the shortest prediction length across all models
        min_len = min(len(p) for p in model_preds.values())
        y_test = np.asarray(results['y_test']).astype(float)
        
        # Align y_test and all predictions to the last min_len samples
        # This ensures that even if some models have lookback (seq_len), they still align by the final indices.
        y_test_aligned = y_test[-min_len:]
        aligned_preds = {mt: p[-min_len:] for mt, p in model_preds.items()}
        
        # Calculate ensemble averaging
        ensemble_preds = np.mean(list(aligned_preds.values()), axis=0)
        
        # Compute basic ensemble backtesting metrics
        # NOTE: Simple implementation; enhance with full portfolio simulation if needed
        if len(ensemble_preds) > 1:
            # Total return: cumulative percentage change from first to last prediction
            total_return = (ensemble_preds[-1] - ensemble_preds[0]) / ensemble_preds[0] if ensemble_preds[0] != 0 else 0.0
            
            # Sharpe ratio: simplified as mean return / std return (annualized roughly)
            returns = np.diff(ensemble_preds) / ensemble_preds[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0.0
            
            # Win rate: fraction where prediction sign matches actual sign (direction accuracy)
            # Need to compare changes to match 'direction' better
            actual_changes = np.diff(y_test_aligned)
            pred_changes = np.diff(ensemble_preds)
            win_rate = np.mean(np.sign(pred_changes) == np.sign(actual_changes)) if len(actual_changes) > 0 else 0.0
        else:
            total_return = 0.0
            sharpe = 0.0
            win_rate = 0.0
        
        metrics = {
            'total_return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate
        }
        
        # Add ensemble plot if show_plots is enabled
        if self.show_plots:
            try:
                import matplotlib.pyplot as plt
                ti = results.get('test_idx')
                if ti is not None:
                    # Account for alignment in time-series index
                    ti_aligned = ti[-min_len:]
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(ti_aligned, y_test_aligned, '.-', label='Actual', color='black', alpha=0.8)
                    plt.plot(ti_aligned, ensemble_preds, label='Ensemble Predicted', color='tab:purple', linewidth=2, alpha=0.9)
                    
                    plt.title(f'Ensemble Model Predictions vs Actual ({self.task.upper()})')
                    
                    is_date = isinstance(ti_aligned, pd.DatetimeIndex) or 'datetime' in str(getattr(ti_aligned, 'dtype', '')).lower()
                    if not is_date and ti_aligned is not None and len(ti_aligned) > 0:
                        is_date = hasattr(ti_aligned[0], 'year') or 'datetime' in str(type(ti_aligned[0])).lower()
                        
                    plt.xlabel('Date' if is_date else 'Sample Index')
                    plt.ylabel('Value')
                    plt.legend()
                    if isinstance(ti_aligned, pd.DatetimeIndex):
                        plt.gcf().autofmt_xdate()
                    plt.grid(True, alpha=0.3)
                    plt.show()
            except Exception as e:
                print(f"Could not plot ensemble predictions: {e}")

        return {
            'ensemble_predictions': ensemble_preds,
            'aligned_y_test': y_test_aligned,
            'metrics': metrics
        }
    
    def _safe_format(self, value, fmt='.4f'):
        """
        Safely format a value for display.
        
        Args:
            value: The value to format (numeric or string).
            fmt: Format specifier (default '.4f' for floats).
            
        Returns:
            str: Formatted string if numeric, else string representation.
        """
        if isinstance(value, (int, float)):
            return f"{value:{fmt}}"
        else:
            return str(value)
    
    def get_summary(self) -> str:
        """
        Generate a comprehensive text summary of the pipeline results.
        
        This method aggregates performance metrics from traditional models, 
        tree-based models, and ensemble strategies into a formatted string 
        suitable for human review or LLM analysis.

        Returns:
            A formatted string containing model comparisons and strategy metrics.
            
        Note:
            $R^2$ values are used for regression metrics, while accuracy 
            is used for classification tasks.
        """
        if not self.results:
            return "No results available. Run the pipeline first."
            
        summary = []
        summary.append("Pipeline Results Summary")
        summary.append("=" * 50)
        
        if 'traditional' in self.results:
            trad = self.results['traditional']
            if self.task == 'regression':
                summary.append(f"Linear Regression R: {self._safe_format(trad.get('r2_lr', 'N/A'))}")
                summary.append(f"SVR R: {self._safe_format(trad.get('r2_svr', 'N/A'))}")
            else:
                summary.append(f"Logistic Regression Acc: {self._safe_format(trad.get('logistic_regression', {}).get('accuracy', 'N/A'))}")
                summary.append(f"Random Forest Acc: {self._safe_format(trad.get('random_forest', {}).get('accuracy', 'N/A'))}")
                summary.append(f"SVC Acc: {self._safe_format(trad.get('svc', {}).get('accuracy', 'N/A'))}")
        
        if 'trees' in self.results:
            trees = self.results['trees']
            if self.task == 'regression':
                summary.append(f"Random Forest R: {self._safe_format(trees.get('rf_test_r2', 'N/A'))}")
                summary.append(f"Gradient Boosting R: {self._safe_format(trees.get('gb_test_r2', 'N/A'))}")
            else:
                summary.append(f"Random Forest Acc: {self._safe_format(trees.get('rf_test_acc', 'N/A'))}")
                summary.append(f"Gradient Boosting Acc: {self._safe_format(trees.get('gb_test_acc', 'N/A'))}")
        
        if 'deep_learning' in self.results:
            dl = self.results['deep_learning']
            summary.append(f"Deep Learning R: {self._safe_format(dl.get('r2_score', 'N/A'))}")
            summary.append(f"Deep Learning MAE: {self._safe_format(dl.get('mae', 'N/A'))}")
        
        if 'ensemble' in self.results:
            ens: Dict[str, Any] = self.results['ensemble']
            metrics: Dict[str, Any] = ens.get('metrics', {})
            summary.append(f"Ensemble Total Return: {self._safe_format(metrics.get('total_return', 'N/A'))}")
            summary.append(f"Ensemble Sharpe: {self._safe_format(metrics.get('sharpe', 'N/A'))}")
            summary.append(f"Ensemble Win Rate: {self._safe_format(metrics.get('win_rate', 'N/A'))}")

        # Add Chinese indicator explanations as requested
        summary.append("\n指標解釋:")
        summary.append("- R² (決定係數): 衡量模型解釋資料變異的比例。值越接近1.0越好；負值表示模型表現比簡單平均差。")
        summary.append("- MAE (平均絕對誤差): 預測值與實際值之間的平均絕對差異。值越小表示準確度越高。")
        summary.append("- Ensemble Total Return: 基於整體預測在測試期間的累計百分比回報。")
        summary.append("- Ensemble Sharpe: 風險調整後的回報指標（越高越好，年化近似值）。")
        summary.append("- Ensemble Win Rate: 預測方向（上漲/下跌）與實際價格變動相符的比例。")

        return "\n".join(summary)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main entry point for the pipeline module."""
    import argparse
    import json
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Unified ML Pipeline for Stock Price Prediction")
    parser.add_argument("data", type=str, help="Path to the input data file (JSON format)")
    parser.add_argument("--config", type=str, help="Path to the configuration file (JSON format)", default="")
    parser.add_argument("--output", type=str, help="Path to save the output results (JSON format)", default="results.json")
    parser.add_argument("--start_date", type=str, help="Start date for data filtering (YYYY-MM-DD)", default=None)
    parser.add_argument("--end_date", type=str, help="End date for data filtering (YYYY-MM-DD)", default=None)
    parser.add_argument("--day_shift", "--day-shift", type=int, help="Lag/shift days for the target variable", default=-1)
    parser.add_argument("--test_ratio", type=float, help="Fraction of data to use for testing", default=0.1)
    parser.add_argument("--n_splits", type=int, help="Number of splits for TimeSeriesSplit", default=5)
    parser.add_argument("--random_state", type=int, help="Random seed for reproducibility", default=42)
    parser.add_argument("--use_ensemble", action="store_true", help="Enable ensemble backtesting", default=True)
    parser.add_argument("--include_raw", action="store_true", help="Include large raw datasets in the output JSON", default=False)
    # Plotting toggles: default to True, but allow suppression with --no-plots
    parser.add_argument("--show_plots", "--show-plots", action="store_true", dest="show_plots", help="Display plots for model results")
    parser.add_argument("--no_plots", "--no-plots", action="store_false", dest="show_plots", help="Suppress all plots")
    parser.set_defaults(show_plots=True)
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    
    # Merge command-line args with config file, prioritizing command-line args
    pipeline_args = {k: v for k, v in vars(args).items() if v is not None}
    pipeline_args.update(config)
    
    # Convert boolean flags from strings to actual booleans
    for flag in ["use_ensemble", "show_plots"]:
        if flag in pipeline_args:
            pipeline_args[flag] = pipeline_args[flag] not in [0, "0", "false", "False"]
    
    # Convert date strings to actual dates
    for date_arg in ["start_date", "end_date"]:
        if date_arg in pipeline_args and isinstance(pipeline_args[date_arg], str):
            try:
                pipeline_args[date_arg] = datetime.strptime(pipeline_args[date_arg], "%Y-%m-%d").date()
            except ValueError:
                print(f"Invalid date format for {date_arg}. Expected YYYY-MM-DD.")
                sys.exit(1)
    
    print("Running pipeline with the following parameters:")
    for key, value in pipeline_args.items():
        print(f"  {key}: {value}")
    
    # Initialize and run the pipeline
    pipeline = UnifiedPipeline(
        model_types=pipeline_args.get("model_types", ["traditional", "trees", "deep_learning"]),
        task=pipeline_args.get("task", "regression"),
        n_splits=pipeline_args.get("n_splits", 5),
        test_ratio=pipeline_args.get("test_ratio", 0.2),
        random_state=pipeline_args.get("random_state", 42),
        use_ensemble=pipeline_args.get("use_ensemble", True),
        show_plots=pipeline_args.get("show_plots", False)
    )
    

    results = pipeline.run_pipeline(
        data_path=pipeline_args["data"],
        start_date=pipeline_args.get("start_date"),
        end_date=pipeline_args.get("end_date"),
        day_shift=pipeline_args.get("day_shift", -1),
        test_ratio=pipeline_args.get("test_ratio", 0.2),
        n_splits=pipeline_args.get("n_splits", 5),
        random_state=pipeline_args.get("random_state", 42),
        use_ensemble=pipeline_args.get("use_ensemble", True),
        show_plots=pipeline_args.get("show_plots", False),
        include_raw=pipeline_args.get("include_raw", False)
    )

    # Print summary of results
    print("\n" + pipeline.get_summary())

    # Output results to file    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
