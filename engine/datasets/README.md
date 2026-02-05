# datasets

This folder contains utility scripts for processing, analyzing, and machine learning on stock data from the twstock engine.

## Scripts

### Data Processing & Analysis
- `preprocessing.py`: Merges and transforms raw FinMind API data into combined daily JSON/CSV files
- `data_analysis.py`: Prepares pandas DataFrames and statistics from preprocessed data, including technical indicators
- `data_visual.py`: Generates interactive Plotly charts from stock data
- `indicators.py`: Technical indicators for financial analysis (SuperTrend, Moving Average, RSI, Stochastic, MACD, OBV)
- `dict.md`: Field name dictionary for data columns

### Machine Learning Models
- `unified_pipeline.py`: **NEW** Unified ML pipeline integrating all models with TimeSeriesSplit cross-validation (uses PyTorch backend for DL)
- `ml_model.py`: Traditional ML models (Linear Regression, SVR, Random Forest, SVC) with hyperparameter tuning
- `train_trees.py`: Time-series optimized tree-based models (Random Forest, Gradient Boosting, LightGBM)
- `dl_model.py`: Deep learning models (LSTM, Dense networks) using Keras 3 with PyTorch backend

## Installation

Install required dependencies for machine learning features:

```bash
pip install scikit-learn scikit-optimize torch torchvision keras pandas numpy matplotlib
```

## Usage Examples

### Python API

```python
# Preprocess combined data
from engine.datasets.preprocessing import preprocess_data
preprocess_data("input.json", "output.json", "output.csv")

# Prepare analysis data with technical indicators
from engine.datasets.data_analysis import prepare_analysis_data
results = prepare_analysis_data("preprocessed.json")

# Generate visualization
from engine.datasets.data_visual import prepare_data_for_chart
price_data, inst_data, margin_data, rev_data = prepare_data_for_chart(data)

# Run unified ML pipeline
from engine.datasets.unified_pipeline import UnifiedPipeline
pipeline = UnifiedPipeline(model_types=['traditional', 'trees', 'deep_learning'])
results = pipeline.run_pipeline("preprocessed_stock.json")
print(pipeline.get_summary())
```

### CLI Usage

#### Data Processing Pipeline
```bash
# Preprocess FinMind data into merged daily files
python preprocessing.py path/to/combined_input.json --out-json output.json --out-csv output.csv

# Prepare dataframes and stats from preprocessed JSON (no charts)
python data_analysis.py path/to/preprocessed.json

# Generate interactive chart.html from preprocessed JSON
python data_visual.py path/to/preprocessed.json --stock_id 2330
```

#### Technical Indicators
```bash
# Compute technical indicators with synthetic data (default parameters)
python indicators.py

# Compute SuperTrend with custom parameters
python indicators.py --period 10 --multiplier 3.0

# Load data from JSON file and compute indicators
python indicators.py path/to/preprocessed.json --start_date 2024-01-01
```

#### Machine Learning Pipeline
```bash
# Run complete ML pipeline with all model types
python unified_pipeline.py path/to/preprocessed.json --model-types traditional trees deep_learning

# Run only traditional ML models for regression
python unified_pipeline.py path/to/preprocessed.json --model-types traditional --task regression

# Run classification with tree models only
python unified_pipeline.py path/to/preprocessed.json --model-types trees --task classification --no-plots

# Run deep learning with Bayesian hyperparameter tuning
python dl_model.py  # (when called with tune_hyperparams=True in code)
```

## Machine Learning Features

### Unified Pipeline (`unified_pipeline.py`)
- **TimeSeriesSplit Cross-Validation**: Prevents data leakage in temporal data
- **Multiple Model Types**: Traditional ML, Tree-based, Deep Learning
- **Ensemble Backtesting**: Combines predictions for trading signals
- **Technical Indicators**: Automatic calculation of SMA, RSI, MACD, etc.
- **Flexible Configuration**: Choose model types, tasks, and parameters

### Model Types

#### Traditional ML (`ml_model.py`)
- Linear Regression with Polynomial Features
- Support Vector Regression (SVR) with hyperparameter tuning
- Random Forest and SVC for classification
- Uses TimeSeriesSplit for proper temporal validation

#### Tree-Based Models (`train_trees.py`)
- Random Forest Regressor/Classifier
- Gradient Boosting Regressor/Classifier
- LightGBM integration
- Optimized for time-series data with proper CV

#### Deep Learning (`dl_model.py`)
- LSTM networks for time-series prediction
- Dense neural networks as baseline
- Bayesian hyperparameter optimization
- Early stopping and model checkpointing

### Key Features
- **Temporal Data Handling**: All models respect time ordering
- **Hyperparameter Tuning**: Automated optimization for best performance
- **Technical Analysis**: Built-in calculation of 15+ technical indicators
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Visualization**: Performance plots and backtesting results

## Example Output

```
Pipeline Results Summary
==================================================
Linear Regression R²: -0.1629
SVR R²: -0.1003
Random Forest R²: 0.2341
Gradient Boosting R²: 0.3124
Deep Learning R²: 0.2876
Ensemble Total Return: 0.1542
Ensemble Sharpe: 1.2345
Ensemble Win Rate: 0.5432
```

## Contributing

Please add tests for any new dataset loaders or significant changes to existing ones. Follow the repository's testing and linting conventions.

For machine learning contributions:
- Ensure TimeSeriesSplit is used for temporal data validation
- Add proper hyperparameter tuning for new models
- Include performance metrics and backtesting for trading models
- Document model assumptions and limitations
