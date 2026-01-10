# datasets

This folder contains utility scripts for processing and analyzing stock data from the twstock engine.

## Scripts

- `preprocessing.py`: Merges and transforms raw FinMind API data into combined daily JSON/CSV files
- `data_analysis.py`: Prepares pandas DataFrames and statistics from preprocessed data
- `data_visual.py`: Generates interactive Plotly charts from stock data
- `dict.md`: Field name dictionary for data columns

## Usage Examples

### Python API

```python
# Preprocess combined data
from engine.datasets.preprocessing import preprocess_data
preprocess_data("input.json", "output.json", "output.csv")

# Prepare analysis data
from engine.datasets.data_analysis import prepare_analysis_data
results = prepare_analysis_data("preprocessed.json")

# Generate visualization
from engine.datasets.data_visual import prepare_data_for_chart
price_data, inst_data, margin_data, rev_data = prepare_data_for_chart(data)
```

### CLI Usage

#### Preprocessing Script
```bash
# Preprocess FinMind data into merged daily files
python preprocessing.py path/to/combined_input.json --out-json output.json --out-csv output.csv
```

#### Data Analysis Script
```bash
# Prepare dataframes and stats from preprocessed JSON (no charts)
python data_analysis.py path/to/preprocessed.json
```

#### Data Visualization Script
```bash
# Generate interactive chart.html from preprocessed JSON
python data_visual.py path/to/preprocessed.json --stock_id 2330
```

## Contributing

Please add tests for any new dataset loaders or significant changes to existing ones. Follow the repository's testing and linting conventions.
