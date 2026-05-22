# Institutional Net Buy Fetcher

Batch-fetches institutional investor (法人) buy/sell data from the [FinMind](https://finmindtrade.com/) API for a set of TWSE stocks, normalizes the output, and exports it to CSV and optionally TFRecord format.

## Features

- **Three stock-source modes**: full TWSE list (`all`), a text file (`file`), or an explicit comma-separated list (`list`)
- **Quota-aware adaptive throttle**: monitors remaining API requests and adjusts sleep time dynamically
- **Retry with exponential backoff**: recovers from transient API errors automatically
- **Daily close price enrichment**: fetches and merges `close` price alongside institutional flow data
- **Dual export**: CSV (always) and TFRecord (optional, for TensorFlow pipelines)

## Output Schema

| Column | Type | Description |
|---|---|---|
| `date` | str | Trading date (YYYY-MM-DD) |
| `stock_id` | str | TWSE stock code |
| `stock_name` | str | Stock display name |
| `investor_name` | str | 外資 / 投信 / 自營商 … |
| `buy` | int | Buy volume (千股) |
| `sell` | int | Sell volume (千股) |
| `net_buy` | int | Net buy = buy − sell (千股) |
| `close` | float | Daily closing price (TWD); 0.0 if unavailable |

## Prerequisites

1. Python 3.10+, venv activated.
2. `FINMIND_API_KEY` set in a `.env` file at the project root (or as an environment variable).
3. `docs/tw_stock_info.json` present (required for `all` and name-mapping in all modes).

## Quick Start

```powershell
# Fetch the last 7 days for two specific stocks
python institutional_net_buy_fetcher.py `
    --stock-source list `
    --stocks 2330,2317 `
    --days 7

# Fetch the last 90 days for all TWSE stocks (with quota protection)
python institutional_net_buy_fetcher.py `
    --stock-source all `
    --days 90 `
    --end-date 2026-05-11 `
    --api-throttle-enabled `
    --export-tfrecord

# Also export a TFRecord file
python institutional_net_buy_fetcher.py `
    --stock-source list `
    --stocks 2330,2317 `
    --days 30 `
    --export-tfrecord
```

Output files are written to the working directory by default:

```
institutional_net_buy_<start>_<end>.csv
institutional_net_buy_<start>_<end>.tfrecord   # only with --export-tfrecord
```

## Scripts Overview

This folder contains several helper scripts for institutional flow monitoring and analysis:

- `run_all_analysis.py`: Runs the full automated workflow including fetching, trend monitoring, target extraction, and visualization.
- `institutional_net_buy_fetcher.py`: Fetches institutional flow and close price data from FinMind for the selected stocks, then exports CSV and optionally TFRecord.
- `institutional_net_buy_scanner.py`: Uses `targets.txt` as a stock list and fetches the latest institutional buy/sell data directly into a CSV file.
- `institutional_net_buy_ml.py`: Trains a dilated 1D CNN model using TFRecord data for 5-day ahead multi-step close price prediction, with per-stock chronological train/validation split and train-only min-max scaling to reduce leakage risk. `--window-size 1` builds current-day feature inputs and labels the next 5 closing prices.
- `institutional_net_buy_ml_lstm_attention.py`: Trains an Attention + LSTM model with the same 12 engineered features, using per-stock chronological train/validation split and train-only min-max scaling for leakage-safe evaluation.
- `institutional_net_buy_predict.py`: Loads a trained `.keras` model and runs predictions on TFRecord data.
- `institutional_net_buy_predict_visual.py`: Generates comparison plots (PNG) of predicted vs. actual prices for stocks listed in `targets.txt`, with stock ID normalization (`strip`, `.0` cleanup) across TFRecord/targets/stats and an execution summary (`成功輸出圖檔`, `skip *`) for easier debugging.
- `institutional_net_buy_predict_visual_all.py`: Runs the trained model across all stocks found in the TFRecord file and creates visualizations for the top-ranked predictions. It still reads `targets.txt` to mark tracked stocks, but does not limit prediction to that list.
- `institutional_net_buy_trend_monitor.py`: Non-ML analyzer that searches for stocks with increasing or sustained-high institutional net-buy trends and prints recommendations.
- `parse_stock_id.py`: Utility script to extract `stock_id` from CSV files (such as trend monitor outputs) and save them as a plain text list (e.g., `targets.txt`) while preserving formatting like leading zeros.
- `institutional_net_buy_visualizer.py`: Plots cumulative institutional net-buy and close price for stock IDs listed in `targets.txt`.

### Example Flows

1.  Run the full automated analysis workflow:
    ```powershell
    python run_all_analysis.py
    ```

2.  Fetch data for a specific stock list and create TFRecord:
    ```powershell
    python institutional_net_buy_fetcher.py `
        --stock-source list `
        --stocks 2330,2317 `
        --days 90 `
        --export-tfrecord
    ```

3.  Monitor institutional trend candidates without ML:
    ```powershell
    python institutional_net_buy_trend_monitor.py `
        --tfrecord-path institutional_net_buy_2026-02-08_2026-05-09.tfrecord `
        --recent-days 10 `
        --baseline-days 20
    ```

4.  Extract target stock IDs from the trend monitor output CSV:
    ```powershell
    python parse_stock_id.py `
        --input-csv trend_candidates_2026-02-09_2026-05-08.csv `
        --output-txt targets.txt
    ```

5.  Visualize only the stock IDs from `targets.txt`:
    ```powershell
    python institutional_net_buy_visualizer.py `
        --csv-filename institutional_net_buy_2026-02-10_2026-05-11.csv `
        --targets-filename targets.txt
    ```

6.  Train a model from TFRecord data using the leakage-fixed dilated CNN pipeline:
    ```powershell
    python institutional_net_buy_ml.py `
        --tfrecord-path institutional_net_buy_2026-02-20_2026-05-21.tfrecord `
        --epochs 50 `
        --batch-size 64 `
        --window-size 5 `
        --val-ratio 0.2
    ```

    Use `--window-size 1` to train the 5-day multi-step forecast model, where a single day of institutional features predicts the next 5 close prices. Larger window sizes will create a multi-day input sequence instead.
    
    **Label Normalization**: Both input features and target labels (close prices) are normalized per stock using train-only Min-Max scaling. This script writes `*_leakage_fixed.stats.json` with `target_min` and `target_max`, which are required for denormalizing predictions.

7.  Train using Attention + LSTM with a temporal split and train-only scaling:
    ```powershell
    python institutional_net_buy_ml_lstm_attention.py `
        --tfrecord-path institutional_net_buy_2026-02-20_2026-05-21.tfrecord `
        --epochs 50 `
        --batch-size 64 `
        --window-size 5 `
        --val-ratio 0.2
    ```

    This script uses per-stock chronological validation and denormalized validation metrics for more realistic time-series evaluation.

8.  Predict using a trained model:
    ```powershell
    python institutional_net_buy_predict.py `
        --model-path institutional_net_buy_model.keras `
        --tfrecord-path institutional_net_buy_2026-02-20_2026-05-21.tfrecord `
        --window-size 5 `
        --limit 100
    ```
9.  Visualize model predictions for target stocks:
    for the dilated CNN:
    ```powershell
    python institutional_net_buy_predict_visual.py `
        --model-path institutional_net_buy_v2_dilated.keras `
        --tfrecord-path institutional_net_buy_2026-02-20_2026-05-21.tfrecord `
        --stats-path institutional_net_buy_2026-02-20_2026-05-21.leakage_fixed.stats.json `
        --window-size 5
    ```
    for LSTM+Attention:
    ```powershell
    python institutional_net_buy_predict_visual.py `
        --model-path institutional_net_buy_v3_lstm_attention.keras `
        --tfrecord-path institutional_net_buy_2026-02-20_2026-05-21.tfrecord `
        --stats-path institutional_net_buy_2026-02-20_2026-05-21_lstmattn.stats.json `
        --window-size 5
    ```
    for all-stock visualization:
    ```powershell
    python institutional_net_buy_predict_visual_all.py `
        --model-path institutional_net_buy_v3_lstm_attention.keras `
        --tfrecord-path institutional_net_buy_2026-02-20_2026-05-21.tfrecord `
        --stats-path institutional_net_buy_2026-02-20_2026-05-21.leakage_fixed.stats.json `
        --window-size 5
    ```

    Plots are saved to the `predict_plot/` folder as `{stock_id}_{name}.png`.
    The script automatically denormalizes model predictions using `target_min` and `target_max` from the provided stats file (`--stats-path`), ensuring accurate visualization of predicted vs. actual prices in their original value ranges.
    The script prints a run summary including `成功輸出圖檔` and `skip` counters (`not_in_targets`, `not_in_stats`, `too_short`, `no_window`) for quick root-cause checks when output is empty.

## Testing

The generated validation scripts for the institutional ML workflow are placed under the repository `test/` directory.
Run them from the project root:

```powershell
python test/test_model_build.py
python test/test_dataset_pipeline.py
python test/test_integration.py
```

These scripts verify the updated model architecture, TFRecord multi-step label pipeline, and end-to-end training flow.

## CLI Reference

### Stock Selection

| Argument | Default | Description |
|---|---|---|
| `--stock-source` | `all` | Source mode: `all` / `file` / `list` |
| `--target-file` | `targets.txt` | Path to stock ID file (used with `--stock-source file`) |
| `--stocks` | `` | Comma-separated stock IDs (used with `--stock-source list`) |
| `--max-stocks` | `0` | Cap the number of stocks processed (0 = no cap) |
| `--stock-offset` | `0` | Skip the first N stocks in the list |

### Date Range

| Argument | Default | Description |
|---|---|---|
| `--end-date` | today | End date in `YYYY-MM-DD` format |
| `--days` | `90` | Number of calendar days to look back |

### Rate Limiting

| Argument | Default | Description |
|---|---|---|
| `--request-delay-seconds` | `0.5` | Base sleep between requests (seconds) |
| `--request-delay-jitter` | `0.0` | Random jitter added to delay (seconds) |
| `--api-usage-threshold` | `50` | Stop fetching when remaining quota falls below this |
| `--api-usage-threshold-unit` | `absolute` | `absolute` (count) or `percent` of total limit |
| `--api-retry-max` | `3` | Maximum retries per stock on transient error |
| `--api-throttle-enabled` | *(flag)* | Enable adaptive throttle based on remaining quota |
| `--api-throttle-min-seconds` | `1.0` | Minimum adaptive sleep (when quota is high) |
| `--api-throttle-max-seconds` | `10.0` | Maximum adaptive sleep (when quota is low) |
| `--api-throttle-mode` | `linear` | Throttle curve shape (`linear`) |

### Output

| Argument | Default | Description |
|---|---|---|
| `--csv-path` | auto-named | Override the output CSV path |
| `--export-tfrecord` | *(flag)* | Also write a `.tfrecord` file |
| `--tfrecord-path` | auto-named | Override the output TFRecord path |
| `--log-level` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## Institutional Trend Monitoring (Non-ML)

The `institutional_net_buy_trend_monitor.py` script monitors institutional flow trends directly from TFRecord data. It does not train models.

Target:
- Find stocks where cumulative institutional net-buy is increasing
- Or where net-buy remains at sustained high levels
- Include price trend context and output recommendation reasons

### Monitoring Workflow

1.  **Generate Data**: Use the fetcher to create a TFRecord file (pivoted 5-feature schema).
2.  **Run Trend Monitor**:
    ```powershell
    python institutional_net_buy_trend_monitor.py `
        --tfrecord-path "institutional_net_buy_2026-02-08_2026-05-09.tfrecord" `
        --recent-days 10 `
        --baseline-days 20 `
        --top-k 30 `
        --min-history-days 30 `
        --min-positive-ratio 0.6 `
    ```

    If `--output-csv` is omitted, the script writes the result beside the TFRecord using the default name `trend_candidates_{start_date}_{end_date}.csv`.

### Trend Monitor CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--tfrecord-path` | *(required)* | Path to the source `.tfrecord` file |
| `--recent-days` | `10` | Recent lookback window used for trend signal |
| `--baseline-days` | `20` | Baseline window used for high-level comparison |
| `--min-history-days` | `30` | Minimum history length required per stock |
| `--min-positive-ratio` | `0.6` | Minimum ratio of positive net-buy days in recent window |
| `--top-k` | `30` | Maximum number of stocks to output |
| `--output-csv` | auto-named | Optional path for CSV export; default is `trend_candidates_{start_date}_{end_date}.csv` |

### Output Interpretation

For each selected stock, the monitor prints:
- Stock name and stock ID
- Signal strength ranking
- Recent total net-buy, positive-day ratio, and recent price change
- Observed phenomenon summary
- Recommendation reason based on flow + price alignment

## TFRecord Schema (Pivoted)

Each `tf.train.Example` contains a vectorized day-slice:

```python
{
    "date":                   tf.string,   # YYYY-MM-DD
    "stock_id":               tf.string,   # e.g. "2330"
    "close":                  tf.float32,  # Target label
    "foreign_net_buy":        tf.int64,    # Feature 1
    "foreign_dealer_net_buy": tf.int64,    # Feature 2
    "trust_net_buy":          tf.int64,    # Feature 3
    "dealer_hedge_net_buy":   tf.int64,    # Feature 4
    "dealer_net_buy":         tf.int64,    # Feature 5
}
```

## Notes

- Each stock triggers **two** FinMind API calls (institutional data + daily price). Account for this when setting quota thresholds.
- Stocks with no institutional data in the requested date range are logged as `empty` and excluded from output.
- Price data unavailability (e.g. newly listed stocks) results in `close = 0.0` rather than a failed run.
