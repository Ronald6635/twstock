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
- `institutional_net_buy_ml.py`: Trains a dilated 1D CNN model using TFRecord data for 5-day ahead multi-step close price prediction, with per-stock chronological train/validation split and train-only **Z-Score standardization** to enhance resistance to outliers and distribution drift.
- `institutional_net_buy_ml_lstm_attention.py`: Trains an Attention + LSTM model with the same 11 engineered features, using per-stock chronological train/validation split and train-only **Z-Score standardization** for stationary feature inputs.
- `institutional_net_buy_predict.py`: Loads a trained `.keras` model and runs predictions on TFRecord data.
- `institutional_net_buy_predict_visual.py`: Generates comparison plots (PNG) of predicted vs. actual prices for stocks listed in `targets.txt`, with stock ID normalization (`strip`, `.0` cleanup) across TFRecord/targets/stats and an execution summary (`成功輸出圖檔`, `skip *`) for easier debugging. **Features 5-prediction consensus ensemble** with exponential/linear/recency aggregation methods, confidence scoring (0-1), and optional individual prediction scatter points.
- `institutional_net_buy_predict_visual_all.py`: Runs the trained model across all stocks found in the TFRecord file and creates visualizations for the top-ranked predictions. It still reads `targets.txt` to mark tracked stocks, but does not limit prediction to that list. **Includes identical consensus ensemble features** for batch processing with confidence-based ranking.
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
        --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
        --epochs 100 `
        --batch-size 256 `
        --window-size 32 `
        --val-ratio 0.2 `
        --loss mse `
        --target-transform tanh `
        --tanh-scale 50 `
        --no-cache

    or
        --target-transform none `
        --loss quantile `
        --quantile 0.75
    ```
    for V3 edition:
    ```powershell
    python institutional_net_buy_ml_v3.py `
        --tfrecord-path .\institutional_net_buy_2026-02-21_2026-05-22.tfrecord `
        --epochs 100 `
        --batch-size 256 `
        --window-size 10 `
        --loss huber
    ```
    Use `--window-size 1` to train the 5-day multi-step forecast model, where a single day of institutional features predicts the next 5 close prices. Larger window sizes will create a multi-day input sequence instead.
    
    **Return Target Transform**: Input features are scaled per stock using train-only **Z-Score standardization**. Target labels are modeled as future log-returns and support configurable transforms via `--target-transform` (`clip` / `tanh` / `none`).
    The script writes `*.leakage_fixed.stats.json` including per-stock target-transform metadata and Z-Score stats (`mean`, `std`) used by visualization scripts to decode model outputs.

7.  Train using Attention + LSTM with a temporal split and train-only Z-Score scaling:
    ```powershell
    python institutional_net_buy_ml_lstm_attention.py `
        --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
        --epochs 50 `
        --batch-size 256 `
        --window-size 20 `
        --val-ratio 0.2
    ```

    This script uses per-stock chronological validation and denormalized validation metrics for more realistic time-series evaluation.

8.  Predict using a trained model:
    ```powershell
    python institutional_net_buy_predict.py `
        --model-path institutional_net_buy_model.keras `
        --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
        --window-size 20 `
        --limit 100
    ```
9.  Visualize model predictions for target stocks with consensus ensemble:
    for the dilated CNN with exponential ensemble (default):
    ```powershell
    python institutional_net_buy_predict_visual.py `
        --model-path institutional_net_buy_v2_dilated.keras `
        --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
        --stats-path institutional_net_buy_2024-05-22_2026-05-22.leakage_fixed.stats.json `
        --window-size 32 `
        --ensemble-method exponential `
        --ensemble-future true
    ```
    for LSTM+Attention with linear aggregation:
    ```powershell
    python institutional_net_buy_predict_visual.py `
        --model-path institutional_net_buy_v3_lstm_attention.keras `
        --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
        --stats-path institutional_net_buy_2024-05-22_2026-05-22_lstmattn.stats.json `
        --window-size 20 `
        --ensemble-method linear `
        --show-individual-preds true
    ```
    for all-stock visualization with confidence filtering:
    ```powershell
    python institutional_net_buy_predict_visual_all.py `
        --model-path institutional_net_buy_v3_lstm_attention.keras `
        --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
        --stats-path institutional_net_buy_2024-05-22_2026-05-22.leakage_fixed.stats.json `
        --window-size 20 `
        --top-n 50 `
        --ensemble-method exponential `
        --confidence-threshold 0.5
    ```

    for CNN V3 edition:
    ```powershell
    python institutional_net_buy_predict_visual_all_v3.py `
        --model-path "model_w10_p5_dim12.keras" `
        --tfrecord-path institutional_net_buy_2026-02-21_2026-05-22.tfrecord `
        --scaler-path ".cache/scaler_dim12.npz" `
        --top-n 20
    ```

    Plots are saved to the `predict_plot/` folder as `{stock_id}_{name}.png`.
    The script automatically decodes model outputs using target-transform metadata from the provided stats file (`--stats-path`), ensuring consistent interpretation when training with `clip`/`tanh` target transforms.
    **Consensus Ensemble** aggregates the 5 daily predictions (D+1 through D+5) generated by sliding windows:
    - **Exponential weights** [16,8,4,2,1] emphasize the most recent prediction
    - **Linear weights** [5,4,3,2,1] provide gradual decay
    - **Recency** uses only the most recent (D+1) prediction
    - **Confidence score** (0-1) quantifies signal strength: `1/(1+std)` where std is the standard deviation of the 5 predictions
    - **Future ensemble** applies the same aggregation to future 5-day predictions
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

## Consensus Ensemble Parameters

### 5-Prediction Aggregation (Visualization Scripts)

Both `institutional_net_buy_predict_visual.py` and `institutional_net_buy_predict_visual_all.py` support consensus ensemble mechanism to aggregate the 5 daily predictions generated by sliding windows:

| Argument | Default | Description |
|---|---|---|
| `--ensemble-method` | `exponential` | Aggregation strategy: `exponential` (weights [16,8,4,2,1]), `linear` (weights [5,4,3,2,1]), `recency` (D+1 only), or `none` (simple average) |
| `--ensemble-future` | `true` | Apply consensus ensemble to future 5-day predictions |
| `--show-individual-preds` | `false` | Display individual D+1-D+5 predictions as scatter points on visualization |
| `--confidence-threshold` | `0.0` | Minimum confidence score (0-1) to display bars; higher values filter low-confidence predictions |

### Confidence Scoring

The confidence score is computed as: `1 / (1 + std(predictions))`, where std is the standard deviation of the 5 daily predictions.
- **Range**: 0 to 1 (1 = perfect consensus, 0 = high disagreement)
- **Visualization**: Deep green bars (≥0.7), light green bars (0.5-0.7), gray bars (<threshold)

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

## 機器學習模型訓練 (ML Training)

本模組支援 Dilated CNN 與 LSTM+Attention 兩種架構，設計用於捕捉法人籌碼與價格間的長短期依賴關係。

### 特徵處理 (Feature Engineering)
- **歸一化策略**：全面採用 **Z-Score Standardization**。模型輸入為特徵偏離歷史平均的標準差倍數，能有效識別「異常量能」。
- **定常化處理**：價格斜率 (Slope) 已除以股價，轉化為每日漲跌百分比，確保高低價股具備相同尺標。
- **目標值轉換**：預設使用 `tanh` 轉換將報酬率壓縮至舒適區間，減少極端離群值對 Loss 的干擾。

### 訓練範例
```powershell
# 訓練 CNN 模型 (預設參數)
python institutional_net_buy_ml.py --window-size 32 --target-transform tanh

# 訓練 LSTM+Attention 模型
python institutional_net_buy_ml_lstm_attention.py --epochs 50 --batch-size 128
