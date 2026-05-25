# Institutional Net Buy Fetcher

Batch-fetches institutional investor (法人) buy/sell data from the [FinMind](https://finmindtrade.com/) API for a set of TWSE stocks, normalizes the output, and exports it to CSV and optionally TFRecord format.

## Features

- **Three stock-source modes**: `all`, `file`, or `list`
- **Quota-aware adaptive throttle**: monitors remaining API requests and adjusts sleep time dynamically
- **Retry with exponential backoff**: recovers from transient API errors automatically
- **Daily close-price enrichment**: fetches and merges `close` price alongside institutional flow data
- **Optional TFRecord export**: for downstream ML and visualization pipelines

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

- Python 3.10+
- Virtual environment activated (recommended)
- `FINMIND_API_KEY` set in a `.env` file at the project root or as an environment variable
- `docs/tw_stock_info.json` present (required for `stock-source=all` and name mapping in all modes)

## Run the Fetcher

```powershell
python institutional_net_buy_fetcher.py [options]
```

### Common fetcher examples

Fetch the last 7 days for two stocks:

```powershell
python institutional_net_buy_fetcher.py `
  --stock-source list `
  --stocks 2330,2317 `
  --days 7
```

Fetch the last 90 days for all TWSE stocks and export TFRecord:

```powershell
python institutional_net_buy_fetcher.py `
  --stock-source all `
  --days 90 `
  --api-throttle-enabled `
  --export-tfrecord
```

Fetch stocks from a text file (`targets.txt`):

```powershell
python institutional_net_buy_fetcher.py `
  --stock-source file `
  --target-file targets.txt `
  --days 30 `
  --export-tfrecord
```

### Key fetcher arguments

- `--stock-source {all,file,list}` (default: `all`)
- `--target-file <path>` (for `file` mode)
- `--stocks <csv>` (comma-separated IDs for `list` mode)
- `--end-date <YYYY-MM-DD>` (default: today)
- `--days <int>` (default: 90)
- `--csv-path <path>` (custom CSV output path)
- `--export-tfrecord` (create TFRecord output)
- `--tfrecord-path <path>` (custom TFRecord output path)
- `--api-throttle-enabled` (enable adaptive quota-based delay)
- `--api-usage-threshold <int>` (default: 50)
- `--api-usage-threshold-unit {absolute,percent}` (default: `absolute`)
- `--api-retry-max <int>` (default: 3)
- `--request-delay-seconds <float>` (default: 0.5)
- `--request-delay-jitter <float>` (default: 0.0)
- `--api-throttle-min-seconds <float>` (default: 1.0)
- `--api-throttle-max-seconds <float>` (default: 10.0)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` (default: `INFO`)

## Additional Scripts

### run_all_analysis.py

A convenience orchestrator that runs:

1. `institutional_net_buy_fetcher.py`
2. `institutional_net_buy_trend_monitor.py`
3. `parse_stock_id.py`
4. `institutional_net_buy_visualizer.py`

Use it when you want a single end-to-end workflow.

```powershell
python run_all_analysis.py
```

### institutional_net_buy_scanner.py

Fetches institutional data for stocks listed in `targets.txt` and writes a CSV.

```powershell
python institutional_net_buy_scanner.py
```

### parse_stock_id.py

Extracts `stock_id` values from a CSV and writes them to a plain text list.

```powershell
python parse_stock_id.py `
  --input-csv trend_candidates_2026-02-09_2026-05-08.csv `
  --output-txt targets.txt
```

### institutional_net_buy_visualizer.py

Plots cumulative institutional net-buy and close price for stock IDs in `targets.txt`.

```powershell
python institutional_net_buy_visualizer.py `
  --csv-filename institutional_net_buy_2026-02-10_2026-05-11.csv `
  --targets-filename targets.txt
```

### institutional_net_buy_trend_monitor.py

Detects stocks with increasing or sustained institutional net-buy trends.

```powershell
python institutional_net_buy_trend_monitor.py `
  --recent-days 10 `
  --baseline-days 20
```

Arguments:

- `--tfrecord-path <path>` (required)
- `--recent-days <int>` (default: 10)
- `--baseline-days <int>` (default: 20)
- `--min-history-days <int>` (default: 40)
- `--min-positive-ratio <float>` (default: 0.6)
- `--top-k <int>` (default: 20)

### institutional_net_buy_trend_monitor_v3.py

Same as the non-V3 trend monitor, with an explicit `--output-csv` option.

### institutional_net_buy_ml.py

Train the dilated CNN model on TFRecord data.

```powershell
python institutional_net_buy_ml.py `
  --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
  --epochs 100 `
  --batch-size 256 `
  --window-size 32 `
  --loss mse `
  --target-transform tanh `
  --tanh-scale 50 `
  --no-cache
```

Key options:

- `--tfrecord-path <path>` (required)
- `--epochs <int>` (default: 50)
- `--batch-size <int>` (default: 512)
- `--window-size <int>` (default: 10)
- `--val-ratio <float>` (default: 0.2)
- `--loss {mse,huber,quantile}` (default: `mse`)
- `--quantile <float>` (default: 0.75)
- `--target-transform {clip,tanh,none}` (default: `tanh`)
- `--clip-lower <float>` (default: -50.0)
- `--clip-upper <float>` (default: 70.0)
- `--tanh-scale <float>` (default: 50.0)
- `--no-cache`

### institutional_net_buy_ml_lstm_attention.py

Train the Attention + LSTM model.

```powershell
python institutional_net_buy_ml_lstm_attention.py `
  --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
  --epochs 50 `
  --batch-size 256 `
  --window-size 20 `
  --val-ratio 0.2
```

Key options:

- `--tfrecord-path <path>` (required)
- `--epochs <int>` (default: 50)
- `--batch-size <int>` (default: 256)
- `--window-size <int>` (default: 20)
- `--val-ratio <float>` (default: 0.2)
- `--lstm-units <int>` (default: 64)
- `--attention-heads <int>` (default: 4)
- `--dropout <float>` (default: 0.2)
- `--loss {mse,huber,quantile}` (default: `huber`)
- `--quantile <float>` (default: 0.75)
- `--metrics-every <int>` (default: 3)
- `--metrics-val-fraction <float>` (default: 0.3)
- `--no-cache`

### institutional_net_buy_predict.py

Run predictions using a trained `.keras` model.

```powershell
python institutional_net_buy_predict.py `
  --model-path institutional_net_buy_model.keras `
  --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
  --window-size 20 `
  --limit 100
```

Key options:

- `--model-path <path>` (default: `institutional_net_buy_model.keras`)
- `--tfrecord-path <path>` (required)
- `--window-size <int>` (default: 1)
- `--limit <int>` (default: 10)
- `--stats-path <path>` (default: `institutional_net_buy_model.stats.json`)

### institutional_net_buy_predict_visual.py

Visualize predictions for target stocks.

```powershell
python institutional_net_buy_predict_visual.py `
  --model-path institutional_net_buy_v2_dilated.keras `
  --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
  --stats-path institutional_net_buy_2024-05-22_2026-05-22.leakage_fixed.stats.json `
  --window-size 32 `
  --ensemble-method exponential `
  --confidence-threshold 0.5
```

- `--model-path <path>` (default: `institutional_net_buy_v2_dilated.keras`)
- `--tfrecord-path <path>` (required)
- `--stats-path <path>`
- `--window-size <int>` (default: 10)
- `--vol-factor <float>` (default: 1.0)
- `--ensemble-method {exponential,linear,recency,none}` (default: `exponential`)
- `--ensemble-future <bool>` (default: `True`)
- `--show-individual-preds <bool>` (default: `False`)
- `--confidence-threshold <float>` (default: 0.0)

### institutional_net_buy_predict_visual_all.py

Visualize predictions for all stocks found in TFRecord.

```powershell
python institutional_net_buy_predict_visual_all.py `
  --model-path institutional_net_buy_v2_dilated.keras `
  --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
  --stats-path institutional_net_buy_2024-05-22_2026-05-22.leakage_fixed.stats.json `
  --window-size 20 `
  --top-n 50
```

Same options as `institutional_net_buy_predict_visual.py`, plus:

- `--top-n <int>` (default: 20)

### V3 visualization scripts

`institutional_net_buy_predict_visual_v3.py` and `institutional_net_buy_predict_visual_all_v3.py` use the same visualization arguments, but they require `--scaler-path <path>` instead of `--stats-path`.

## Notes

- `--export-tfrecord` is recommended when you want to feed the output into trend monitoring, ML training, or visualization scripts.
- `stock-source=file` requires `--target-file <path>`.
- `stock-source=list` requires `--stocks`.
- `run_all_analysis.py` is the easiest way to execute a full fetch → trend monitor → target extraction → visualization flow.

## Output Files

- `institutional_net_buy_<start>_<end>.csv`
- `institutional_net_buy_<start>_<end>.tfrecord` (when `--export-tfrecord` is enabled)
- `trend_candidates_<start>_<end>.csv`
- `targets.txt`
