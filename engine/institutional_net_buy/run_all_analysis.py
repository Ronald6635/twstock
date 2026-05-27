"""
Automates the entire workflow for institutional net buy analysis, from data fetching to trend monitoring and visualization.
This script sequentially executes the following steps:
1. Fetch institutional net buy data for a specified date range and export to CSV and TFRecord.
2. Analyze trends in the institutional net buy data to identify candidate stocks.   
3. Extract target stock IDs from the trend analysis results.
4. Visualize cumulative net buy trends for the identified target stocks.
Each step is executed as a separate subprocess, allowing for modularity and easier debugging.
Usage:
    python run_all_analysis.py
    Note: Ensure that all required scripts (fetcher, trend monitor, parser, visualizer) are present in the same directory as this script.
"""
import subprocess
import datetime
from pathlib import Path
import os
import sys

# --- Configuration ---
# Root directory of the entire project, relative to this script's location.
# Assumes this script is in `twstock/engine/institutional_net_buy/`
ROOT_DIR: Path = Path(__file__).parent.parent.parent
SCRIPTS_DIR: Path = Path(__file__).parent
DEFAULT_STOCK_INFO_PATH: Path = ROOT_DIR / "docs" / "tw_stock_info.json"
DEFAULT_TARGET_FILE: Path = SCRIPTS_DIR / "targets.txt"

# Parameters for institutional_net_buy_fetcher.py
# Set END_DATE to today's date for dynamic execution.
# END_DATE = datetime.date.today().strftime("%Y-%m-%d")
END_DATE = "2026-05-25" # Static end date for consistent testing
# DAYS_LOOKBACK = 730 # Number of calendar days to fetch data for (2 years for robust ML training)
DAYS_LOOKBACK = 90 # Shorter lookback for faster testing and visualization
STOCK_SOURCE = "all" # Choices: "all", "file", "list"
STOCKS_LIST = "2330,2317" # Comma-separated stock IDs (only if STOCK_SOURCE is "list")
API_THROTTLE_ENABLED = True # Enable adaptive API throttling
EXPORT_TFRECORD = True # Export data to TFRecord format for ML/trend analysis

# Parameters for institutional_net_buy_trend_monitor.py
RECENT_DAYS = 10 # Recent lookback window for trend signal
BASELINE_DAYS = 20 # Baseline window for high-level comparison
MIN_HISTORY_DAYS = 30 # Minimum history length required per stock for analysis
MIN_POSITIVE_RATIO = 0.6 # Minimum ratio of positive net-buy days in recent window
TOP_K_CANDIDATES = 50 # Maximum number of top candidate stocks to output

# --- Derived Paths and Dates (DO NOT MODIFY MANUALLY) ---
start_date_obj = datetime.datetime.strptime(END_DATE, "%Y-%m-%d") - datetime.timedelta(days=DAYS_LOOKBACK)
START_DATE = start_date_obj.strftime("%Y-%m-%d")

FETCH_CSV_FILENAME = f"institutional_net_buy_{START_DATE}_{END_DATE}.csv"
FETCH_TFRECORD_FILENAME = f"institutional_net_buy_{START_DATE}_{END_DATE}.tfrecord"
TREND_CANDIDATES_CSV_FILENAME = f"trend_candidates_{START_DATE}_{END_DATE}.csv"

FETCH_CSV_PATH = SCRIPTS_DIR / FETCH_CSV_FILENAME
FETCH_TFRECORD_PATH = SCRIPTS_DIR / FETCH_TFRECORD_FILENAME
TREND_CANDIDATES_CSV_PATH = SCRIPTS_DIR / TREND_CANDIDATES_CSV_FILENAME


def run_command(script_name: str, args: list[str]):
    """Helper function to execute a Python script using subprocess."""
    full_command = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    
    print(f"\n--- Running: {script_name} ---")
    print(f"Command: {' '.join(full_command)}")
    
    process = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        check=False,
        encoding='cp950'  # Changed from 'utf-8' to 'cp950' for Traditional Chinese Windows
    )
    
    if process.stdout:
        print("STDOUT:\n", process.stdout)
    if process.stderr:
        print("STDERR:\n", process.stderr)
    
    if process.returncode != 0:
        print(f"ERROR: Script '{script_name}' failed with exit code {process.returncode}")
        sys.exit(1) # Exit the workflow if any script fails
    print(f"--- '{script_name}' completed successfully ---")

def main():
    print("==================================================")
    print("  Starting Automated Institutional Stock Analysis ")
    print("==================================================")
    print(f"Analysis Period: {START_DATE} to {END_DATE} ({DAYS_LOOKBACK} days)")
    print(f"Output Directory: {SCRIPTS_DIR}\n")

    # --- Step 1: Fetch Institutional Net Buy Data ---
    fetcher_args = [
        "--stock-source", STOCK_SOURCE,
        "--end-date", END_DATE,
        "--days", str(DAYS_LOOKBACK),
        "--csv-path", str(FETCH_CSV_PATH),
    ]
    if STOCK_SOURCE == "list":
        fetcher_args.extend(["--stocks", STOCKS_LIST])
    if EXPORT_TFRECORD:
        fetcher_args.extend(["--export-tfrecord", "--tfrecord-path", str(FETCH_TFRECORD_PATH)])
    if API_THROTTLE_ENABLED:
        fetcher_args.append("--api-throttle-enabled")
    
    run_command("institutional_net_buy_fetcher.py", fetcher_args)

    if EXPORT_TFRECORD and not FETCH_TFRECORD_PATH.exists():
        print(f"ERROR: TFRecord file '{FETCH_TFRECORD_PATH}' was expected but not created by fetcher.")
        sys.exit(1)
    if not FETCH_CSV_PATH.exists():
        print(f"ERROR: CSV file '{FETCH_CSV_PATH}' was expected but not created by fetcher.")
        sys.exit(1)

    # --- Step 2: Monitor Institutional Trends ---
    if not EXPORT_TFRECORD:
        print("Skipping institutional_net_buy_trend_monitor.py because TFRecord export was disabled.")
        # If TFRecord is essential for the next step, the workflow should stop or be configured differently.
        # For this example, we assume TFRecord is required for the trend monitor.
        sys.exit(1) # Force exit if TFRecord is not available but needed.

    trend_monitor_args = [
        "--tfrecord-path", str(FETCH_TFRECORD_PATH),
        "--recent-days", str(RECENT_DAYS),
        "--baseline-days", str(BASELINE_DAYS),
        "--min-history-days", str(MIN_HISTORY_DAYS),
        "--min-positive-ratio", str(MIN_POSITIVE_RATIO),
        "--top-k", str(TOP_K_CANDIDATES),
        "--output-csv", str(TREND_CANDIDATES_CSV_PATH),
    ]
    run_command("institutional_net_buy_trend_monitor.py", trend_monitor_args)

    if not TREND_CANDIDATES_CSV_PATH.exists():
        print(f"ERROR: Trend candidates CSV file '{TREND_CANDIDATES_CSV_PATH}' was expected but not created by monitor.")
        sys.exit(1)

    # --- Step 3: Extract Target Stock IDs ---
    parse_stock_id_args = [
        "--input-csv", str(TREND_CANDIDATES_CSV_PATH),
        "--output-txt", str(DEFAULT_TARGET_FILE),
    ]
    run_command("parse_stock_id.py", parse_stock_id_args)

    if not DEFAULT_TARGET_FILE.exists():
        print(f"ERROR: Target stocks file '{DEFAULT_TARGET_FILE}' was expected but not created by parser.")
        sys.exit(1)

    # --- Step 4: Visualize Institutional Net Buy Trends ---
    print("\n==================================================")
    print("  Starting Visualization Step                     ")
    print("==================================================")
    
    visualizer_args = [
        "--csv-filename", str(FETCH_CSV_PATH),
        "--targets-filename", str(DEFAULT_TARGET_FILE),
    ]
    run_command("institutional_net_buy_visualizer.py", visualizer_args)

    print("\n--- Automated Workflow Completed ---")

if __name__ == "__main__":
    main()