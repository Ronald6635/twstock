"""
Automated Stock Analysis Robot

This script automates the process of fetching Taiwan stock data, preprocessing it,
computing technical indicators, and generating professional trading analysis reports.

It integrates batch_fetch.py, preprocessing.py, and indicators.py to produce
actionable insights for specified stocks.

Usage:
    python automated_stock_robot.py

Requires:
    - config.json in the same directory
    - FINMIND_API_KEY environment variable
    - Access to scripts/batch_fetch.py, engine/datasets/preprocessing.py, engine/datasets/indicators.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import logging
import twstock
import glob
import requests
# Optionally load .env for FINMIND_API_KEY
try:
    from dotenv import load_dotenv
    _dotenv_loaded = False
    _env_path = Path(__file__).parent.parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
        _dotenv_loaded = True
except Exception:
    _dotenv_loaded = False
    # python-dotenv not installed; we'll continue but warn later if no env found

def setup_logging(log_dir):
    """Setup logging to file and console."""
    log_file = os.path.join(log_dir, f"robot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_stocks(config):
    """Load stock list from config or tw_stock_info.json."""
    if config['stocks'] == "all":
        stock_info_path = Path(__file__).parent.parent.parent / 'docs' / 'tw_stock_info.json'
        with open(stock_info_path, 'r', encoding='utf-8') as f:
            stock_data = json.load(f)
        
        # Filter for TWSE stocks with stock_id, excluding indices
        twse_stocks = [
            item['stock_id'] for item in stock_data 
            if item.get('type') == 'twse' and 'stock_id' in item and item.get('industry_category') != 'Index'
        ]
        return list(set(twse_stocks))  # Remove duplicates
    else:
        return config['stocks']

def update_settings_json(stock_id, config):
    """Update scripts/settings.json for batch_fetch.py."""
    settings = {
        "start_date": config['start_date'],
        "end_date": config['end_date'],
        "stocks": [stock_id]
    }
    settings_path = Path(__file__).parent.parent.parent / 'scripts' / 'settings.json'
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)

def run_batch_fetch(stock_id, config, temp_dir):
    """Run batch_fetch.py for a specific stock."""
    try:
        update_settings_json(stock_id, config)
        cmd = [
            sys.executable, 'scripts/batch_fetch.py'
        ]
        env = os.environ.copy()
        # Determine token source and validate token before running batch_fetch
        api_key_config = config.get('api_key')
        env_key = os.environ.get('FINMIND_API_KEY')
        token = None
        token_source = None

        if api_key_config and api_key_config != 'FINMIND_API_KEY':
            token = api_key_config
            token_source = 'config'
        elif env_key:
            token = env_key
            token_source = 'environment'
        else:
            token = None
            token_source = None

        # If dotenv was present and loaded, re-check environment
        if not token and 'FINMIND_API_KEY' in os.environ:
            token = os.environ.get('FINMIND_API_KEY')
            token_source = '.env'

        if token:
            # Validate token by calling user_info endpoint
            try:
                headers = {"Authorization": f"Bearer {token}"}
                resp = requests.get("https://api.web.finmindtrade.com/v2/user_info", headers=headers, timeout=10)
                if resp.ok:
                    data = resp.json() if resp.headers.get('content-type','').startswith('application/json') else {}
                    user_count = data.get('user_count') if isinstance(data, dict) else None
                    if user_count is not None:
                        logging.info(f"Valid FINMIND_API_KEY found (source: {token_source})")
                        env['FINMIND_API_KEY'] = token
                    else:
                        logging.warning(f"FINMIND_API_KEY present (source: {token_source}) but validation returned unexpected response")
                else:
                    logging.warning(f"FINMIND_API_KEY present (source: {token_source}) but user_info returned {resp.status_code}")
            except Exception as e:
                logging.warning(f"Failed to validate FINMIND_API_KEY (source: {token_source}): {e}")
        else:
            logging.warning('No FINMIND_API_KEY found in config, environment, or .env; batch_fetch may use cache-only')

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, env=env, capture_output=True, text=True)
        # Save fetch stdout/stderr into temp log for diagnosis
        try:
            temp_log_dir = Path(temp_dir)
            temp_log_dir.mkdir(parents=True, exist_ok=True)
            fetch_log = temp_log_dir / f"batch_fetch_{stock_id}.log"
            with open(fetch_log, 'w', encoding='utf-8') as lf:
                lf.write('--- STDOUT ---\n')
                lf.write(result.stdout or '')
                lf.write('\n--- STDERR ---\n')
                lf.write(result.stderr or '')
            logging.info(f"Saved batch_fetch logs to {fetch_log}")
        except Exception as e:
            logging.warning(f"Unable to write batch_fetch log for {stock_id}: {e}")

        if result.returncode != 0:
            logging.error(f"Batch fetch failed for {stock_id}: see {fetch_log} for details")
            return False
        logging.info(f"Batch fetch completed for {stock_id}")
        return True
    except Exception as e:
        logging.error(f"Error running batch fetch for {stock_id}: {e}")
        return False

def run_preprocessing(stock_id, config, data_dir, temp_dir):
    """Run preprocessing.py for a specific stock."""
    try:
        company = twstock.codes.get(stock_id).name if stock_id in twstock.codes else stock_id
        datasets_dir = Path(__file__).parent.parent / "datasets" / f"{company}-{stock_id}"
        existing_preprocessed = datasets_dir / f"preprocessed_{company}-{stock_id}.json"
        
        output_json = os.path.join(data_dir, f"preprocessed_{stock_id}.json")
        output_csv = os.path.join(data_dir, f"preprocessed_{stock_id}.csv")
        
        # Priority 1: Use existing preprocessed file from datasets
        if existing_preprocessed.exists():
            logging.info(f"Using existing preprocessed file: {existing_preprocessed}")
            # Copy to robot data dir
            import shutil
            shutil.copy2(existing_preprocessed, output_json)
            # Also copy CSV if exists
            existing_csv = datasets_dir / f"preprocessed_{company}-{stock_id}.csv"
            if existing_csv.exists():
                shutil.copy2(existing_csv, output_csv)
            else:
                # Create CSV from JSON
                with open(output_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                import pandas as pd
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame(data)
                df.to_csv(output_csv, index=False)
            logging.info(f"Preprocessed data copied for {stock_id}")
            return True
        
        # Priority 2: Use combined files from datasets or cache
        combined_dirs = [
            datasets_dir,  # engine/datasets/
            Path(__file__).parent.parent.parent / "cache" / f"{company}-{stock_id}"  # cache/
        ]
        combined_file = None
        for combined_dir in combined_dirs:
            if combined_dir.exists():
                combined_pattern = str(combined_dir / "combined_*.json")
                combined_files = glob.glob(combined_pattern)
                if combined_files:
                    combined_file = combined_files[0]
                    break
        
        # If we found a combined file, try to preprocess it
        if combined_file:
            cmd = [
                sys.executable, 'engine/datasets/preprocessing.py',
                combined_file,
                '--out-json', output_json,
                '--out-csv', output_csv
            ]
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Preprocessing completed for {stock_id}")
                return True
            else:
                logging.warning(f"Preprocessing failed for {stock_id}, trying fallback: {result.stderr}")
                # Attempt to salvage price records directly from the combined JSON
                try:
                    import json as _json
                    with open(combined_file, 'r', encoding='utf-8') as _cf:
                        _combined = _json.load(_cf)
                    _price_ds = _combined.get('cache', {}).get('finmind_taiwan_stock_price', {})
                    _price_records = _price_ds.get('records') if isinstance(_price_ds, dict) else None
                    if _price_records:
                        logging.info(f"Extracted {len(_price_records)} price records from combined file for {stock_id}, creating preprocessed file")
                        preprocessed_data = {'data': _price_records}
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
                        import pandas as _pd
                        _pd.DataFrame(_price_records).to_csv(output_csv, index=False)
                        logging.info(f"Preprocessed data created for {stock_id} from combined file price records")
                        return True
                    else:
                        logging.info(f"No price records found inside combined file for {stock_id}")
                except Exception as _e:
                    logging.warning(f"Failed to extract price records from combined for {stock_id}: {_e}")

        # If no combined or preprocessing failed, attempt to fetch data via batch_fetch and re-check
        logging.info(f"No usable combined file found for {stock_id}, attempting to fetch data via batch_fetch")
        try:
            fetched = run_batch_fetch(stock_id, config, temp_dir)
        except Exception as e:
            fetched = False
            logging.warning(f"run_batch_fetch raised exception for {stock_id}: {e}")

        if fetched:
            # Re-scan combined_dirs for a new combined file
            for combined_dir in combined_dirs:
                if combined_dir.exists():
                    combined_files = glob.glob(str(combined_dir / "combined_*.json"))
                    if combined_files:
                        combined_file = combined_files[0]
                        break
            if combined_file:
                cmd = [
                    sys.executable, 'engine/datasets/preprocessing.py',
                    combined_file,
                    '--out-json', output_json,
                    '--out-csv', output_csv
                ]
                result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)
                if result.returncode == 0:
                    logging.info(f"Preprocessing completed for {stock_id} after fetching")
                    return True
                else:
                    logging.warning(f"Preprocessing still failed after fetching for {stock_id}: {result.stderr}")

        # Fallback: look for price file in cache (broader search)
        price_filename = "2020-01-01_2026-01-28_finmind_taiwan_stock_price.json"
        cache_candidates = [
            Path(__file__).parent.parent.parent / "cache" / f"{company}-{stock_id}",
            Path(__file__).parent.parent.parent / "cache" / stock_id,
            Path(__file__).parent.parent / "datasets" / f"{company}-{stock_id}"
        ]
        price_file = None
        for cdir in cache_candidates:
            if cdir.exists():
                logging.debug(f"Checking directory for price file: {cdir}")
                candidate = cdir / price_filename
                if candidate.exists():
                    price_file = candidate
                    break
                # also accept generic price files in directory
                for p in cdir.glob(f"*_finmind_taiwan_stock_price.json"):
                    price_file = p
                    break
            if price_file:
                break

        if price_file:
            logging.info(f"Price file found: {price_file}")
            with open(price_file, 'r', encoding='utf-8') as f:
                price_data = json.load(f)
            # Create preprocessed directly from price_data
            preprocessed_data = {'data': price_data}
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)
            # Create CSV
            import pandas as pd
            df = pd.DataFrame(price_data)
            df.to_csv(output_csv, index=False)
            logging.info(f"Preprocessed data created for {stock_id} from price data")
            return True
        else:
            logging.warning(f"No data found for {stock_id} after attempting fetch and fallbacks")
            return False
    except Exception as e:
        logging.error(f"Error running preprocessing for {stock_id}: {e}")
        return False

def run_indicators(stock_id, config, data_dir, reports_dir):
    """Run indicators.py for a specific stock."""
    try:
        preprocessed_file = os.path.join(data_dir, f"preprocessed_{stock_id}.json")
        if not os.path.exists(preprocessed_file):
            logging.error(f"Preprocessed file not found: {preprocessed_file}")
            return False
        cmd = [
            sys.executable, 'engine/datasets/indicators.py',
            preprocessed_file,
            '--start_date', config['start_date']
        ]
        # Redirect output to report file
        report_file = os.path.join(reports_dir, f"analysis_{stock_id}.txt")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # Control interactive plotting via config
        interactive = bool(config.get('interactive_plots', False))
        if not interactive:
            # Prevent plotly from opening a browser and matplotlib from using a GUI backend
            env['PLOTLY_RENDERER'] = 'svg'
            env['MPLBACKEND'] = 'Agg'
            env['ROBOT_NO_PLOT'] = '1'
        with open(report_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, stdout=f, stderr=subprocess.PIPE, text=True, env=env)
        if result.returncode != 0:
            logging.error(f"Indicators failed for {stock_id}: {result.stderr}")
            return False
        logging.info(f"Indicators completed for {stock_id}, report saved to {report_file}")
        return True
    except Exception as e:
        logging.error(f"Error running indicators for {stock_id}: {e}")
        return False

def generate_summary(stocks, summary_dir, reports_dir):
    """Generate a summary of all reports."""
    try:
        summary_file = os.path.join(summary_dir, f"summary_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Stock Analysis Summary\n")
            f.write("="*50 + "\n")
            for stock_id in stocks:
                report_file = os.path.join(reports_dir, f"analysis_{stock_id}.txt")
                if os.path.exists(report_file):
                    f.write(f"\nStock: {stock_id}\n")
                    with open(report_file, 'r', encoding='utf-8') as rf:
                        f.write(rf.read())
                    f.write("\n" + "-"*30 + "\n")
                else:
                    f.write(f"\nStock: {stock_id} - No report available\n")
        logging.info(f"Summary generated: {summary_file}")
    except Exception as e:
        logging.error(f"Error generating summary: {e}")

def main():
    """Main function to run the automated stock robot."""
    robot_dir = Path(__file__).parent
    config_path = robot_dir / 'config.json'
    config = load_config(config_path)

    # Load stock list
    stocks = load_stocks(config)
    logging.info(f"Loaded {len(stocks)} stocks to process")

    # Setup directories
    data_dir = robot_dir / config['output_dirs']['data']
    reports_dir = robot_dir / config['output_dirs']['reports']
    logs_dir = robot_dir / config['output_dirs']['logs']
    summary_dir = robot_dir / config['output_dirs']['summary']
    temp_dir = robot_dir / config['output_dirs']['temp']

    # Ensure directories exist
    for d in [data_dir, reports_dir, logs_dir, summary_dir, temp_dir]:
        d.mkdir(exist_ok=True)

    # Setup logging
    log_file = setup_logging(logs_dir)
    logging.info("Automated Stock Robot started")

    # Process each stock with progress and company name printed to terminal
    total = len(stocks)
    for idx, stock_id in enumerate(stocks, start=1):
        company = twstock.codes.get(stock_id).name if stock_id in twstock.codes else stock_id
        msg = f"[{idx}/{total}] Processing: {company} ({stock_id})"
        print(msg)
        logging.info(msg)
        if run_batch_fetch(stock_id, config, temp_dir):
            if run_preprocessing(stock_id, config, data_dir, temp_dir):
                run_indicators(stock_id, config, data_dir, reports_dir)
            else:
                logging.warning(f"Skipping indicators for {stock_id} due to no data available")
        else:
            logging.warning(f"Skipping preprocessing and indicators for {stock_id} due to fetch failure")

    # Generate summary
    generate_summary(stocks, summary_dir, reports_dir)

    logging.info("Automated Stock Robot completed")

if __name__ == "__main__":
    main()