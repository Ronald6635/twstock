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


# Helper: fetch and normalize FinMind /v2/user_info response
def fetch_finmind_usage(token: str, timeout: int = 10) -> dict:
    """Return a normalized dict for /v2/user_info.

    Returns keys: ok (bool), status_code (int or None), data (dict),
    error (str when exception), text (raw text, truncated).
    """
    url = "https://api.web.finmindtrade.com/v2/user_info"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        data = {}
        if resp.ok:
            ct = resp.headers.get("content-type", "").lower()
            if ct.startswith("application/json"):
                try:
                    data = resp.json()
                except Exception:
                    data = {}
        return {
            "ok": bool(resp.ok),
            "status_code": resp.status_code,
            "data": data if isinstance(data, dict) else {},
            "text": (resp.text[:200] if hasattr(resp, "text") else ""),
        }
    except Exception as e:
        return {"ok": False, "status_code": None, "data": {}, "error": str(e), "text": ""}


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
    """Load configuration from JSON file.

    Supports JS-style comments (// and /* */) so `config.json` can contain
    inline documentation. Performs basic validation for the FinMind quota
    keys and returns the parsed dict.
    """
    import re
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    txt = p.read_text(encoding='utf-8')
    # strip single-line (//) and block (/* */) comments but keep line breaks
    cleaned = re.sub(r"//.*?$|/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), txt, flags=re.M | re.S)
    cfg = json.loads(cleaned)

    # Basic validation and sensible defaults for new quota-related keys
    if 'api_usage_threshold' not in cfg:
        cfg['api_usage_threshold'] = 50
    if not isinstance(cfg['api_usage_threshold'], int):
        raise ValueError('api_usage_threshold must be an integer (absolute count or percent)')

    # unit may be 'absolute' or 'percent' — default to absolute for backward compat
    unit = str(cfg.get('api_usage_threshold_unit', 'absolute')).lower()
    if unit not in ('absolute', 'percent'):
        raise ValueError("api_usage_threshold_unit must be 'absolute' or 'percent'")
    cfg['api_usage_threshold_unit'] = unit

    # retry config
    cfg['api_retry_max'] = int(cfg.get('api_retry_max', 3))

    # Throttle configuration (adaptive backoff between requests)
    cfg['api_throttle_enabled'] = bool(cfg.get('api_throttle_enabled', False))
    cfg['api_throttle_min_seconds'] = float(cfg.get('api_throttle_min_seconds', 1.0))
    cfg['api_throttle_max_seconds'] = float(cfg.get('api_throttle_max_seconds', 10.0))
    cfg['api_throttle_mode'] = str(cfg.get('api_throttle_mode', 'linear')).lower()
    cfg['api_throttle_show_progress'] = bool(cfg.get('api_throttle_show_progress', True))

    # Validation / sensible clamping for throttle values
    if cfg['api_throttle_min_seconds'] < 0:
        logging.warning('api_throttle_min_seconds < 0; clamping to 0')
        cfg['api_throttle_min_seconds'] = 0.0
    if cfg['api_throttle_max_seconds'] < cfg['api_throttle_min_seconds']:
        logging.warning('api_throttle_max_seconds < api_throttle_min_seconds; adjusting to match min')
        cfg['api_throttle_max_seconds'] = cfg['api_throttle_min_seconds']

    return cfg


# Helper: quota math (percent <-> absolute) for a reported API limit
from math import ceil

def compute_effective_cutoff(api_usage_threshold: int, *, api_usage_threshold_unit: str = 'absolute', api_limit: int | None = None) -> int:
    """Return absolute remaining-request cutoff from config values.

    - If unit == 'absolute' -> return api_usage_threshold
    - If unit == 'percent' and api_limit given -> ceil(api_limit * threshold/100)
    - If unit == 'percent' and no api_limit -> fall back to configured value as absolute
    """
    unit = (api_usage_threshold_unit or 'absolute').lower()
    if unit == 'absolute':
        return int(api_usage_threshold)
    if unit == 'percent':
        if api_limit and isinstance(api_limit, int) and api_limit > 0:
            return int(ceil(api_limit * (api_usage_threshold / 100.0)))
        return int(api_usage_threshold)
    return int(api_usage_threshold)


def compute_throttle_seconds(*, remaining: int | None, limit: int | None, cutoff: int, min_seconds: float = 1.0, max_seconds: float = 10.0, mode: str = 'linear') -> float:
    """Map (remaining,limit,cutoff) -> throttle seconds.

    - linear mode: remaining==cutoff -> max_seconds, remaining==limit -> min_seconds
    - clamps to [min_seconds, max_seconds]
    - if limit is missing, uses a conservative heuristic (smaller throttle)

    Returns 0.0 when no throttle is needed or inputs are invalid.
    """
    try:
        if remaining is None:
            return 0.0
        rem = float(remaining)
        cut = float(cutoff)
        min_s = float(min_seconds)
        max_s = float(max_seconds)
        if max_s <= 0 or min_s < 0:
            return 0.0
        if rem <= cut:
            # caller will normally skip when rem <= cut; no throttle necessary here
            return 0.0
        # If we have an absolute limit, map rem in [cut, limit] -> [max_s, min_s]
        if isinstance(limit, int) and limit > cut:
            lim = float(limit)
            # normalize between cutoff..limit
            t = (rem - cut) / max((lim - cut), 1.0)
            t = max(0.0, min(1.0, t))
            if mode == 'linear':
                return float(max_s + (min_s - max_s) * t)
            # fallback to linear for unknown modes
            return float(max_s + (min_s - max_s) * t)
        else:
            # No limit available: use a decreasing linear heuristic where larger remaining => smaller sleep
            # Map rem in [cut, cut*5] -> [max_s, min_s]
            span_top = max(cut * 5.0, cut + 1.0)
            t = (rem - cut) / (span_top - cut)
            t = max(0.0, min(1.0, t))
            return float(max_s + (min_s - max_s) * t)
    except Exception:
        return 0.0


def _sleep_with_progress(total_seconds: float, *, show: bool = True, label: str | None = None) -> None:
    """Sleep while showing a terminal progress bar (non-blocking UI helper).

    Implementation notes:
    - Performs a single blocking time.sleep(total_seconds) in the caller so tests
      that monkeypatch time.sleep still observe the expected call.
    - A background thread updates the progress bar (uses Event.wait) and does not
      add extra time.sleep calls that could break tests that assert a single sleep.
    - If stdout is not a TTY or show is False, behaves like time.sleep(total_seconds).
    """
    import threading
    import time
    import sys

    if total_seconds <= 0:
        return
    if not show or not sys.stdout.isatty():
        time.sleep(total_seconds)
        return

    stop = threading.Event()

    def _updater():
        start = time.perf_counter()
        last_print = ''
        try:
            while not stop.is_set():
                elapsed = time.perf_counter() - start
                remaining = max(0.0, total_seconds - elapsed)
                frac = min(1.0, max(0.0, elapsed / max(total_seconds, 1e-6)))
                bar_w = 30
                filled = int(bar_w * frac)
                bar = ('=' * filled) + ('>' if filled < bar_w else '') + (' ' * max(0, bar_w - filled - 1))
                lbl = f" {label}:" if label else ''
                s = f"{lbl} [{bar}] {remaining:5.1f}s remaining"
                # only rewrite when changed to reduce terminal churn
                if s != last_print:
                    sys.stdout.write('\r' + s)
                    sys.stdout.flush()
                    last_print = s
                # wait briefly without calling time.sleep in a tight loop
                stop.wait(0.12)
                if elapsed >= total_seconds:
                    break
        except Exception:
            pass
        finally:
            # clear line and print a short confirmation
            try:
                sys.stdout.write('\r' + ' ' * (len(last_print) + 2) + '\r')
                sys.stdout.write(f"{label or 'Waiting'}: done\n")
                sys.stdout.flush()
            except Exception:
                pass

    t = threading.Thread(target=_updater, daemon=True)
    t.start()
    try:
        # keep a single sleep call so existing tests that monkeypatch time.sleep still work
        time.sleep(total_seconds)
    finally:
        stop.set()
        t.join(timeout=0.5)


def should_fetch_based_on_quota(remaining: int | None, limit: int | None, api_usage_threshold: int, api_usage_threshold_unit: str = 'absolute') -> bool:
    """Return True if it's safe to fetch (remaining > cutoff). If remaining is None, allow by default."""
    if remaining is None:
        return True
    try:
        rem = int(remaining)
    except Exception:
        return True
    cutoff = compute_effective_cutoff(api_usage_threshold, api_usage_threshold_unit=api_usage_threshold_unit, api_limit=limit)
    return rem > cutoff

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

def run_batch_fetch(stock_id: str, config: dict, temp_dir: Path) -> bool:
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

        # Configurable protection and retry settings
        api_usage_threshold = int(config.get('api_usage_threshold', 50))
        api_usage_threshold_unit = str(config.get('api_usage_threshold_unit', 'absolute')).lower()
        api_retry_max = int(config.get('api_retry_max', 3))

        # Ensure these exist on all code paths
        usage_parts: list[str] = []
        remaining = None
        limit = None
        # collect parent messages early so any early-return paths can record reasons
        parent_messages: list[str] = []

        if token:
            # Use helper to fetch usage info and apply threshold/backoff logic
            try:
                attempt = 0
                usage = fetch_finmind_usage(token)
                while attempt < api_retry_max and usage.get('status_code') == 429:
                    # Respect Retry-After when provided
                    retry_after = None
                    try:
                        # fetch_finmind_usage doesn't expose headers; do a light GET to read Retry-After
                        r = requests.get("https://api.web.finmindtrade.com/v2/user_info", headers={"Authorization": f"Bearer {token}"}, timeout=10)
                        retry_after = r.headers.get('Retry-After')
                    except Exception:
                        retry_after = None
                    wait = int(retry_after) if (retry_after and retry_after.isdigit()) else (5 * (2 ** attempt))
                    logging.warning(f"FinMind returned 429 (attempt {attempt+1}/{api_retry_max}), sleeping {wait}s before retry")
                    attempt += 1
                    import time
                    time.sleep(wait)
                    usage = fetch_finmind_usage(token)

                if not usage.get('ok') and usage.get('status_code') != 429:
                    logging.warning(f"FINMIND /user_info check failed (source: {token_source}): {usage.get('error') or usage.get('text')}")
                    # proceed — batch_fetch may still succeed using cache
                else:
                    data = usage.get('data', {}) or {}
                    # prefer explicit remaining; if missing, derive from reported limit & user_count
                    remaining = data.get('api_requests_remaining') or data.get('remaining') or data.get('api_remaining')
                    limit = data.get('api_request_limit') or data.get('api_limit')
                    user_count = data.get('user_count')
                    if remaining is None and isinstance(limit, int) and isinstance(user_count, int):
                        # computed remaining = available requests = limit - used_by_user(s)
                        remaining = int(limit) - int(user_count)
                        logging.debug("Computed API remaining from limit-user_count: %s", remaining)

                    # compute cutoff and decide; keep a parent_log snippet for visibility
                    effective_cutoff = compute_effective_cutoff(api_usage_threshold, api_usage_threshold_unit=api_usage_threshold_unit, api_limit=limit)
                    parent_log_lines = [
                        f"Computed API remaining={remaining!s} (limit={limit!s}, user_count={user_count!s})",
                        f"Computed cutoff={effective_cutoff} ({api_usage_threshold!s} {api_usage_threshold_unit})",
                    ]

                    if isinstance(remaining, int) and remaining <= effective_cutoff:
                        msg = f"API requests remaining is low ({remaining} <= {effective_cutoff}) — skipping fetch to avoid hitting quota"
                        logging.warning(msg)
                        parent_messages.append(msg)
                        parent_messages.append(f"Skipping fetch: remaining={remaining} <= cutoff={effective_cutoff}")
                        # write parent-only temp log so operator can see why we skipped
                        try:
                            temp_log_dir = Path(temp_dir)
                            temp_log_dir.mkdir(parents=True, exist_ok=True)
                            fetch_log = temp_log_dir / f"batch_fetch_{stock_id}.log"
                            with open(fetch_log, 'w', encoding='utf-8') as lf:
                                lf.write('--- PARENT ---\n')
                                lf.write("\n".join(parent_messages))
                                lf.write("\n\n")
                                lf.write('--- STDOUT ---\n')
                                lf.write('')
                                lf.write('\n--- STDERR ---\n')
                                lf.write('')
                            logging.info(f"Wrote parent-only batch_fetch log to {fetch_log} (skipped due to quota)")
                        except Exception as _e:
                            logging.warning(f"Unable to write parent-only batch_fetch log for {stock_id}: {_e}")
                        return False

                    if not should_fetch_based_on_quota(remaining, limit, api_usage_threshold, api_usage_threshold_unit):
                        parent_log_lines.append(f"Skipping fetch to avoid hitting quota (remaining={remaining} <= cutoff={effective_cutoff})")
                        # write parent section into the per-stock temp log (test expects this)
                        batch_log = Path(temp_dir) / f"batch_fetch_{stock_id}.log"
                        batch_log.write_text("--- PARENT ---\n" + "\n".join(parent_log_lines) + "\n", encoding="utf-8")
                        logging.warning("Skipping fetch for %s due to quota (remaining=%s, cutoff=%s)", stock_id, remaining, effective_cutoff)
                        return False

                    # export computed cutoff for downstream visibility (kept from earlier change)
                    env['ROBOT_API_USAGE_CUTOFF'] = str(effective_cutoff)

                    # If remaining not provided but we have user_count+limit, compute remaining
                    if remaining is None and isinstance(limit, int) and isinstance(data.get('user_count'), int):
                        try:
                            computed = int(limit) - int(data.get('user_count'))
                        except Exception:
                            computed = None
                        if isinstance(computed, int):
                            remaining = computed
                            logging.info(f"Computed api_requests_remaining from api_request_limit - user_count: {limit} - {data.get('user_count')} = {remaining}")
                            parent_messages.append(f"Computed API remaining={remaining} (from api_request_limit and user_count)")

                    # defensive: treat negative remaining as 0 (already over quota)
                    if isinstance(remaining, int) and remaining < 0:
                        logging.warning(f"API shows usage above limit (user_count={data.get('user_count')}, api_request_limit={limit}); treating remaining=0")
                        remaining = 0

                    # Log normalized usage info for terminal + logs
                    usage_parts = []
                    for k in ('user_count', 'api_request_limit', 'api_requests_remaining'):
                        if k in data:
                            usage_parts.append(f"{k}={data.get(k)}")
                    if usage_parts:
                        logging.info("API Usage: " + ", ".join(usage_parts))
                    if remaining is None:
                        logging.debug("api_requests_remaining not available from FinMind; quota enforcement will use computed values when possible")

                    # Enforce quota when we have a remaining value
                    if isinstance(remaining, int) and remaining <= effective_cutoff:
                        logging.warning(f"API requests remaining is low ({remaining} <= {effective_cutoff}) — skipping fetch to avoid hitting quota")
                        parent_messages.append(f"Skipping fetch: remaining={remaining} <= cutoff={effective_cutoff}")

                        # write parent-only temp log so operator can see why we skipped
                        try:
                            temp_log_dir = Path(temp_dir)
                            temp_log_dir.mkdir(parents=True, exist_ok=True)
                            fetch_log = temp_log_dir / f"batch_fetch_{stock_id}.log"
                            with open(fetch_log, 'w', encoding='utf-8') as lf:
                                lf.write('--- PARENT ---\n')
                                lf.write("\n".join(parent_messages))
                                lf.write("\n\n")
                                lf.write('--- STDOUT ---\n')
                                lf.write('')
                                lf.write('\n--- STDERR ---\n')
                                lf.write('')
                            logging.info(f"Wrote parent-only batch_fetch log to {fetch_log} (skipped due to quota)")
                        except Exception as _e:
                            logging.warning(f"Unable to write parent-only batch_fetch log for {stock_id}: {_e}")

                        return False

                    # Apply adaptive throttle when enabled and fetch is allowed
                    api_throttle_enabled = bool(config.get('api_throttle_enabled', False))
                    if api_throttle_enabled and isinstance(remaining, int) and remaining > effective_cutoff:
                        min_s = float(config.get('api_throttle_min_seconds', 1.0))
                        max_s = float(config.get('api_throttle_max_seconds', 10.0))
                        mode = str(config.get('api_throttle_mode', 'linear')).lower()
                        throttle = compute_throttle_seconds(remaining=remaining, limit=limit, cutoff=effective_cutoff, min_seconds=min_s, max_seconds=max_s, mode=mode)
                        if throttle and throttle > 0:
                            show_progress = bool(config.get('api_throttle_show_progress', True))
                            logging.info(f"Applying adaptive throttle before fetching {stock_id}: sleeping {throttle:.1f}s (remaining={remaining}, cutoff={effective_cutoff})")
                            env['ROBOT_API_THROTTLE_SECONDS'] = str(throttle)
                            _sleep_with_progress(throttle, show=show_progress, label=f"Throttle {stock_id}")

                    # If we received a valid response, export token for child process
                    if usage.get('ok'):
                        env['FINMIND_API_KEY'] = token
                        # also export reported remaining for downstream visibility
                        if isinstance(remaining, int):
                            env['FINMIND_API_REQUESTS_REMAINING'] = str(remaining)
            except Exception as e:
                logging.warning(f"Failed to validate FINMIND_API_KEY (source: {token_source}): {e}")
        else:
            logging.warning('No FINMIND_API_KEY found in config, environment, or .env; batch_fetch may use cache-only')

        # Decide whether to stream child output in real time or capture-only
        stream_child = bool(config.get('stream_child_output', False))

        # Collect parent messages to always write into the temp log
        parent_messages: list[str] = []

        # predefine fetch_log so error paths can reference it safely
        temp_log_dir = Path(temp_dir)
        fetch_log = temp_log_dir / f"batch_fetch_{stock_id}.log"

        # (Add parent messages from earlier checks if any were logged)
        # Note: keep any usage_parts logged above — reproduce them into parent_messages
        try:
            if usage_parts:
                parent_messages.append('Fetching API usage info...')
                parent_messages.append('API Usage: ' + ', '.join(usage_parts))
        except Exception:
            pass

        if not stream_child:
            # Default behavior: capture child output and write to temp log after completion
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, env=env, capture_output=True, text=True)
            # Save fetch stdout/stderr into temp log for diagnosis
            try:
                temp_log_dir = Path(temp_dir)
                temp_log_dir.mkdir(parents=True, exist_ok=True)
                fetch_log = temp_log_dir / f"batch_fetch_{stock_id}.log"
                with open(fetch_log, 'w', encoding='utf-8') as lf:
                    # Write parent messages first for easier diagnosis
                    if parent_messages:
                        lf.write('--- PARENT ---\n')
                        lf.write("\n".join(parent_messages))
                        lf.write("\n\n")
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
        else:
            # Stream child stdout/stderr to terminal while also capturing to disk
            from threading import Thread
            import io

            proc = subprocess.Popen(cmd, cwd=Path(__file__).parent.parent.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            def _reader(stream, buf, log_fn):
                try:
                    for line in iter(stream.readline, ''):
                        if not line:
                            break
                        # Mirror to terminal
                        log_fn(line.rstrip('\n'))
                        # Capture
                        buf.write(line)
                except Exception:
                    pass
                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

            t_out = Thread(target=_reader, args=(proc.stdout, stdout_buf, logging.info), daemon=True)
            t_err = Thread(target=_reader, args=(proc.stderr, stderr_buf, logging.error), daemon=True)
            t_out.start()
            t_err.start()
            proc.wait()
            t_out.join(timeout=1)
            t_err.join(timeout=1)

            # Persist combined log (parent + child)
            try:
                temp_log_dir = Path(temp_dir)
                temp_log_dir.mkdir(parents=True, exist_ok=True)
                fetch_log = temp_log_dir / f"batch_fetch_{stock_id}.log"
                with open(fetch_log, 'w', encoding='utf-8') as lf:
                    if parent_messages:
                        lf.write('--- PARENT ---\n')
                        lf.write("\n".join(parent_messages))
                        lf.write("\n\n")
                    lf.write('--- STDOUT ---\n')
                    lf.write(stdout_buf.getvalue())
                    lf.write('\n--- STDERR ---\n')
                    lf.write(stderr_buf.getvalue())
                logging.info(f"Saved batch_fetch logs to {fetch_log}")
            except Exception as e:
                logging.warning(f"Unable to write batch_fetch log for {stock_id}: {e}")

            if proc.returncode != 0:
                logging.error(f"Batch fetch failed for {stock_id}: see {fetch_log} for details")
                return False
            logging.info(f"Batch fetch completed for {stock_id}")
            return True
    except Exception as e:
        logging.error(f"Error running batch fetch for {stock_id}: {e}")
        return False

def _check_preprocessed_date_range(json_path: Path, config_start: str, config_end: str) -> bool:
    """Check if preprocessed JSON file covers the date range in config.
    
    Inspects the 'date' column in the preprocessed file and verifies that
    it includes [config_start, config_end]. Returns True if coverage is
    sufficient, False otherwise or if unable to determine.
    
    Args:
        json_path: Path to preprocessed JSON file.
        config_start: Start date from config (YYYY-MM-DD).
        config_end: End date from config (YYYY-MM-DD).
    
    Returns:
        True if file's date range covers the config range, False otherwise.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract records list
        if isinstance(data, dict) and 'data' in data:
            records = data['data']
        else:
            records = data if isinstance(data, list) else []
        
        if not records or len(records) == 0:
            logging.debug(f"Preprocessed file has no records: {json_path}")
            return False
        
        # Find date column (try common names)
        first_record = records[0]
        if not isinstance(first_record, dict):
            logging.debug(f"First record is not a dict: {type(first_record)}")
            return False
        
        date_col = None
        for key in ['date', 'Date', 'DATE']:
            if key in first_record:
                date_col = key
                break
        
        if not date_col:
            logging.debug(f"No 'date' column found in preprocessed file. Available keys: {list(first_record.keys())[:5]}")
            return False
        
        # Extract and sort dates
        dates = [rec.get(date_col) for rec in records if isinstance(rec, dict) and date_col in rec]
        if not dates:
            logging.debug(f"No date values extracted from '{date_col}' column")
            return False
        
        # Convert to strings and sort
        dates_str = [str(d) for d in dates]
        dates_sorted = sorted(dates_str)
        file_start = dates_sorted[0]
        file_end = dates_sorted[-1]
        
        # String comparison works for YYYY-MM-DD format
        config_start_str = str(config_start)
        config_end_str = str(config_end)
        covers = file_start <= config_start_str and file_end >= config_end_str
        
        if not covers:
            logging.info(
                f"Preprocessed file date range [{file_start}, {file_end}] "
                f"does not fully cover config range [{config_start_str}, {config_end_str}]"
            )
        else:
            logging.info(f"Preprocessed file date range [{file_start}, {file_end}] covers config range")
        
        return covers
    except Exception as e:
        logging.warning(f"Unable to check preprocessed file date range: {e}")
        return False


def run_preprocessing(stock_id, config, data_dir, temp_dir):
    """Run preprocessing.py for a specific stock."""
    try:
        company = twstock.codes.get(stock_id).name if stock_id in twstock.codes else stock_id
        datasets_dir = Path(__file__).parent.parent / "datasets" / f"{company}-{stock_id}"
        existing_preprocessed = datasets_dir / f"preprocessed_{company}-{stock_id}.json"
        
        output_json = os.path.join(data_dir, f"preprocessed_{stock_id}.json")
        output_csv = os.path.join(data_dir, f"preprocessed_{stock_id}.csv")
        
        # Priority 1: Use existing preprocessed file from datasets (with date range check)
        if existing_preprocessed.exists():
            # Check if file's date range covers config requirements
            config_start = config.get('start_date', '2020-01-01')
            config_end = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
            
            if _check_preprocessed_date_range(existing_preprocessed, config_start, config_end):
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
            else:
                # Date range insufficient, proceed to regenerate
                logging.info(f"Preprocessed file exists but date range insufficient, will regenerate from combined/fetch")
        
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

        indicator_params = config.get('indicator_params', {}) if isinstance(config, dict) else {}
        period = int(indicator_params.get('supertrend_period', 14))
        multiplier = float(indicator_params.get('supertrend_multiplier', 2.5))
        analysis_days = int(config.get('analysis_days', 14))
        use_fundamentals = bool(indicator_params.get('use_fundamentals', False))
        margin_balance_key = str(indicator_params.get('margin_balance_key', 'MarginPurchaseTodayBalance'))
        score_weights = indicator_params.get('score_weights', None)

        cmd = [
            sys.executable, 'engine/datasets/indicators.py',
            preprocessed_file,
            '--start_date', config['start_date']
        ]

        # Ensure the indicators script uses the robot's configured parameters
        cmd.extend(['--period', str(period)])
        cmd.extend(['--multiplier', str(multiplier)])
        cmd.extend(['--analysis_days', str(analysis_days)])

        # Enable chip/fundamental integration (dict.md: MarginPurchaseTodayBalance)
        if use_fundamentals:
            cmd.append('--with_fundamentals')
            cmd.extend(['--margin_balance_key', margin_balance_key])

        # Optional: override weighted scoring weights from config
        if isinstance(score_weights, dict) and score_weights:
            try:
                weights_json = json.dumps(score_weights, ensure_ascii=False)
                cmd.extend(['--score_weights_json', weights_json])
            except Exception as e:
                logging.warning(f"Unable to serialize score_weights for {stock_id}: {e}")

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
        stream_child = bool(config.get('stream_child_output', False))
        if not stream_child:
            with open(report_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, stdout=f, stderr=subprocess.PIPE, text=True, env=env)
            if result.returncode != 0:
                logging.error(f"Indicators failed for {stock_id}: {result.stderr}")
                return False
            logging.info(f"Indicators completed for {stock_id}, report saved to {report_file}")
            return True

        # stream_child == True: stream indicator output to terminal AND write to report file
        from threading import Thread
        import io
        proc = subprocess.Popen(cmd, cwd=Path(__file__).parent.parent.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        def _reader(stream, buf, write_fn):
            try:
                for line in iter(stream.readline, ''):
                    if not line:
                        break
                    write_fn(line.rstrip('\n'))
                    buf.write(line)
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = Thread(target=_reader, args=(proc.stdout, stdout_buf, logging.info), daemon=True)
        t_err = Thread(target=_reader, args=(proc.stderr, stderr_buf, logging.error), daemon=True)
        t_out.start()
        t_err.start()
        proc.wait()
        t_out.join(timeout=1)
        t_err.join(timeout=1)

        # write the streamed output into the report file so behavior is identical
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(stdout_buf.getvalue())
        except Exception as e:
            logging.warning(f"Unable to write indicators report for {stock_id}: {e}")

        if proc.returncode != 0:
            logging.error(f"Indicators failed for {stock_id}: see {report_file}")
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