"""
Institutional Net Buy Fetcher

Fetches institutional investor buy/sell data from FinMind for a target stock set,
normalizes output columns, and exports results to CSV.

Key features:
- Stock source mode: all/file/list
- Quota-aware protection with adaptive throttle
- Retry with exponential backoff for unstable API calls
- Optional TFRecord export for TensorFlow pipelines
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import re
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from FinMind.data import DataLoader

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ==========================================
# PATHS AND CONSTANTS
# ==========================================
ROOT_DIR: Path = Path(__file__).parent.parent.parent
DEFAULT_STOCK_INFO_PATH: Path = ROOT_DIR / "docs" / "tw_stock_info.json"
DEFAULT_TARGET_FILE: Path = Path(__file__).parent / "targets.txt"

INVESTOR_NAME_MAP: dict[str, str] = {
    "Foreign_Investor": "外資",
    "ForeignInvestor": "外資",
    "Investment_Trust": "投信",
    "Dealer_self": "自營商",
    "Dealer_Hedging": "避險自營商",
    "Foreign_Dealer_Self": "外資自營商",
}


# ==========================================
# CONFIG MODEL
# ==========================================
@dataclass
class FetchConfig:
    """Runtime configuration for institutional net-buy fetching."""

    stock_source: str
    target_file: Path
    stocks_csv: str
    end_date: str
    days: int
    max_stocks: int
    stock_offset: int
    request_delay_seconds: float
    request_delay_jitter: float
    api_usage_threshold: int
    api_usage_threshold_unit: str
    api_retry_max: int
    api_throttle_enabled: bool
    api_throttle_min_seconds: float
    api_throttle_max_seconds: float
    api_throttle_mode: str
    csv_path: Path
    export_tfrecord: bool
    tfrecord_path: Path
    log_level: str


# ==========================================
# FINMIND HELPERS
# ==========================================
def fetch_finmind_usage(token: str, timeout: int = 10) -> dict[str, Any]:
    """Return a normalized dict for /v2/user_info."""
    url = "https://api.web.finmindtrade.com/v2/user_info"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        payload: dict[str, Any] = {}
        if resp.ok:
            content_type = resp.headers.get("content-type", "").lower()
            if content_type.startswith("application/json"):
                try:
                    payload = resp.json()
                except Exception:
                    payload = {}
        return {
            "ok": bool(resp.ok),
            "status_code": resp.status_code,
            "data": payload if isinstance(payload, dict) else {},
            "text": (resp.text[:200] if hasattr(resp, "text") else ""),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "data": {},
            "error": str(exc),
            "text": "",
        }


def compute_effective_cutoff(
    api_usage_threshold: int,
    *,
    api_usage_threshold_unit: str = "absolute",
    api_limit: int | None = None,
) -> int:
    """Return absolute remaining-request cutoff from threshold config."""
    unit = (api_usage_threshold_unit or "absolute").lower()
    if unit == "absolute":
        return int(api_usage_threshold)
    if unit == "percent":
        if api_limit and isinstance(api_limit, int) and api_limit > 0:
            return int(ceil(api_limit * (api_usage_threshold / 100.0)))
        return int(api_usage_threshold)
    return int(api_usage_threshold)


def should_fetch_based_on_quota(
    remaining: int | None,
    limit: int | None,
    api_usage_threshold: int,
    api_usage_threshold_unit: str = "absolute",
) -> bool:
    """Return True when remaining requests are above cutoff."""
    if remaining is None:
        return True
    try:
        rem = int(remaining)
    except Exception:
        return True
    cutoff = compute_effective_cutoff(
        api_usage_threshold,
        api_usage_threshold_unit=api_usage_threshold_unit,
        api_limit=limit,
    )
    return rem > cutoff


def compute_throttle_seconds(
    *,
    remaining: int | None,
    limit: int | None,
    cutoff: int,
    min_seconds: float = 1.0,
    max_seconds: float = 10.0,
    mode: str = "linear",
) -> float:
    """Map quota signals to adaptive sleep duration."""
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
            return 0.0

        if isinstance(limit, int) and limit > cut:
            lim = float(limit)
            t = (rem - cut) / max((lim - cut), 1.0)
            t = max(0.0, min(1.0, t))
            if mode == "linear":
                return float(max_s + (min_s - max_s) * t)
            return float(max_s + (min_s - max_s) * t)

        span_top = max(cut * 5.0, cut + 1.0)
        t = (rem - cut) / (span_top - cut)
        t = max(0.0, min(1.0, t))
        return float(max_s + (min_s - max_s) * t)
    except Exception:
        return 0.0


# ==========================================
# STOCK LOADING
# ==========================================
def load_twse_stock_map(stock_info_path: Path) -> dict[str, str]:
    """Load TWSE stock_id -> stock_name map from tw_stock_info.json."""
    if not stock_info_path.exists():
        raise FileNotFoundError(f"找不到股票清單檔案: {stock_info_path}")

    with open(stock_info_path, "r", encoding="utf-8") as file:
        stock_data = json.load(file)

    if not isinstance(stock_data, list):
        raise ValueError("tw_stock_info.json 內容格式錯誤，應為 list")

    stock_map: dict[str, str] = {}
    for item in stock_data:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "twse":
            continue
        if item.get("industry_category") == "Index":
            continue

        stock_id = str(item.get("stock_id", "")).strip()
        if not stock_id:
            continue

        stock_name = str(item.get("stock_name", "")).strip()
        # Preserve deterministic first-seen order.
        if stock_id not in stock_map:
            stock_map[stock_id] = stock_name

    return stock_map


def load_target_stocks_from_file(file_path: Path) -> list[str]:
    """Load stock IDs from a text file, one ID per line."""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到 target 檔案: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        stocks = [line.strip() for line in file if line.strip()]

    return stocks


def _deduplicate_preserve_order(stocks: list[str]) -> list[str]:
    """Return unique stock IDs while preserving input order."""
    seen: set[str] = set()
    result: list[str] = []
    for stock_id in stocks:
        sid = str(stock_id).strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        result.append(sid)
    return result


def build_stock_list(config: FetchConfig, stock_name_map: dict[str, str]) -> list[str]:
    """Build stock list from configured source mode."""
    source = config.stock_source.lower()

    if source == "all":
        stocks = list(stock_name_map.keys())
    elif source == "file":
        stocks = load_target_stocks_from_file(config.target_file)
    elif source == "list":
        stocks = [s.strip() for s in config.stocks_csv.split(",") if s.strip()]
    else:
        raise ValueError("stock_source 必須是 all / file / list")

    stocks = _deduplicate_preserve_order(stocks)

    if config.stock_offset > 0:
        stocks = stocks[config.stock_offset :]
    if config.max_stocks > 0:
        stocks = stocks[: config.max_stocks]

    return stocks


# ==========================================
# DATA FETCHING
# ==========================================
def _safe_request_delay(base_delay: float, jitter: float) -> None:
    """Sleep with optional jitter to smooth request burst patterns."""
    delay = max(0.0, float(base_delay))
    if jitter > 0:
        delay += random.uniform(0.0, float(jitter))
    if delay > 0:
        time.sleep(delay)


def _normalize_usage_data(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract remaining, limit, user_count from user_info payload."""
    remaining = data.get("api_requests_remaining") or data.get("remaining") or data.get("api_remaining")
    limit = data.get("api_request_limit") or data.get("api_limit")
    user_count = data.get("user_count")

    if remaining is None and isinstance(limit, int) and isinstance(user_count, int):
        remaining = int(limit) - int(user_count)

    if isinstance(remaining, int) and remaining < 0:
        remaining = 0

    return remaining, limit, user_count


def _fetch_single_stock_with_retry(
    api: DataLoader,
    stock_id: str,
    start_date: str,
    end_date: str,
    retry_max: int,
) -> pd.DataFrame:
    """Fetch one stock with bounded retry on transient errors."""
    attempt = 0
    while True:
        try:
            df = api.taiwan_stock_institutional_investors(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            attempt += 1
            if attempt > retry_max:
                raise RuntimeError(f"{stock_id} 取得失敗（重試超過上限）: {exc}") from exc
            wait = min(30.0, 2.0**attempt)
            logging.warning(
                "%s 取得失敗，%s 秒後重試 (%s/%s): %s",
                stock_id,
                wait,
                attempt,
                retry_max,
                exc,
            )
            time.sleep(wait)


def _fetch_stock_price_with_retry(
    api: DataLoader,
    stock_id: str,
    start_date: str,
    end_date: str,
    retry_max: int,
) -> pd.DataFrame:
    """Fetch daily OHLCV price for one stock with bounded retry on transient errors."""
    attempt = 0
    while True:
        try:
            df = api.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            attempt += 1
            if attempt > retry_max:
                logging.warning("%s 收盤價取得失敗（重試超過上限）: %s", stock_id, exc)
                return pd.DataFrame()
            wait = min(30.0, 2.0 ** attempt)
            logging.warning(
                "%s 收盤價取得失敗，%s 秒後重試 (%s/%s): %s",
                stock_id, wait, attempt, retry_max, exc,
            )
            time.sleep(wait)


def get_smart_money_consensus(
    api: DataLoader,
    stock_list: list[str],
    stock_name_map: dict[str, str],
    start_date: str,
    end_date: str,
    token: str,
    config: FetchConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch and normalize institutional net-buy data for stock list."""
    all_data: list[pd.DataFrame] = []
    status = {"success": 0, "empty": 0, "error": 0, "skipped_quota": 0, "cached": 0}

    cache_dir = Path("log") / f"{start_date}_{end_date}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info("本機暫存目錄: %s", cache_dir.absolute())

    for idx, stock_id in enumerate(stock_list, start=1):
        logging.info("\n==================================================================")
        logging.info("[%s/%s] 處理 %s", idx, len(stock_list), stock_id)

        stock_name = stock_name_map.get(stock_id, "")
        # 移除 Windows 檔名不允許的特殊字元 (例如 *, ?, /, \)
        safe_stock_name = re.sub(r'[\\/*?:"<>|]', "", stock_name)
        
        cache_file = cache_dir / f"{safe_stock_name}_{stock_id}.csv"

        if cache_file.exists():
            logging.info("%s 已有暫存資料，略過抓取", stock_id)
            try:
                cached_df = pd.read_csv(cache_file, dtype={"stock_id": str})
                if not cached_df.empty:
                    cached_df["date"] = cached_df["date"].astype(str)
                    for col in ["stock_id", "stock_name"]:
                        if col in cached_df.columns:
                            cached_df[col] = cached_df[col].astype(str)
                    # If stock_id was accidentally saved as integer (e.g. '903' instead of '00903')
                    # enforce the original stock_id from the loop to preserve leading zeros
                    cached_df["stock_id"] = str(stock_id)
                    all_data.append(cached_df)
                    status["success"] += 1
                else:
                    status["empty"] += 1
                status["cached"] += 1
                continue
            except Exception as e:
                logging.warning("%s 讀取暫存檔失敗，將重新抓取: %s", stock_id, e)

        remaining: int | None = None
        limit: int | None = None
        if token:
            usage = fetch_finmind_usage(token)
            if usage.get("ok"):
                usage_data = usage.get("data", {}) or {}
                remaining, limit, user_count = _normalize_usage_data(usage_data)

                effective_cutoff = compute_effective_cutoff(
                    config.api_usage_threshold,
                    api_usage_threshold_unit=config.api_usage_threshold_unit,
                    api_limit=limit,
                )
                if not should_fetch_based_on_quota(
                    remaining,
                    limit,
                    config.api_usage_threshold,
                    config.api_usage_threshold_unit,
                ):
                    status["skipped_quota"] += 1
                    logging.warning(
                        "跳過 %s：quota 偏低 (remaining=%s, limit=%s, user_count=%s, cutoff=%s)",
                        stock_id,
                        remaining,
                        limit,
                        user_count,
                        effective_cutoff,
                    )
                    continue

                if config.api_throttle_enabled and isinstance(remaining, int) and remaining > effective_cutoff:
                    throttle = compute_throttle_seconds(
                        remaining=remaining,
                        limit=limit,
                        cutoff=effective_cutoff,
                        min_seconds=config.api_throttle_min_seconds,
                        max_seconds=config.api_throttle_max_seconds,
                        mode=config.api_throttle_mode,
                    )
                    if throttle > 0:
                        logging.info(
                            "Adaptive throttle for %s: %.1fs (remaining=%s, cutoff=%s)",
                            stock_id,
                            throttle,
                            remaining,
                            effective_cutoff,
                        )
                        time.sleep(throttle)
            else:
                logging.warning("無法取得 FinMind quota 資訊，改用一般節流: %s", usage.get("error") or usage.get("text"))

        try:
            df = _fetch_single_stock_with_retry(
                api,
                stock_id,
                start_date,
                end_date,
                retry_max=config.api_retry_max,
            )
        except Exception as exc:
            status["error"] += 1
            logging.error("%s 抓取失敗: %s", stock_id, exc)
            _safe_request_delay(config.request_delay_seconds, config.request_delay_jitter)
            continue

        if df.empty:
            status["empty"] += 1
            logging.info("%s 無資料", stock_id)
            pd.DataFrame(columns=["date", "stock_id", "stock_name", "close"]).to_csv(cache_file, index=False, encoding="utf-8-sig")
        else:
            df = df.copy()
            df["stock_id"] = str(stock_id)  # Ensure string type to preserve leading zeros
            df["stock_name"] = stock_name

            # Fetch and merge daily close price
            price_df = _fetch_stock_price_with_retry(
                api, stock_id, start_date, end_date, retry_max=config.api_retry_max
            )
            if not price_df.empty and "close" in price_df.columns:
                price_df = price_df[["date", "close"]].copy()
                price_df["date"] = price_df["date"].astype(str)
                df["date"] = df["date"].astype(str)
                df = df.merge(price_df, on="date", how="left")
            else:
                df["close"] = float("nan")

            # Force stock_id to string before saving CSV
            df["stock_id"] = df["stock_id"].astype(str)
            df.to_csv(cache_file, index=False, encoding="utf-8-sig")
            all_data.append(df)
            status["success"] += 1

        _safe_request_delay(config.request_delay_seconds, config.request_delay_jitter)

    if not all_data:
        return pd.DataFrame(), status

    combined_df = pd.concat(all_data, ignore_index=True)
    if combined_df.empty:
        return pd.DataFrame(), status

    if "name" in combined_df.columns:
        combined_df["investor_name"] = combined_df["name"].replace(INVESTOR_NAME_MAP)
    else:
        combined_df["investor_name"] = ""

    for col in ["buy", "sell"]:
        if col not in combined_df.columns:
            combined_df[col] = 0
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(0)

    if "close" not in combined_df.columns:
        combined_df["close"] = float("nan")
    combined_df["close"] = pd.to_numeric(combined_df["close"], errors="coerce").fillna(0.0)

    combined_df["net_buy"] = combined_df["buy"] - combined_df["sell"]
    combined_df["buy"] = (combined_df["buy"] / 1000).astype(int)
    combined_df["sell"] = (combined_df["sell"] / 1000).astype(int)
    combined_df["net_buy"] = (combined_df["net_buy"] / 1000).astype(int)

    if "date" not in combined_df.columns:
        combined_df["date"] = ""

    for col in ["stock_id", "stock_name", "investor_name", "date"]:
        combined_df[col] = combined_df[col].astype(str)

    result_columns = [
        "date",
        "stock_id",
        "stock_name",
        "investor_name",
        "buy",
        "sell",
        "net_buy",
        "close",
    ]
    consensus_df = combined_df[result_columns].sort_values(
        by=["stock_id", "date", "net_buy"],
        ascending=[True, True, False],
    )

    return consensus_df, status


# ==========================================
# EXPORT HELPERS
# ==========================================
def export_to_tfrecord(df: pd.DataFrame, tfrecord_path: Path) -> None:
    """Export DataFrame rows into TFRecord (tf.train.Example).

    Uses column-wise numpy array access instead of iterrows() for
    significantly better performance on large DataFrames.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("tensorflow is not installed. Run: pip install tensorflow")

    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)

    # 進行 Pivot，將原本同一天的 5 筆法人的 net_buy 展開變成獨立特徵欄位
    pivot_df = df.pivot_table(
        index=["date", "stock_id", "stock_name", "close"],
        columns="investor_name",
        values="net_buy",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # 確保五大法人欄位皆存在
    expected_investors = ["外資", "外資自營商", "投信", "避險自營商", "自營商"]
    for inv in expected_investors:
        if inv not in pivot_df.columns:
            pivot_df[inv] = 0

    pivot_df = pivot_df.sort_values(by=["stock_id", "date"]).reset_index(drop=True)

    n_rows = len(pivot_df)
    import numpy as np
    
    foreign_cost = np.zeros(n_rows, dtype=float)
    foreign_dealer_cost = np.zeros(n_rows, dtype=float)
    trust_cost = np.zeros(n_rows, dtype=float)
    dealer_hedge_cost = np.zeros(n_rows, dtype=float)
    dealer_cost = np.zeros(n_rows, dtype=float)
    cost_arrays = [foreign_cost, foreign_dealer_cost, trust_cost, dealer_hedge_cost, dealer_cost]

    prev_stock = None
    inv_state = {inv: {"inventory": 0.0, "total_cost": 0.0} for inv in expected_investors}

    for i in range(n_rows):
        curr_stock = pivot_df.at[i, "stock_id"]
        if curr_stock != prev_stock:
            for inv in expected_investors:
                inv_state[inv] = {"inventory": 0.0, "total_cost": 0.0}
            prev_stock = curr_stock
        
        close_price = pivot_df.at[i, "close"]
        
        for inv_idx, inv in enumerate(expected_investors):
            net_buy = pivot_df.at[i, inv]
            
            if net_buy > 0:
                inv_state[inv]["inventory"] += net_buy
                inv_state[inv]["total_cost"] += net_buy * close_price
            elif net_buy < 0:
                # Sell: reduce inventory proportionally
                if inv_state[inv]["inventory"] > 0:
                    avg_c = inv_state[inv]["total_cost"] / inv_state[inv]["inventory"]
                    inv_state[inv]["inventory"] += net_buy
                    if inv_state[inv]["inventory"] <= 0:
                        inv_state[inv]["inventory"] = 0.0
                        inv_state[inv]["total_cost"] = 0.0
                    else:
                        inv_state[inv]["total_cost"] = inv_state[inv]["inventory"] * avg_c
            
            c_val = 0.0
            if inv_state[inv]["inventory"] > 0:
                c_val = inv_state[inv]["total_cost"] / inv_state[inv]["inventory"]
            cost_arrays[inv_idx][i] = c_val

    # Extract columns as numpy arrays once
    dates = pivot_df["date"].to_numpy(dtype=str)
    stock_ids = pivot_df["stock_id"].to_numpy(dtype=str)
    stock_names = pivot_df["stock_name"].to_numpy(dtype=str)
    closes = pivot_df["close"].to_numpy(dtype=float)

    foreign_net = pivot_df["外資"].to_numpy(dtype=int)
    foreign_dealer_net = pivot_df["外資自營商"].to_numpy(dtype=int)
    trust_net = pivot_df["投信"].to_numpy(dtype=int)
    dealer_hedge_net = pivot_df["避險自營商"].to_numpy(dtype=int)
    dealer_net = pivot_df["自營商"].to_numpy(dtype=int)

    total = len(dates)
    logging.info("TFRecord 序列化開始 (Pivot後已展開特徵): %s 筆", total)

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for i in range(total):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "date": tf.train.Feature(bytes_list=tf.train.BytesList(value=[dates[i].encode("utf-8")])),
                        "stock_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stock_ids[i].encode("utf-8")])),
                        "stock_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[stock_names[i].encode("utf-8")])),
                        "close": tf.train.Feature(float_list=tf.train.FloatList(value=[float(closes[i])])),
                        "foreign_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(foreign_net[i])])),
                        "foreign_dealer_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(foreign_dealer_net[i])])),
                        "trust_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(trust_net[i])])),
                        "dealer_hedge_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(dealer_hedge_net[i])])),
                        "dealer_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(dealer_net[i])])),
                        "foreign_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[float(foreign_cost[i])])),
                        "foreign_dealer_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[float(foreign_dealer_cost[i])])),
                        "trust_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[float(trust_cost[i])])),
                        "dealer_hedge_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[float(dealer_hedge_cost[i])])),
                        "dealer_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[float(dealer_cost[i])])),
                    }
                )
            )
            writer.write(example.SerializeToString())


# ==========================================
# ARGUMENT PARSING
# ==========================================
def parse_args() -> FetchConfig:
    """Parse command-line arguments into FetchConfig."""
    parser = argparse.ArgumentParser(description="Institutional net-buy batch fetcher")
    parser.add_argument("--stock-source", default="all", choices=["all", "file", "list"])
    parser.add_argument("--target-file", default=str(DEFAULT_TARGET_FILE))
    parser.add_argument("--stocks", default="", help="Comma-separated stock IDs for stock-source=list")

    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--days", type=int, default=90)

    parser.add_argument("--max-stocks", type=int, default=0, help="0 means no cap")
    parser.add_argument("--stock-offset", type=int, default=0)

    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    parser.add_argument("--request-delay-jitter", type=float, default=0.0)

    parser.add_argument("--api-usage-threshold", type=int, default=50)
    parser.add_argument("--api-usage-threshold-unit", default="absolute", choices=["absolute", "percent"])
    parser.add_argument("--api-retry-max", type=int, default=3)

    parser.add_argument("--api-throttle-enabled", action="store_true")
    parser.add_argument("--api-throttle-min-seconds", type=float, default=1.0)
    parser.add_argument("--api-throttle-max-seconds", type=float, default=10.0)
    parser.add_argument("--api-throttle-mode", default="linear")

    parser.add_argument("--csv-path", default="")
    parser.add_argument("--export-tfrecord", action="store_true")
    parser.add_argument("--tfrecord-path", default="")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    end_date = str(args.end_date)
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=int(args.days))).strftime("%Y-%m-%d")

    csv_path = Path(args.csv_path) if args.csv_path else Path(
        f"institutional_net_buy_{start_date}_{end_date}.csv"
    )
    tfrecord_path = Path(args.tfrecord_path) if args.tfrecord_path else Path(
        f"institutional_net_buy_{start_date}_{end_date}.tfrecord"
    )

    return FetchConfig(
        stock_source=str(args.stock_source),
        target_file=Path(args.target_file),
        stocks_csv=str(args.stocks),
        end_date=end_date,
        days=int(args.days),
        max_stocks=int(args.max_stocks),
        stock_offset=int(args.stock_offset),
        request_delay_seconds=float(args.request_delay_seconds),
        request_delay_jitter=float(args.request_delay_jitter),
        api_usage_threshold=int(args.api_usage_threshold),
        api_usage_threshold_unit=str(args.api_usage_threshold_unit),
        api_retry_max=int(args.api_retry_max),
        api_throttle_enabled=bool(args.api_throttle_enabled),
        api_throttle_min_seconds=float(args.api_throttle_min_seconds),
        api_throttle_max_seconds=float(args.api_throttle_max_seconds),
        api_throttle_mode=str(args.api_throttle_mode),
        csv_path=csv_path,
        export_tfrecord=bool(args.export_tfrecord),
        tfrecord_path=tfrecord_path,
        log_level=str(args.log_level),
    )


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    """Main entrypoint."""
    config = parse_args()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Fail fast: check TF availability before making any API calls
    if config.export_tfrecord and not _TF_AVAILABLE:
        raise RuntimeError("--export-tfrecord 需要 tensorflow，請先執行: pip install tensorflow")

    load_dotenv()
    token = os.getenv("FINMIND_API_KEY")
    if not token:
        raise ValueError("請設定 FINMIND_API_KEY")

    stock_name_map = load_twse_stock_map(DEFAULT_STOCK_INFO_PATH)
    stocks = build_stock_list(config, stock_name_map)

    if not stocks:
        logging.warning("沒有可處理的股票，請檢查 stock source 設定")
        return

    target_start_date = (
        pd.Timestamp(config.end_date) - pd.Timedelta(days=config.days)
    ).strftime("%Y-%m-%d")

    logging.info("將抓取近 %s 天資料：%s -> %s", config.days, target_start_date, config.end_date)
    logging.info("股票數量: %s", len(stocks))

    api = DataLoader()
    api.login_by_token(api_token=token)

    consensus_df, status = get_smart_money_consensus(
        api=api,
        stock_list=stocks,
        stock_name_map=stock_name_map,
        start_date=target_start_date,
        end_date=config.end_date,
        token=token,
        config=config,
    )

    logging.info(
        "完成: success=%s (含快取 %s), empty=%s, error=%s, skipped_quota=%s",
        status["success"],
        status["cached"],
        status["empty"],
        status["error"],
        status["skipped_quota"],
    )

    if consensus_df.empty:
        logging.warning("找不到指定條件的法人資料")
        return

    config.csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure stock_id formatting is preserved as string for the main CSV export
    consensus_df["stock_id"] = consensus_df["stock_id"].astype(str)
    consensus_df.to_csv(config.csv_path, index=False, encoding="utf-8-sig")
    logging.info("CSV 匯出完成: %s", config.csv_path)

    if config.export_tfrecord:
        export_to_tfrecord(consensus_df, config.tfrecord_path)
        logging.info("TFRecord 匯出完成: %s", config.tfrecord_path)


if __name__ == "__main__":
    main()
