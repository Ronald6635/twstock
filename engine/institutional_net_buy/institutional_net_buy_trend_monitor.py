"""
Institutional Trend Monitor Module

This module monitors institutional net-buy features from TFRecord data
without using machine learning models.

Key features:
- Loads pivoted institutional TFRecord records into tabular data
- Detects stocks with increasing or sustained-high cumulative net buy
- Adds price trend context and outputs recommendation reasons
- Exports results to a CSV file for further analysis or visualization

Usage:
```powershell
python institutional_net_buy_trend_monitor.py `
  --tfrecord-path institutional_net_buy_2026-02-08_2026-05-09.tfrecord `
  --recent-days 10 `
  --baseline-days 20 `
  --min-history-days 40 `
  --min-positive-ratio 0.6 `
  --top-k 20 `
  --output-csv trend_candidates_2026-02-08_2026-05-09.csv
```
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


FEATURE_COLUMNS = [
    "foreign_net_buy",
    "foreign_dealer_net_buy",
    "trust_net_buy",
    "dealer_hedge_net_buy",
    "dealer_net_buy",
]

DEFAULT_WEIGHTS = {
    "foreign_net_buy": 1.0,
    "foreign_dealer_net_buy": 1.0,
    "trust_net_buy": 1.0,
    # 自營商避險部位通常屬於短線、被動性買賣超，對波段趨勢的影響應適度降低。
    "dealer_hedge_net_buy": 0.5,
    # 自營商一般部位仍保留權重，但相較外資/投信可適度弱化。
    "dealer_net_buy": 0.8,
}


def _safe_slope(values: pd.Series) -> float:
    """Calculate linear slope for a series; return 0.0 on short/invalid input."""
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    x_axis = np.arange(len(clean), dtype=float)
    slope, _ = np.polyfit(x_axis, clean.to_numpy(dtype=float), deg=1)
    return float(slope)


def _compute_weighted_total_net_buy(dataframe: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute weighted institutional net buy, lowering hedge weight for trend stability."""
    missing_columns = [col for col in weights if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for weighted total net buy: {missing_columns}")

    weighted_columns = [dataframe[col].astype(float) * weight for col, weight in weights.items()]
    return sum(weighted_columns)


def load_tfrecord_to_dataframe(tfrecord_path: Path) -> pd.DataFrame:
    """Load pivoted TFRecord rows into a pandas DataFrame."""
    feature_description = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "stock_id": tf.io.FixedLenFeature([], tf.string),
        "stock_name": tf.io.FixedLenFeature([], tf.string),
        "close": tf.io.FixedLenFeature([], tf.float32),
        "foreign_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "foreign_dealer_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "trust_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "dealer_hedge_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "dealer_net_buy": tf.io.FixedLenFeature([], tf.int64),
    }

    rows: list[dict[str, object]] = []
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path), buffer_size=1024 * 1024 * 100) # 100MB buffer for efficiency
    for raw_example in raw_dataset:
        parsed = tf.io.parse_single_example(raw_example, feature_description)
        rows.append(
            {
                "date": parsed["date"].numpy().decode("utf-8"),
                "stock_id": str(parsed["stock_id"].numpy().decode("utf-8")),
                "stock_name": parsed["stock_name"].numpy().decode("utf-8"),
                "close": float(parsed["close"].numpy()),
                "foreign_net_buy": int(parsed["foreign_net_buy"].numpy()),
                "foreign_dealer_net_buy": int(parsed["foreign_dealer_net_buy"].numpy()),
                "trust_net_buy": int(parsed["trust_net_buy"].numpy()),
                "dealer_hedge_net_buy": int(parsed["dealer_hedge_net_buy"].numpy()),
                "dealer_net_buy": int(parsed["dealer_net_buy"].numpy()),
            }
        )

    if not rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(rows)
    dataframe["stock_id"] = dataframe["stock_id"].astype(str)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["date"])
    dataframe["total_net_buy"] = _compute_weighted_total_net_buy(dataframe, DEFAULT_WEIGHTS)
    dataframe = dataframe.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return dataframe


def _build_recommendation(price_change_pct: float, recent_positive_ratio: float) -> str:
    """Generate recommendation text by combining flow trend and price trend."""
    if price_change_pct > 3:
        return "法人籌碼與股價同向偏多，可列入強勢續抱/追蹤名單。"
    if price_change_pct >= -1:
        if recent_positive_ratio >= 0.7:
            return "法人連續偏買但股價仍整理，屬於可能補漲的觀察名單。"
        return "法人偏多但股價尚未明顯表態，建議等待突破訊號。"
    return "法人買盤增加但股價走弱，屬於背離型態，建議小倉位觀察風險。"


def analyze_trends(
    dataframe: pd.DataFrame,
    recent_days: int,
    baseline_days: int,
    min_history_days: int,
    min_positive_ratio: float,
    top_k: int,
) -> pd.DataFrame:
    """Analyze institutional and price trends; return top candidate stocks."""
    if dataframe.empty:
        return pd.DataFrame()

    candidates: list[dict[str, object]] = []
    grouped = dataframe.groupby(["stock_id", "stock_name"], sort=False)

    for (stock_id, stock_name), stock_data in grouped:
        stock_data = stock_data.sort_values("date").reset_index(drop=True)
        if len(stock_data) < max(min_history_days, recent_days + 5):
            continue

        recent_slice = stock_data.tail(recent_days).copy()
        baseline_pool = stock_data.iloc[: -recent_days] if len(stock_data) > recent_days else stock_data
        baseline_slice = baseline_pool.tail(baseline_days).copy()
        if baseline_slice.empty:
            continue

        recent_total = float(recent_slice["total_net_buy"].sum())
        recent_avg = float(recent_slice["total_net_buy"].mean())
        recent_positive_ratio = float((recent_slice["total_net_buy"] > 0).mean())
        cumulative_recent_slope = _safe_slope(recent_slice["total_net_buy"].cumsum())
        baseline_avg = float(baseline_slice["total_net_buy"].mean())

        increasing_signal = (
            recent_total > 0
            and recent_positive_ratio >= min_positive_ratio
            and cumulative_recent_slope > 0
        )
        sustained_high_signal = (
            recent_avg > 0
            and recent_positive_ratio >= min_positive_ratio
            and recent_avg >= max(1.15 * baseline_avg, baseline_avg + 100)
        )

        if not (increasing_signal or sustained_high_signal):
            continue

        close_recent = recent_slice["close"].astype(float)
        price_slope = _safe_slope(close_recent)
        price_start = float(close_recent.iloc[0])
        price_end = float(close_recent.iloc[-1])
        if abs(price_start) < 1e-6:
            price_change_pct = 0.0
        else:
            price_change_pct = (price_end - price_start) / price_start * 100.0

        if price_change_pct > 2:
            price_trend = "上升"
        elif price_change_pct < -2:
            price_trend = "下降"
        else:
            price_trend = "盤整"

        signal_strength = 0.0
        signal_strength += min(40.0, recent_positive_ratio * 40.0)
        signal_strength += min(30.0, max(0.0, recent_avg / (abs(baseline_avg) + 1.0) * 15.0))
        signal_strength += min(20.0, max(0.0, cumulative_recent_slope / (abs(recent_avg) + 1.0) * 20.0))
        signal_strength += min(10.0, max(-5.0, price_change_pct))

        observed = (
            f"近{recent_days}日法人合計淨買超 {recent_total:,.0f}，"
            f"正向日占比 {recent_positive_ratio:.0%}；"
            f"近期待續均值 {recent_avg:,.0f}，基準期均值 {baseline_avg:,.0f}。"
        )
        if increasing_signal and sustained_high_signal:
            observed += "型態: 累積買超逐步增加且維持高檔。"
        elif increasing_signal:
            observed += "型態: 累積買超逐步增加。"
        else:
            observed += "型態: 買超維持高檔。"

        observed += (
            f" 股價趨勢: {price_trend}"
            f" ({price_start:.2f} -> {price_end:.2f}, {price_change_pct:+.2f}%, 斜率 {price_slope:+.3f})。"
        )

        candidates.append(
            {
                "stock_id": stock_id,
                "stock_name": stock_name,
                "signal_strength": round(signal_strength, 2),
                "recent_total_net_buy": round(recent_total, 2),
                "recent_positive_ratio": round(recent_positive_ratio, 4),
                "price_change_pct": round(price_change_pct, 2),
                "observed_phenomenon": observed,
                "recommendation_reason": _build_recommendation(
                    price_change_pct=price_change_pct,
                    recent_positive_ratio=recent_positive_ratio,
                ),
            }
        )

    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame(candidates).sort_values("signal_strength", ascending=False)
    return result.head(top_k).reset_index(drop=True)


def build_default_output_path(dataframe: pd.DataFrame, tfrecord_path: Path) -> Path:
    """Build the default CSV path from the input date range."""
    start_date = dataframe["date"].min()
    end_date = dataframe["date"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        return tfrecord_path.with_name("trend_candidates.csv")

    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    return tfrecord_path.with_name(f"trend_candidates_{start_str}_{end_str}.csv")


def main() -> None:
    """Entry point for institutional trend monitoring."""
    parser = argparse.ArgumentParser(description="監控法人特徵趨勢 (非 ML)")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="輸入 TFRecord 檔案路徑")
    parser.add_argument("--recent-days", type=int, default=10, help="近期觀察天數")
    parser.add_argument("--baseline-days", type=int, default=20, help="基準期天數")
    parser.add_argument("--min-history-days", type=int, default=40, help="單一股票最少歷史資料天數")
    parser.add_argument("--min-positive-ratio", type=float, default=0.6, help="近期淨買超為正的最小占比")
    parser.add_argument("--top-k", type=int, default=20, help="輸出前 N 檔")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="選填，輸出結果 CSV 檔案路徑；未提供時自動命名為 trend_candidates_{start_date}_{end_date}.csv",
    )
    args = parser.parse_args()

    tfrecord_path = Path(args.tfrecord_path)
    if not tfrecord_path.exists():
        print(f"找不到檔案: {tfrecord_path}")
        return

    dataframe = load_tfrecord_to_dataframe(tfrecord_path)
    if dataframe.empty:
        print("TFRecord 無有效資料，無法進行趨勢監控。")
        return

    result = analyze_trends(
        dataframe=dataframe,
        recent_days=args.recent_days,
        baseline_days=args.baseline_days,
        min_history_days=args.min_history_days,
        min_positive_ratio=args.min_positive_ratio,
        top_k=args.top_k,
    )

    if result.empty:
        print("未找到符合「法人累積買超增加或維持高檔」條件的股票。")
        return

    print("\n符合條件的股票 (依訊號強度排序):")
    for rank, row in enumerate(result.itertuples(index=False), start=1):
        print("-" * 90)
        print(
            f"{rank}. {row.stock_name} ({row.stock_id}) | "
            f"Signal={row.signal_strength} | "
            f"近期待淨買超={row.recent_total_net_buy:,.0f} | "
            f"正向占比={row.recent_positive_ratio:.0%} | "
            f"近期待股價變化={row.price_change_pct:+.2f}%"
        )
        print(f"   觀察現象: {row.observed_phenomenon}")
        print(f"   推薦原因: {row.recommendation_reason}")

    output_path = Path(args.output_csv) if args.output_csv else build_default_output_path(dataframe, tfrecord_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure stock_id formatting remains as string for CSV export (to preserve leading zeros)
    result["stock_id"] = result["stock_id"].astype(str)
    
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n已輸出結果至: {output_path}")


if __name__ == "__main__":
    main()