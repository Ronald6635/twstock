"""
使用訓練好的 V3 模型進行全市場選股 (40天精準全對齊版)
==================================================================================
全面批量推理全市場個股，挑選 Top 20 潛力回報股，並生成 40 天歷史日期完美咬合的看盤圖表。
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_WEIGHTS = {
    "foreign": 1.0, "foreign_dealer": 1.0, "trust": 1.0,
    "dealer": 0.5, "dealer_hedge": 0.5
}

ALL_FEATURES = {
    "net_buy": ["foreign_net_buy", "trust_net_buy", "dealer_net_buy", "foreign_dealer_net_buy", "dealer_hedge_net_buy"],
    "intensity": ["conviction_intensity"],
    "cost_deviation": ["foreign_cost_dev", "trust_cost_dev", "dealer_cost_dev", "foreign_dealer_cost_dev", "dealer_hedge_cost_dev"],
    "price_action": ["vwap_change_pct"]
}
SELECTED_FEATURES = tuple(
    ALL_FEATURES["intensity"] + ALL_FEATURES["cost_deviation"] +
    ALL_FEATURES["net_buy"] + ALL_FEATURES["price_action"]
)
INPUT_DIM = len(SELECTED_FEATURES)

# Visualization constants
DEFAULT_PLOT_HISTORY_LEN = 40
FUTURE_DAYS = 5
DEFAULT_CONFIDENCE = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.7
CONFIDENCE_COLOR_HIGH = '#e8f5e9'  # Deep green
CONFIDENCE_COLOR_MED = '#f1f8e9'   # Light green
CONFIDENCE_COLOR_LOW = '#f5f5f5'   # Gray

class DynamicStandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None
    def load(self, filepath: str):
        npz = np.load(filepath)
        self.mean = npz['mean']
        self.scale = npz['scale']
        self.scale[self.scale == 0] = 1.0
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.scale

@keras.saving.register_keras_serializable(package="CustomLoss", name="quantile_loss_75")
def quantile_loss_75(y_true, y_pred):
    q = 0.75
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error), axis=-1)

def parse_v3_from_tfrecord(tfrecord_path):
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    feature_desc = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "stock_id": tf.io.FixedLenFeature([], tf.string),
        "stock_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "close": tf.io.FixedLenFeature([], tf.float32),
        "vwap": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    for prefix in ["foreign", "trust", "dealer", "foreign_dealer", "dealer_hedge"]:
        feature_desc[f"{prefix}_net_buy"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        feature_desc[f"{prefix}_buy"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        feature_desc[f"{prefix}_sell"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        feature_desc[f"{prefix}_cost"] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

    columns = list(feature_desc.keys())
    values_buffer = {col: [] for col in columns}
    for batch_raw in raw_ds.batch(4096):
        parsed = tf.io.parse_example(batch_raw, feature_desc)
        for col in columns: values_buffer[col].append(parsed[col].numpy())
        
    data_dict = {}
    string_cols = {"date", "stock_id", "stock_name"}
    for col in columns:
        arr = np.concatenate(values_buffer[col], axis=0)
        if col in string_cols:
            if arr.dtype.kind in ("S", "a"): data_dict[col] = np.char.decode(arr, "utf-8", errors="replace")
            else: data_dict[col] = np.array([x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in arr])
        else: data_dict[col] = arr
    return pd.DataFrame(data_dict)

def _weighted_average_predictions(predictions: list, method: str = "exponential") -> float:
    """Calculate weighted average of predictions with time decay."""
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) == 0:
        return float("nan")
    
    # Select weighting scheme
    if method == "exponential":
        weights = np.array([16, 8, 4, 2, 1], dtype=np.float64)
    elif method == "linear":
        weights = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    elif method == "recency":
        weights = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    else:
        weights = np.ones(5, dtype=np.float64) / 5.0
    
    n_valid = len(finite_preds)
    if n_valid < len(weights):
        weights = weights[-n_valid:]
    weights_normalized = weights[:n_valid] / weights[:n_valid].sum()
    return float(np.dot(finite_preds, weights_normalized))

def normalize_stock_id(value) -> str:
    """Normalize stock ID by removing trailing .0 and whitespace."""
    sid = str(value).strip()
    if sid.endswith(".0"):
        sid = sid[:-2]
    return sid


def _safe_log_return_pct(curr_price: float, prev_price: float) -> float:
    if not np.isfinite(curr_price) or not np.isfinite(prev_price):
        return float("nan")
    if curr_price <= 0.0 or prev_price <= 0.0:
        return float("nan")
    return float(np.log(curr_price / prev_price) * 100.0)


def _safe_simple_return_pct(curr_price: float, prev_price: float) -> float:
    if not np.isfinite(curr_price) or not np.isfinite(prev_price):
        return float("nan")
    if curr_price <= 0.0 or prev_price <= 0.0:
        return float("nan")
    return float((curr_price / prev_price - 1.0) * 100.0)


def _safe_price_from_return_pct(prev_price: float, return_pct: float) -> float:
    if not np.isfinite(prev_price) or not np.isfinite(return_pct):
        return float("nan")
    if prev_price <= 0.0:
        return float("nan")
    exp_arg: float = float(np.clip(return_pct / 100.0, -50.0, 50.0))
    out: float = float(prev_price * np.exp(exp_arg))
    return out if np.isfinite(out) else float("nan")


def _robust_symmetric_ylim(*arrays: np.ndarray, default: float = 10.0) -> float:
    finite_chunks: list[np.ndarray] = []
    for arr in arrays:
        arr_np: np.ndarray = np.asarray(arr, dtype=np.float64).ravel()
        finite: np.ndarray = arr_np[np.isfinite(arr_np)]
        if finite.size > 0:
            finite_chunks.append(np.abs(finite))

    if not finite_chunks:
        return default

    merged: np.ndarray = np.concatenate(finite_chunks)
    bound: float = float(np.nanpercentile(merged, 99.0))
    if not np.isfinite(bound) or bound <= 0.0:
        return default
    return max(bound, 1.0)


def _decode_model_output_log_return(pred_value: float) -> float:
    return float(pred_value)


def _consensus_confidence(predictions: list) -> float:
    """Calculate consensus confidence based on prediction standard deviation."""
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) <= 1:
        return DEFAULT_CONFIDENCE
    
    std = float(np.std(finite_preds))
    if not np.isfinite(std) or std < 0:
        return DEFAULT_CONFIDENCE
    return float(1.0 / (1.0 + std))


def get_confidence_color(confidence: float, threshold: float) -> str:
    """Get background color based on confidence level."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return CONFIDENCE_COLOR_HIGH
    elif confidence >= threshold:
        return CONFIDENCE_COLOR_MED
    return CONFIDENCE_COLOR_LOW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tfrecord-path", type=str, required=True)
    parser.add_argument("--scaler-path", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--vol-factor", type=float, default=1.0)
    parser.add_argument("--ensemble-method", type=str, default="exponential",
                        choices=["exponential", "linear", "recency", "none"])
    parser.add_argument("--ensemble-future", type=bool, default=True)
    parser.add_argument("--show-individual-preds", type=bool, default=False)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()

    custom_objects = {"CustomLoss>quantile_loss_75": quantile_loss_75, "quantile_loss_75": quantile_loss_75}
    model = keras.models.load_model(args.model_path, custom_objects=custom_objects)
    
    scaler = DynamicStandardScaler()
    scaler.load(args.scaler_path)

    all_df = parse_v3_from_tfrecord(args.tfrecord_path)
    all_df["stock_id"] = all_df["stock_id"].map(normalize_stock_id)
    # 物理除噪：排除 ETF 與特股
    all_df = all_df[~all_df['stock_id'].str.startswith('00')]
    all_df = all_df[~all_df['stock_id'].str.contains(r'[A-Za-z]', na=False)]
    all_df = all_df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    total_net = sum(all_df[f"{p}_net_buy"] * DEFAULT_WEIGHTS[p] for p in DEFAULT_WEIGHTS)
    total_vol = sum((all_df[f"{p}_buy"] + all_df[f"{p}_sell"]) * DEFAULT_WEIGHTS[p] for p in DEFAULT_WEIGHTS)
    all_df["conviction_intensity"] = total_net / (total_vol + 1e-5)

    for prefix in ["foreign", "trust", "dealer", "foreign_dealer", "dealer_hedge"]:
        all_df[f"{prefix}_cost_dev"] = (all_df["vwap"] - all_df[f"{prefix}_cost"]) / (all_df[f"{prefix}_cost"] + 1e-8)
        
    all_df["vwap_change_pct"] = all_df.groupby("stock_id")["vwap"].pct_change().fillna(0)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_cols = list(SELECTED_FEATURES)
    output_dir = Path(__file__).parent / "predict_plot"
    output_dir.mkdir(exist_ok=True)

    all_prediction_results = []
    print(f"開始掃描全市場個股...")
    
    for stock_id, stock_group in all_df.groupby("stock_id"):
        stock_group = stock_group.reset_index(drop=True)
        if len(stock_group) <= args.window_size:
            continue

        all_windows, valid_indices = [], []
        for i in range(args.window_size, len(stock_group)):
            window_data = stock_group.iloc[i - args.window_size : i][feature_cols].values
            scaled_window = np.clip(scaler.transform(window_data), -5.0, 5.0)
            all_windows.append(scaled_window)
            valid_indices.append(i)

        if not all_windows:
            continue

        preds_denorm = model.predict(np.array(all_windows), verbose=0)

        # Initialize prediction columns
        stock_group["pred_returns_all"] = [[] for _ in range(len(stock_group))]
        stock_group["pred_return_consensus"] = np.nan
        stock_group["pred_confidence"] = np.nan
        stock_group["pred_return_d1"] = np.nan
        stock_group["actual_return_d1"] = np.nan
        stock_group["pred_price_d1"] = np.nan

        # Process predictions (use .at[] for list assignments)
        for idx_pos, pred_values in zip(valid_indices, preds_denorm):
            all_predictions = [
                _decode_model_output_log_return(float(pv)) * args.vol_factor
                for pv in pred_values
            ]
            
            consensus_pred = (
                all_predictions[0] if args.ensemble_method == "none"
                else _weighted_average_predictions(all_predictions, method=args.ensemble_method)
            )

            stock_group.at[idx_pos, "pred_returns_all"] = all_predictions
            stock_group.at[idx_pos, "pred_return_consensus"] = consensus_pred
            stock_group.at[idx_pos, "pred_confidence"] = _consensus_confidence(all_predictions)
            stock_group.at[idx_pos, "pred_return_d1"] = consensus_pred
            
            if idx_pos > 0:
                stock_group.at[idx_pos, "actual_return_d1"] = _safe_simple_return_pct(
                    float(stock_group.at[idx_pos, "close"]),
                    float(stock_group.at[idx_pos - 1, "close"])
                )
                stock_group.at[idx_pos, "pred_price_d1"] = _safe_price_from_return_pct(
                    float(stock_group.at[idx_pos - 1, "close"]),
                    float(consensus_pred)
                )

        # Future 5-day predictions with ensemble
        last_hist_price = float(stock_group["close"].iloc[-1])
        future_preds_ensemble = {f"d{i}": [] for i in range(1, FUTURE_DAYS + 1)}
        
        if args.ensemble_future:
            for offset in range(min(FUTURE_DAYS, args.window_size)):
                window_start = len(stock_group) - args.window_size - offset
                if window_start < 0:
                    break
                future_window = stock_group.iloc[window_start : window_start + args.window_size][feature_cols].values
                future_window_scaled = np.clip(scaler.transform(future_window), -5.0, 5.0)
                future_preds_tmp = model.predict(np.array([future_window_scaled]), verbose=0)[0]
                for k in range(FUTURE_DAYS):
                    future_preds_ensemble[f"d{k+1}"].append(float(future_preds_tmp[k]))
        else:
            future_window = stock_group.iloc[-args.window_size:][feature_cols].values
            future_window_scaled = np.clip(scaler.transform(future_window), -5.0, 5.0)
            future_preds_tmp = model.predict(np.array([future_window_scaled]), verbose=0)[0]
            for k in range(FUTURE_DAYS):
                future_preds_ensemble[f"d{k+1}"].append(float(future_preds_tmp[k]))

        future_preds_consensus = []
        future_confidences = []
        for k in range(1, FUTURE_DAYS + 1):
            preds_for_day = [p * args.vol_factor for p in future_preds_ensemble[f"d{k}"]]
            future_preds_consensus.append(
                _weighted_average_predictions(preds_for_day, method=args.ensemble_method)
            )
            future_confidences.append(_consensus_confidence(preds_for_day))

        trend_val = (np.exp(float(future_preds_consensus[FUTURE_DAYS - 1]) / 100.0) - 1.0) * 100.0
        all_prediction_results.append({
            "stock_id": stock_id,
            "stock_name": str(stock_group["stock_name"].iloc[-1]).strip() if "stock_name" in stock_group.columns else "",
            "trend_pct": trend_val,
            "future_preds": future_preds_consensus,
            "future_confidences": future_confidences,
            "df_group": stock_group
        })

    if not all_prediction_results:
        print("未產出任何預測結果。")
        return
    results_df = pd.DataFrame(all_prediction_results).sort_values("trend_pct", ascending=False).reset_index(drop=True)

    plot_history_len = DEFAULT_PLOT_HISTORY_LEN
    print(f"\n✅ 掃描完成！正在生成 Top {args.top_n} 強勢股之 {plot_history_len} 天時序全對齊看板...")
    
    for rank_idx, row in results_df.head(args.top_n).iterrows():
        stock_id_str = normalize_stock_id(row["stock_id"])
        stock_name_str = row["stock_name"]
        stock_group = row["df_group"]
        trend_val = row["trend_pct"]
        future_cum_log = row["future_preds"]
        future_confidences = row["future_confidences"]

        plot_data = stock_group.tail(plot_history_len).copy().reset_index(drop=True)
        
        # Format dates properly for x-axis labels (avoid OutOfBoundsDatetime)
        dates_hist = pd.to_datetime(plot_data["date"]).dt.strftime('%Y-%m-%d').tolist()
        actual_prices_hist = plot_data["close"].values
        predicted_prices_hist = plot_data["pred_price_d1"].values

        # Convert log returns to simple returns for display
        pred_log_ret_series = plot_data["pred_return_d1"].astype(float)
        hist_pred_ret = np.expm1(pred_log_ret_series / 100.0) * 100.0
        hist_pred_ret = np.where(np.isfinite(hist_pred_ret), hist_pred_ret, np.nan)
        hist_act_ret = plot_data["actual_return_d1"].values

        future_daily_log = [future_cum_log[0]] + [future_cum_log[j] - future_cum_log[j - 1] for j in range(1, 5)]
        future_forecast_returns = [(np.exp(d / 100.0) - 1.0) * 100.0 for d in future_daily_log]
        future_prices = [
            _safe_price_from_return_pct(actual_prices_hist[-1], float(r))
            for r in future_cum_log
        ]

        all_x_labels = dates_hist + ["未來 D+1", "未來 D+2", "未來 D+3", "未來 D+4", "未來 D+5"]
        x_indices = np.arange(len(all_x_labels))
        bar_width = 0.25

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [4, 5]})
        fig.suptitle(f"Rank {rank_idx+1} Strong Stock: {stock_id_str} ({stock_name_str}) V3 集成預測決策看板", fontsize=15, fontweight='bold')

        ax1.plot(x_indices[:plot_history_len], actual_prices_hist, marker='o', label='實際收盤價', color='#4a81ad', markersize=4)
        ax1.plot(x_indices[:plot_history_len], predicted_prices_hist, marker='x', linestyle='--', label='歷史 D+1 預測價', color='#f0a05e', markersize=5, alpha=0.7)
        future_x = [x_indices[plot_history_len - 1]] + list(x_indices[plot_history_len:])
        future_y = [actual_prices_hist[-1]] + future_prices
        ax1.plot(future_x, future_y, marker='*', linestyle='-', label='未來 5 日推估價預報', color='#d62728', linewidth=2, markersize=9)
        for xi, v in zip(x_indices[plot_history_len:], future_prices):
            ax1.text(xi, v, f'{v:.1f}', color='#d62728', fontsize=9, fontweight='bold', ha='center', va='bottom')
        ax1.set_ylabel('股價 (元)')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left')

        # Process confidence and consensus data
        hist_confidence = plot_data['pred_confidence'].fillna(DEFAULT_CONFIDENCE).values
        hist_consensus_ret = plot_data['pred_return_consensus'].apply(
            lambda x: (np.exp(x / 100.0) - 1.0) * 100.0 if np.isfinite(x) else np.nan
        ).values

        # Draw confidence backgrounds for historical data
        for j in range(plot_history_len):
            conf = hist_confidence[j]
            ax2.axvspan(x_indices[j] - 0.5, x_indices[j] + 0.5, 
                       alpha=0.2, color=get_confidence_color(conf, args.confidence_threshold), zorder=0)

        # Draw confidence backgrounds for future predictions
        for j, conf in enumerate(future_confidences):
            idx = plot_history_len + j
            ax2.axvspan(x_indices[idx] - 0.5, x_indices[idx] + 0.5, 
                       alpha=0.2, color=get_confidence_color(conf, args.confidence_threshold), zorder=0)

        ax2.bar(x_indices[:plot_history_len] - bar_width, hist_act_ret, bar_width, label='歷史實際 D+1 回報 %', color='#9999ff', alpha=0.8, zorder=3)
        ax2.bar(x_indices[:plot_history_len], hist_pred_ret, bar_width, label='歷史預測 D+1 回報 %', color='#ff9999', alpha=0.8, zorder=3)
        ax2.bar(x_indices[:plot_history_len] + bar_width, hist_consensus_ret, bar_width, label='歷史共識預測 %', color='#4caf50', alpha=0.8, zorder=3)
        future_colors = ['#ff3333' if r >= 0 else '#00aa00' for r in future_forecast_returns]
        rects_future = ax2.bar(x_indices[plot_history_len:], future_forecast_returns, bar_width * 2, label='未來逐日預期報酬 % (共識)', color=future_colors, edgecolor='black', alpha=0.9, zorder=3)
        for bar, r in zip(rects_future, future_forecast_returns):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + (0.1 if h >= 0 else -0.4), f"{r:+.2f}%", ha='center', va='bottom' if h >= 0 else 'top', fontsize=10, fontweight='bold', zorder=4)

        if args.show_individual_preds:
            for idx, pred_list in enumerate(plot_data['pred_returns_all']):
                if isinstance(pred_list, list) and len(pred_list) == 5:
                    x_pos = x_indices[idx]
                    for k, pred in enumerate(pred_list):
                        pred_simple = (np.exp(pred / 100.0) - 1.0) * 100.0 if np.isfinite(pred) else np.nan
                        if np.isfinite(pred_simple):
                            x_scatter = x_pos + (k - 2) * 0.08
                            ax2.scatter(x_scatter, pred_simple, s=20, alpha=0.4, color='#999999', zorder=2)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(all_x_labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('報酬率 (%)')
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)

        y_bound = _robust_symmetric_ylim(
            np.asarray(hist_act_ret, dtype=np.float64),
            np.asarray(hist_pred_ret, dtype=np.float64),
            np.asarray(hist_consensus_ret, dtype=np.float64),
            np.asarray(future_forecast_returns, dtype=np.float64),
            default=10.0,
        )
        ax2.set_ylim(-y_bound, y_bound)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax2.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        safe_name = ''.join(c for c in stock_name_str.strip() if c not in r'\\/*?:"<>|')
        filename = f"rank_{rank_idx+1:02d}_v3_trend_{trend_val:+.1f}_{stock_id_str}_{safe_name}.png"
        
        try:
            plt.savefig(output_dir / filename, dpi=300)
            print(f"排名 {rank_idx+1} 檔案已成功輸出: {filename}")
        except Exception as e:
            print(f"警告: 無法儲存 {filename}: {e}")
        finally:
            plt.close()

if __name__ == "__main__":
    main()