"""
使用訓練好的 Dilated CNN 模型進行全市場選股 (修復 Unicode 崩潰 + 40天精準全對齊版)
==================================================================================
全面批量推理全市場個股，挑選 Top 20 潛力回報股，並生成 40 天歷史日期完美咬合的看盤圖表。
主要功能：
1. 從 TFRecord 讀取全市場數據，進行特徵工程和標準化處理。
2. 使用訓練好的模型對每支股票進行預測，並解碼回實際的趨勢百分比。
3. 根據預測結果排序，選出 Top 20 強勢股。
4. 為每支 Top 20 股票生成一張包含過去 40 天的法人買賣超趨勢和收盤價的圖表，並在圖表上標註預測的未來趨勢百分比。

使用說明：
```powershell
python institutional_net_buy_predict_visual_all.py `
    --model-path institutional_net_buy_v2_dilated.keras `
    --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
    --stats-path institutional_net_buy_2024-05-22_2026-05-22.leakage_fixed.stats.json `
    --window-size 20 `
    --top-n 50
```

注意事項：
- 確保提供的 TFRecord 文件包含正確格式的數據，並且 stats JSON 文件與 TFRecord 中的股票 ID 完全對應。
- 圖表將保存為 PNG 格式，保存在腳本所在的 `predict_plot` 資料夾中。
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import tensorflow as tf
import keras
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_WEIGHTS = {
    "foreign": 1.0, "foreign_dealer": 1.0, "trust": 1.0,
    "dealer": 0.5, "dealer_hedge": 0.5, "retail": -0.2
}

def normalize_stock_id(value):
    sid = str(value).strip()
    if sid.endswith(".0"): sid = sid[:-2]
    return sid

def get_slope(y):
    if len(y) < 2: return 0.0
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.polyfit(np.arange(len(y_clean)), y_clean, 1)[0])

def parse_all_from_tfrecord(tfrecord_path):
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    feature_desc = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "stock_id": tf.io.FixedLenFeature([], tf.string),
        "stock_name": tf.io.FixedLenFeature([], tf.string),
        "close": tf.io.FixedLenFeature([], tf.float32),
        "foreign_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "foreign_dealer_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "trust_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "dealer_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "dealer_hedge_net_buy": tf.io.FixedLenFeature([], tf.int64),
        "foreign_cost": tf.io.FixedLenFeature([], tf.float32),
        "foreign_dealer_cost": tf.io.FixedLenFeature([], tf.float32),
        "trust_cost": tf.io.FixedLenFeature([], tf.float32),
        "dealer_cost": tf.io.FixedLenFeature([], tf.float32),
        "dealer_hedge_cost": tf.io.FixedLenFeature([], tf.float32),
    }
    columns = list(feature_desc.keys())
    values_buffer = {col: [] for col in columns}
    for batch_raw in raw_ds.batch(4096):
        parsed = tf.io.parse_example(batch_raw, feature_desc)
        for col in columns: values_buffer[col].append(parsed[col].numpy())
        
    # 🔥 核心修復：高防禦力中文位元組解碼器
    data_dict = {}
    string_cols = {"date", "stock_id", "stock_name"}
    for col in columns:
        arr = np.concatenate(values_buffer[col], axis=0)
        if col in string_cols:
            if arr.dtype.kind in ("S", "a"):
                data_dict[col] = np.char.decode(arr, "utf-8", errors="replace")
            elif arr.dtype.kind == "U":
                data_dict[col] = arr
            elif arr.dtype.kind == "O":
                data_dict[col] = np.array([
                    x.decode("utf-8", errors="replace") if isinstance(x, (bytes, bytearray, np.bytes_))
                    else (x if isinstance(x, str) else ("" if x is None else str(x)))
                    for x in arr
                ], dtype=object)
            else:
                try: data_dict[col] = arr.astype(str)
                except Exception: data_dict[col] = np.array([str(x) for x in arr], dtype=object)
        else: data_dict[col] = arr
    return pd.DataFrame(data_dict)


def _decode_model_output_log_return(pred_value: float, stock_stats: dict) -> float:
    """
    Decode model output back to log-return percent using stored target transform metadata.
    """
    mode = str(stock_stats.get("target_transform", "clip"))
    value = float(pred_value)

    if mode == "tanh":
        scale = float(stock_stats.get("target_tanh_scale", 50.0))
        if not np.isfinite(scale) or scale <= 0.0:
            return value
        ratio = np.clip(value / scale, -0.999999, 0.999999)
        return float(np.arctanh(ratio) * scale)

    # clip / none / unknown: identity decode for backward compatibility.
    return value

@keras.saving.register_keras_serializable(package="CustomLoss", name="quantile_loss_75")
def quantile_loss_75(y_true, y_pred):
    """為了順利載入使用 quantile loss 訓練的模型"""
    q = 0.75
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error), axis=-1)


# ==================================================================================
# 🔥 聚合函數模塊：實現 5 預測共識機制（與 predict_visual.py 同步）
# ==================================================================================

def _weighted_average_predictions(predictions: list, method: str = "exponential") -> float:
    """計算加權平均預測（時間衰減）"""
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) == 0:
        return float("nan")
    
    if method == "exponential":
        weights = np.array([16, 8, 4, 2, 1], dtype=np.float64)
    elif method == "linear":
        weights = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    elif method == "recency":
        weights = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    else:
        weights = np.array([1, 1, 1, 1, 1], dtype=np.float64) / 5.0
    
    n_valid = len(finite_preds)
    if n_valid < len(weights):
        weights = weights[-n_valid:]
    
    weights_normalized = weights[:n_valid] / weights[:n_valid].sum()
    return float(np.dot(finite_preds, weights_normalized))


def _consensus_confidence(predictions: list) -> float:
    """計算預測共識強度（基於標準差）"""
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) <= 1:
        return 0.5
    
    std = float(np.std(finite_preds))
    if not np.isfinite(std) or std < 0:
        return 0.5
    
    return float(1.0 / (1.0 + std))


def _direction_vote(predictions: list) -> tuple:
    """投票預測方向（看漲/看跌）"""
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) == 0:
        return 0, "neutral"
    
    bullish_count = (finite_preds > 0).sum()
    total = len(finite_preds)
    
    if bullish_count >= total * 0.6:
        direction = "bullish"
    elif bullish_count <= total * 0.4:
        direction = "bearish"
    else:
        direction = "neutral"
    
    return int(bullish_count), direction


def main():
    parser = argparse.ArgumentParser(description="Market Scanner with 40-Day Aligned Validation")
    parser.add_argument("--model-path", type=str, default="institutional_net_buy_v2_dilated.keras")
    parser.add_argument("--tfrecord-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--vol-factor", type=float, default=1.0, help="波動擴展係數")
    # 🔥 新增共識聚合參數
    parser.add_argument("--ensemble-method", type=str, default="exponential", 
                        choices=["exponential", "linear", "recency", "none"],
                        help="歷史預測聚合策略")
    parser.add_argument("--ensemble-future", type=bool, default=True,
                        help="未來 5 天預測是否使用多窗口聚合")
    parser.add_argument("--show-individual-preds", type=bool, default=False,
                        help="下圖是否顯示個別預測散點")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="信心度閾值")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到模型 {args.model_path}")
        return

    # 建立 custom_objects 以確保載入時能找到 CustomLoss>quantile_loss_75
    custom_objects = {"CustomLoss>quantile_loss_75": quantile_loss_75, "quantile_loss_75": quantile_loss_75}
    model = keras.models.load_model(args.model_path, custom_objects=custom_objects)

    stats_path = args.stats_path or args.tfrecord_path.replace(".tfrecord", ".stats.json")
    with open(stats_path, "r", encoding="utf-8") as f: stats_data = json.load(f)
    stats_index = {normalize_stock_id(k): k for k in stats_data.keys()}

    target_ids = set()
    targets_file = Path(__file__).parent / "targets.txt"
    if targets_file.exists():
        with open(targets_file, "r", encoding="utf-8") as f:
            target_ids = {normalize_stock_id(line) for line in f if line.strip()}

    all_df = parse_all_from_tfrecord(args.tfrecord_path)
    all_df["stock_id"] = all_df["stock_id"].map(normalize_stock_id)
    all_df = all_df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    all_df["retail_nb"] = -(all_df["foreign_net_buy"] + all_df["foreign_dealer_net_buy"] + all_df["trust_net_buy"] + all_df["dealer_net_buy"] + all_df["dealer_hedge_net_buy"])
    all_df["w_buy"] = (all_df["foreign_net_buy"] * DEFAULT_WEIGHTS["foreign"] + all_df["foreign_dealer_net_buy"] * DEFAULT_WEIGHTS["foreign_dealer"] +
                       all_df["trust_net_buy"] * DEFAULT_WEIGHTS["trust"] + all_df["dealer_net_buy"] * DEFAULT_WEIGHTS["dealer"] +
                       all_df["dealer_hedge_net_buy"] * DEFAULT_WEIGHTS["dealer_hedge"] + all_df["retail_nb"] * DEFAULT_WEIGHTS["retail"])

    all_df["sig_total"] = all_df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(10).sum())
    all_df["sig_pos_ratio"] = all_df.groupby("stock_id")["w_buy"].transform(lambda x: (x > 0).astype(float).rolling(10).mean())
    all_df["sig_slope"] = all_df.groupby("stock_id")["w_buy"].transform(lambda x: x.cumsum().rolling(10).apply(get_slope, raw=True))
    all_df["sig_baseline"] = all_df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(20).mean().shift(10))
    all_df["sig_price_pct"] = all_df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(10))
    cost_cols = ["foreign_cost", "trust_cost", "dealer_cost", "dealer_hedge_cost", "foreign_dealer_cost"]
    for c in cost_cols: all_df[c] = (all_df[c] - all_df["close"]) / (all_df["close"] + 1e-8)

    feature_cols = ["w_buy", "sig_total", "sig_pos_ratio", "sig_slope", "sig_baseline", "sig_price_pct"] + cost_cols
    all_df = all_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    output_dir = Path(__file__).parent / "predict_plot"
    output_dir.mkdir(exist_ok=True)

    all_prediction_results = []
    for sid, group in all_df.groupby("stock_id"):
        sid_str = normalize_stock_id(sid)
        stats_key = stats_index.get(sid_str)
        if not stats_key: continue
        g = group.reset_index(drop=True)
        if len(g) <= args.window_size: continue

        s_mean = np.array(stats_data[stats_key]["mean"])
        s_std = np.array(stats_data[stats_key]["std"])
        safe_std = np.where(np.isfinite(s_std) & (s_std != 0.0), s_std, 1.0)

        all_windows, valid_indices = [], []
        for i in range(len(g)):
            if i < args.window_size: continue
            window_data = g.iloc[i - args.window_size : i][feature_cols].values
            # 使用 Z-Score 標準化，並進行寬鬆的離群值處理
            scaled_window = (window_data - s_mean) / safe_std
            scaled_window = np.clip(scaled_window, -10, 10)
            all_windows.append(scaled_window)
            valid_indices.append(i)

        if not all_windows: continue
        preds_denorm = model.predict(np.array(all_windows), verbose=0)
        
        # 【修復真正的 bug】preds_denorm[-1] 的 window 截至倒數第二天，必須用最後 window_size 天建立專用 future window
        future_window = g.iloc[-args.window_size:][feature_cols].values
        future_window_scaled = (future_window - s_mean) / safe_std
        future_window_scaled = np.clip(future_window_scaled, -10, 10)
        future_preds = model.predict(np.array([future_window_scaled]), verbose=0)[0]
        
        last_close = float(g["close"].iloc[-1])
        
        stock_stats = stats_data[stats_key]
        decoded_future_preds = np.array([
            _decode_model_output_log_return(float(v), stock_stats) for v in future_preds
        ], dtype=np.float64)

        all_prediction_results.append({
            "stock_id": sid_str, "stock_name": str(g['stock_name'].iloc[-1]).strip(),
            "last_close": last_close, "trend_pct": (np.exp(float(decoded_future_preds[4]) * args.vol_factor / 100.0) - 1.0) * 100.0,
            "df_group": g, "preds_denorm": preds_denorm, "valid_indices": valid_indices,
            "future_preds": decoded_future_preds, "stock_stats": stock_stats,
            "s_mean": s_mean, "s_std": s_std,
        })

    if not all_prediction_results: return
    results_df = pd.DataFrame(all_prediction_results).sort_values("trend_pct", ascending=False).reset_index(drop=True)

    plot_history_len = 40
    print(f"\n正在生成 Top {args.top_n} 強勢股之 40 天時序全對齊看板...")
    for i, row in results_df.head(args.top_n).iterrows():
        sid_str = row["stock_id"]
        stock_name_str = row["stock_name"]
        g = row["df_group"]
        preds_denorm = row["preds_denorm"]
        valid_indices = row["valid_indices"]
        trend_val = row["trend_pct"]
        future_preds = row["future_preds"]
        stock_stats = row["stock_stats"]

        g["pred_returns_all"] = [[]] * len(g)  # 🔥 新增：存儲每天的所有 5 個預測值
        g["pred_return_consensus"] = np.nan  # 🔥 新增：聚合後的共識預測
        g["pred_confidence"] = np.nan  # 🔥 新增：共識信心度
        g["pred_return_d1"] = np.nan
        g["actual_return_d1"] = np.nan
        g["pred_price_d1"] = np.nan
        
        for idx_pos, p_val in zip(valid_indices, preds_denorm):
            # 🔥 收集所有 5 個預測值（解碼後）
            all_predictions = [
                _decode_model_output_log_return(float(pv), stock_stats) * args.vol_factor
                for pv in p_val
            ]
            g.at[g.index[idx_pos], "pred_returns_all"] = all_predictions
            
            # 🔥 根據 ensemble_method 聚合預測
            if args.ensemble_method == "none":
                consensus_pred = all_predictions[0]
            else:
                consensus_pred = _weighted_average_predictions(all_predictions, method=args.ensemble_method)
            
            g.at[g.index[idx_pos], "pred_return_consensus"] = consensus_pred
            g.at[g.index[idx_pos], "pred_confidence"] = _consensus_confidence(all_predictions)
            g.at[g.index[idx_pos], "pred_return_d1"] = consensus_pred
            
            # window 截至 idx_pos-1
            curr = float(g["close"].iloc[idx_pos])
            prev = float(g["close"].iloc[idx_pos - 1])
            act_ret = float((curr / prev - 1.0) * 100.0) if (np.isfinite(curr) and np.isfinite(prev) and curr > 0 and prev > 0) else float("nan")
            g.at[g.index[idx_pos], "actual_return_d1"] = act_ret
            # pred_price_d1
            exp_arg = float(np.clip(consensus_pred / 100.0, -50.0, 50.0))
            pred_p = float(prev * np.exp(exp_arg)) if (np.isfinite(prev) and prev > 0) else float("nan")
            g.at[g.index[idx_pos], "pred_price_d1"] = pred_p if np.isfinite(pred_p) else float("nan")

        # 對齊切片
        g_plot = g.tail(plot_history_len).copy().reset_index(drop=True)
        dates_hist = list(g_plot['date'].values)
        actual_prices_hist = g_plot['close'].values
        predicted_prices_hist = g_plot['pred_price_d1'].values
        # 口徑對齊：歷史預測柱狀圖使用模型當日預測的 D+1 對數報酬，轉為簡單報酬 %
        # simple_return(%) = exp(log_return/100) - 1，再乘 100
        pred_log_ret_series = g_plot["pred_return_d1"].astype(float)
        hist_pred_ret = np.expm1(pred_log_ret_series / 100.0) * 100.0
        hist_pred_ret = np.asarray(hist_pred_ret, dtype=np.float64)
        hist_pred_ret = np.where(np.isfinite(hist_pred_ret), hist_pred_ret, np.nan)
        hist_act_ret = g_plot["actual_return_d1"].values

        # 未來 5 天預報：方案 A 逐日簡單報酬
        # 🔥 改進：使用多個窗口位置聚合未來預測
        last_hist_price = float(g["close"].iloc[-1])
        
        if args.ensemble_future:
            # 從不同窗口起點進行預測
            future_preds_ensemble = {"d1": [], "d2": [], "d3": [], "d4": [], "d5": []}
            
            for offset in range(min(5, args.window_size)):
                window_start = len(g) - args.window_size - offset
                if window_start < 0:
                    break
                
                future_window = g.iloc[window_start : window_start + args.window_size][feature_cols].values
                future_window_scaled = (future_window - row["s_mean"]) / np.where(np.isfinite(row["s_std"]) & (row["s_std"] != 0.0), row["s_std"], 1.0)
                future_window_scaled = np.clip(future_window_scaled, -10, 10)
                future_preds_tmp = model.predict(np.array([future_window_scaled]), verbose=0)[0]
                
                for k in range(5):
                    future_preds_ensemble[f"d{k+1}"].append(float(future_preds_tmp[k]))
            
            # 對每個未來日期聚合預測
            future_preds_consensus = []
            future_confidences = []
            for k in range(1, 6):
                key = f"d{k}"
                preds_for_day = [
                    _decode_model_output_log_return(float(p), stock_stats) * args.vol_factor
                    for p in future_preds_ensemble[key]
                ]
                consensus = _weighted_average_predictions(preds_for_day, method=args.ensemble_method)
                confidence = _consensus_confidence(preds_for_day)
                future_preds_consensus.append(consensus)
                future_confidences.append(confidence)
        else:
            # 單窗口預測（原始邏輯）
            future_preds_consensus = [float(r) * args.vol_factor for r in future_preds]
            future_confidences = [0.5] * 5
        
        # 計算逐日簡單報酬
        future_cum_log = future_preds_consensus
        future_daily_log = [future_cum_log[0]] + [future_cum_log[i] - future_cum_log[i - 1] for i in range(1, 5)]
        future_forecast_returns = [(np.exp(d / 100.0) - 1.0) * 100.0 for d in future_daily_log]
        future_prices = [
            actual_prices_hist[-1] * np.exp(float(r) / 100.0)
            for r in future_cum_log
        ]
        future_dates = [(pd.to_datetime(dates_hist[-1]) + pd.tseries.offsets.BDay(j)).strftime('%Y-%m-%d') for j in range(1, 6)]

        all_x_labels = dates_hist + ["未來 D+1", "未來 D+2", "未來 D+3", "未來 D+4", "未來 D+5"]
        x_indices = np.arange(len(all_x_labels))
        bar_width = 0.25

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [4, 5]})
        
        # 上圖
        ax1.plot(x_indices[:plot_history_len], actual_prices_hist, marker='o', label='實際收盤價', color='#4a81ad', markersize=4)
        ax1.plot(x_indices[:plot_history_len], predicted_prices_hist, marker='x', linestyle='--', label='歷史 D+1 預測價', color='#f0a05e', markersize=5, alpha=0.7)
        ax1.plot([x_indices[plot_history_len-1]] + list(x_indices[plot_history_len:]), [actual_prices_hist[-1]] + future_prices, marker='*', linestyle='-', label='未來 5 日推估價', color='#d62728', linewidth=2, markersize=9)
        for xi, v in zip(x_indices[plot_history_len:], future_prices): ax1.text(xi, v, f'{v:.1f}', color='#d62728', fontsize=9, fontweight='bold', ha='center', va='bottom')
        ax1.set_title(f"Rank {i+1} | {sid_str} {stock_name_str} | 絕對股價歷史比對")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left')

        # 下圖：報酬率驗證與預測 (垂直完全咬合 + 共識聚合)
        # 🔥 準備信心度背景和共識預測
        hist_consensus_ret = []
        hist_confidence = []
        
        for idx in g_plot.index:
            conf_val = g_plot.loc[idx, "pred_confidence"]
            cons_ret = g_plot.loc[idx, "pred_return_consensus"]
            
            cons_simple = (np.exp(float(cons_ret) / 100.0) - 1.0) * 100.0 if np.isfinite(cons_ret) else np.nan
            
            hist_consensus_ret.append(cons_simple)
            hist_confidence.append(conf_val if np.isfinite(conf_val) else 0.5)
        
        hist_consensus_ret = np.asarray(hist_consensus_ret, dtype=np.float64)
        hist_confidence = np.asarray(hist_confidence, dtype=np.float64)
        
        # 繪製信心度背景色區間（整個歷史部分）
        for idx_i in range(plot_history_len):
            conf = hist_confidence[idx_i]
            if conf >= args.confidence_threshold:
                if conf >= 0.7:
                    bg_color = "#e8f5e9"  # 深綠背景
                else:
                    bg_color = "#f1f8e9"  # 淡綠背景
            else:
                bg_color = "#f5f5f5"  # 灰色背景
            
            ax2.axvspan(x_indices[idx_i] - 0.5, x_indices[idx_i] + 0.5, alpha=0.2, color=bg_color, zorder=0)
        
        # 未來部分信心度背景
        for idx_i, conf in enumerate(future_confidences):
            idx_x = plot_history_len + idx_i
            if conf >= args.confidence_threshold:
                if conf >= 0.7:
                    bg_color = "#e8f5e9"
                else:
                    bg_color = "#f1f8e9"
            else:
                bg_color = "#f5f5f5"
            
            ax2.axvspan(x_indices[idx_x] - 0.5, x_indices[idx_x] + 0.5, alpha=0.2, color=bg_color, zorder=0)
        
        # 繪製柱狀圖
        ax2.bar(x_indices[:plot_history_len] - bar_width, hist_act_ret, bar_width, 
                label='歷史實際 D+1 回報 %', color='#9999ff', alpha=0.8, zorder=3)
        ax2.bar(x_indices[:plot_history_len], hist_pred_ret, bar_width, 
                label='歷史預測 D+1 回報 %', color='#ff9999', alpha=0.8, zorder=3)
        ax2.bar(x_indices[:plot_history_len] + bar_width, hist_consensus_ret, bar_width, 
                label='歷史共識預測 %', color='#4caf50', alpha=0.8, zorder=3)
        
        # 未來部分
        future_colors = ['#ff3333' if r >= 0 else '#00aa00' for r in future_forecast_returns]
        rects_future = ax2.bar(x_indices[plot_history_len:], future_forecast_returns, bar_width * 2, 
                               label='未來逐日預期報酬 % (共識)', color=future_colors, edgecolor='black', 
                               alpha=0.9, zorder=3)
        
        for bar, r in zip(rects_future, future_forecast_returns):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + (0.1 if h>=0 else -0.4), f"{r:+.2f}%", 
                    ha='center', va='bottom' if h>=0 else 'top', fontsize=10, fontweight='bold', zorder=4)
        
        # 🔥 可選：繪製個別預測散點
        if args.show_individual_preds:
            for idx_i, pred_list in enumerate(g_plot["pred_returns_all"]):
                if isinstance(pred_list, list) and len(pred_list) == 5:
                    x_pos = x_indices[idx_i]
                    for k, pred in enumerate(pred_list):
                        pred_simple = (np.exp(pred / 100.0) - 1.0) * 100.0 if np.isfinite(pred) else np.nan
                        if np.isfinite(pred_simple):
                            x_scatter = x_pos + (k - 2) * 0.08
                            ax2.scatter(x_scatter, pred_simple, s=20, alpha=0.4, color='#999999', zorder=2)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(all_x_labels, rotation=45, ha='right', fontsize=9)
        ax2.set_title(f"報酬率完美對齊驗證區 | 5D 累積預期回報: {trend_val:+.2f}%", fontsize=12)
        
        all_ret_vals = [v for v in list(hist_pred_ret) + list(hist_act_ret) + list(hist_consensus_ret) + future_forecast_returns if np.isfinite(v)]
        y_bound = (max(abs(v) for v in all_ret_vals) * 1.3) if all_ret_vals else 10.0
        y_bound = max(y_bound, 1.5)
        ax2.set_ylim(-y_bound, y_bound)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
        ax2.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        filename = f"rank_{i+1:02d}_trend_{trend_val:+.1f}_{sid_str}_{"".join(c for c in stock_name_str if c not in r'\\\\/*?:\"<>|').strip()}.png"
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()
        print(f"排名 {i+1} 檔案已成功輸出: {filename}")

if __name__ == "__main__":
    main()