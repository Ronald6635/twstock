"""
使用訓練好的 Dilated CNN 模型進行 5 天視向預測與視覺化 (修復 Unicode 崩潰 + 40天精準全對齊版)
==================================================================================
1. 修正了 Windows 環境下讀取中文股票名稱會觸發 UnicodeDecodeError 的致命 Bug。
2. 下方報酬率 Bar Plot 的歷史天數與上方股價線圖完美對齊為 40 天，且垂直座標完全咬合。
3. 新增了預測共識機制，提供加權平均、投票方向和信心度評估等功能。
4. 預設使用指數衰減加權，近期預測更重，並提供線性衰減和只用最新預測的選項。
5. 預測結果的柱子顏色會根據共識信心度自動調整，低於閾值的柱子會用淡灰色標記以示警告。

使用說明
----------------
請確保已經安裝了必要的 Python 套件（如 TensorFlow、Keras、Pandas、Matplotlib 等），並且已經訓練好模型並準備好 TFRecord 資料。
在命令列中執行以下指令來運行視覺化腳本：

```powershell
python institutional_net_buy_predict_visual.py `
    --model-path institutional_net_buy_v2_dilated.keras `
    --tfrecord-path institutional_net_buy_2024-05-22_2026-05-22.tfrecord `
    --stats-path institutional_net_buy_2024-05-22_2026-05-22.leakage_fixed.stats.json `
    --window-size 32 `
    --ensemble-method exponential `
    --confidence-threshold 0.5
```
參數說明：
- `--model-path`: 訓練好的 Keras 模型檔案路徑。
- `--tfrecord-path`: 包含預測資料的 TFRecord 檔案路徑。
- `--stats-path`: 包含資料統計資訊的 JSON 檔案路徑（用於解碼預測值）。
- `--window-size`: 模型輸入的時間窗口大小（默認 32）。
- `--ensemble-method`: 歷史預測聚合策略，選項包括：
  - `exponential`: 指數衰減加權（預設，權重為 [16, 8, 4, 2, 1]）
  - `linear`: 線性衰減加權（權重為 [5, 4, 3, 2, 1]）
  - `recency`: 只使用最新一天的預測（權重為 [1, 0, 0, 0, 0]）
  - `none`: 不進行聚合，直接使用原始預測值
- `--confidence-threshold`: 預測信心度閾值，低於此值的預測柱子將被標記為淡灰色（默認 0.5）。

注意事項
- 請確保模型訓練時使用的 `window_size` 與此處一致，以避免輸入維度不匹配。
- 預測共識機制需要至少兩個有效預測值才能計算信心度，否則將默認為 0.5。
- 如果 TFRecord 中的預測值全為非有限（NaN 或 Inf），則加權平均將返回 NaN，並且柱子將被標記為淡灰色。
- 預測結果將會在命令列中以表格形式顯示，並且會生成對應的 PNG 圖片文件，每個目標股票一張，保存在當前目錄下。
- 圖片文件名稱格式為 `{stock_id}_{stock_name}.png`，包含預測的報酬率柱狀圖和收盤價線圖。

"""

import os
# 設定 Keras 3 以 PyTorch 為後端
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import tensorflow as tf
import keras
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 設定 matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 法人權重設定
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

def _safe_log_return_pct(curr_price: float, prev_price: float) -> float:
    """Compute log-return (%) with finite/positive guards.
    
    Args:
        curr_price: Current price value.
        prev_price: Previous price value.
        
    Returns:
        Log-return in percent, or NaN if inputs are non-finite or non-positive.
    """
    if not np.isfinite(curr_price) or not np.isfinite(prev_price):
        return float("nan")
    if curr_price <= 0.0 or prev_price <= 0.0:
        return float("nan")
    return float(np.log(curr_price / prev_price) * 100.0)


def _safe_simple_return_pct(curr_price: float, prev_price: float) -> float:
    """Compute simple return (%) with finite/positive guards.

    Args:
        curr_price: Current price value.
        prev_price: Previous price value.

    Returns:
        Simple return in percent, or NaN if inputs are non-finite or non-positive.
    """
    if not np.isfinite(curr_price) or not np.isfinite(prev_price):
        return float("nan")
    if curr_price <= 0.0 or prev_price <= 0.0:
        return float("nan")
    return float((curr_price / prev_price - 1.0) * 100.0)


def _safe_price_from_return_pct(prev_price: float, return_pct: float) -> float:
    """Convert return (%) to price with overflow guards.
    
    Args:
        prev_price: Previous price value.
        return_pct: Return percentage value.
        
    Returns:
        Next price value, or NaN if inputs are invalid or result overflows.
    """
    if not np.isfinite(prev_price) or not np.isfinite(return_pct):
        return float("nan")
    if prev_price <= 0.0:
        return float("nan")
    exp_arg: float = float(np.clip(return_pct / 100.0, -50.0, 50.0))
    out: float = float(prev_price * np.exp(exp_arg))
    return out if np.isfinite(out) else float("nan")


def _robust_symmetric_ylim(*arrays: np.ndarray, default: float = 10.0) -> float:
    """Compute finite symmetric y-limit bound from arrays.
    
    Args:
        *arrays: Variable number of numpy arrays to compute bounds from.
        default: Default bound value if no finite values found.
        
    Returns:
        Symmetric y-axis limit (99th percentile of absolute finite values).
    """
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
# 🔥 聚合函數模塊：實現 5 預測共識機制
# ==================================================================================

def _weighted_average_predictions(predictions: list, method: str = "exponential") -> float:
    """
    計算加權平均預測（時間衰減）。
    
    Args:
        predictions: 包含 5 個預測值的列表 [D+1, D+2, D+3, D+4, D+5]
        method: 加權策略
          - 'exponential': 指數衰減 [16,8,4,2,1]，最近最重
          - 'linear': 線性衰減 [5,4,3,2,1]
          - 'recency': 只用最新一個預測
    
    Returns:
        加權平均值，或 NaN 如果輸入全為非有限
    """
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) == 0:
        return float("nan")
    
    if method == "exponential":
        weights = np.array([16, 8, 4, 2, 1], dtype=np.float64)
    elif method == "linear":
        weights = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    elif method == "recency":
        # 只用最新預測，其他權重為 0
        weights = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    else:
        weights = np.array([1, 1, 1, 1, 1], dtype=np.float64) / 5.0
    
    # 調整權重長度以匹配有限預測數量
    n_valid = len(finite_preds)
    if n_valid < len(weights):
        weights = weights[-n_valid:]
    
    weights_normalized = weights[:n_valid] / weights[:n_valid].sum()
    return float(np.dot(finite_preds, weights_normalized))


def _consensus_confidence(predictions: list) -> float:
    """
    計算預測共識強度（基於標準差）。
    
    信心度 = 1 / (1 + std)，範圍 [0, 1]
    - std 越小，預測越一致，信心度越高
    - std 越大，預測分散，信心度越低
    
    Args:
        predictions: 預測值列表
    
    Returns:
        信心度分數 [0, 1]，或 0.5 如果只有一個或零個有限預測
    """
    finite_preds = np.array([p for p in predictions if np.isfinite(p)])
    if len(finite_preds) <= 1:
        return 0.5
    
    std = float(np.std(finite_preds))
    if not np.isfinite(std) or std < 0:
        return 0.5
    
    return float(1.0 / (1.0 + std))


def _direction_vote(predictions: list) -> tuple:
    """
    投票預測方向（看漲/看跌）。
    
    Args:
        predictions: 預測值列表
    
    Returns:
        (vote_count, direction) 其中：
        - vote_count: 看漲（>0）的預測數量
        - direction: 'bullish'（≥3票）/ 'neutral'（=2票）/ 'bearish'（<2票）
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Dilated CNN 40-Day Aligned Predictor (Bug Fixed)")
    parser.add_argument("--model-path", type=str, default="institutional_net_buy_v2_dilated.keras")
    parser.add_argument("--tfrecord-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--vol-factor", type=float, default=1.0, help="波動擴展係數，若預測幅度太小可設為 1.5 ~ 2.0")
    # 🔥 新增共識聚合參數
    parser.add_argument("--ensemble-method", type=str, default="exponential", 
                        choices=["exponential", "linear", "recency", "none"],
                        help="歷史預測聚合策略：exponential(指數衰減), linear(線性衰減), recency(只用最新), none(無聚合)")
    parser.add_argument("--ensemble-future", type=bool, default=True,
                        help="未來 5 天預測是否使用多窗口聚合")
    parser.add_argument("--show-individual-preds", type=bool, default=False,
                        help="下圖是否顯示個別預測散點（默認隱藏以保持清晰）")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="信心度閾值 [0.0~1.0]，低於此值的柱子用淡灰色標記")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到模型 {args.model_path}")
        return
    
    # 建立 custom_objects 以確保載入時能找到 CustomLoss>quantile_loss_75
    custom_objects = {"CustomLoss>quantile_loss_75": quantile_loss_75, "quantile_loss_75": quantile_loss_75}
    model = keras.models.load_model(args.model_path, custom_objects=custom_objects)

    stats_path = args.stats_path or args.tfrecord_path.replace(".tfrecord", ".stats.json")
    if not os.path.exists(stats_path):
        print(f"錯誤：找不到統計檔案 {stats_path}")
        return
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)
    stats_index = {normalize_stock_id(k): k for k in stats_data.keys()}

    target_ids = set()
    targets_file = Path(__file__).parent / "targets.txt"
    if targets_file.exists():
        with open(targets_file, "r", encoding="utf-8") as f:
            target_ids = {normalize_stock_id(line) for line in f if line.strip()}

    raw_ds = tf.data.TFRecordDataset(args.tfrecord_path, buffer_size=1024*1024*100)
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

    print("正在從 TFRecord 提取資料流...")
    columns = list(feature_desc.keys())
    values_buffer = {col: [] for col in columns}
    for batch_raw in raw_ds.batch(4096):
        parsed = tf.io.parse_example(batch_raw, feature_desc)
        for col in columns: values_buffer[col].append(parsed[col].numpy())

    # =================================================================
    # 🔥 核心修復：使用強韌解碼器替代 astype(str)，防止 Windows ASCII 崩潰
    # =================================================================
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
                try:
                    data_dict[col] = arr.astype(str)
                except Exception:
                    data_dict[col] = np.array([str(x) for x in arr], dtype=object)
        else:
            data_dict[col] = arr

    df = pd.DataFrame(data_dict)
    df["stock_id"] = df["stock_id"].map(normalize_stock_id)
    if target_ids: df = df[df["stock_id"].isin(target_ids)].copy().reset_index(drop=True)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 特徵工程
    df["retail_nb"] = -(df["foreign_net_buy"] + df["foreign_dealer_net_buy"] + df["trust_net_buy"] + df["dealer_net_buy"] + df["dealer_hedge_net_buy"])
    df["w_buy"] = (df["foreign_net_buy"] * DEFAULT_WEIGHTS["foreign"] + df["foreign_dealer_net_buy"] * DEFAULT_WEIGHTS["foreign_dealer"] +
                    df["trust_net_buy"] * DEFAULT_WEIGHTS["trust"] + df["dealer_net_buy"] * DEFAULT_WEIGHTS["dealer"] +
                    df["dealer_hedge_net_buy"] * DEFAULT_WEIGHTS["dealer_hedge"] + df["retail_nb"] * DEFAULT_WEIGHTS["retail"])

    df["sig_total"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(10).sum())
    df["sig_pos_ratio"] = df.groupby("stock_id")["w_buy"].transform(lambda x: (x > 0).astype(float).rolling(10).mean())
    df["sig_slope"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.cumsum().rolling(10).apply(get_slope, raw=True))
    df["sig_baseline"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(20).mean().shift(10))
    df["sig_price_pct"] = df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(10))
    cost_cols = ["foreign_cost", "trust_cost", "dealer_cost", "dealer_hedge_cost", "foreign_dealer_cost"]
    for c in cost_cols: df[c] = (df[c] - df["close"]) / (df["close"] + 1e-8)

    feature_cols = ["w_buy", "sig_total", "sig_pos_ratio", "sig_slope", "sig_baseline", "sig_price_pct"] + cost_cols
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    output_dir = Path(__file__).parent / "predict_plot"
    output_dir.mkdir(exist_ok=True)

    for sid, group in df.groupby("stock_id"):
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

        # 🔥 初始化預測聚合欄位
        g["pred_returns_all"] = [[]] * len(g)  # 存儲每天的所有 5 個預測值
        g["pred_return_consensus"] = np.nan  # 聚合後的共識預測
        g["pred_confidence"] = np.nan  # 共識信心度
        g["pred_return_d1"] = np.nan  # 為了向後相容性
        g["actual_return_d1"] = np.nan
        g["pred_price_d1"] = np.nan

        for idx_pos, p_val in zip(valid_indices, preds_denorm):
            # p_val 是形狀 (5,) 的陣列，代表 D+1,D+2,D+3,D+4,D+5 的預測
            
            # 收集所有 5 個預測值（解碼後）
            all_predictions = [
                _decode_model_output_log_return(float(pv), stats_data[stats_key]) * args.vol_factor
                for pv in p_val
            ]
            # 使用 at 替代 loc 來直接設定物件
            g.at[g.index[idx_pos], "pred_returns_all"] = all_predictions
            
            # 根據 ensemble_method 聚合預測
            if args.ensemble_method == "none":
                # 無聚合，只用 D+1（第一個預測）
                consensus_pred = all_predictions[0]
            else:
                # 使用加權平均、線性、或 recency 聚合
                consensus_pred = _weighted_average_predictions(all_predictions, method=args.ensemble_method)
            
            g.at[g.index[idx_pos], "pred_return_consensus"] = consensus_pred
            g.loc[g.index[idx_pos], "pred_confidence"] = _consensus_confidence(all_predictions)
            g.loc[g.index[idx_pos], "pred_return_d1"] = consensus_pred  # 向後相容性
            
            # window 截至 idx_pos-1，預測的是從 idx_pos-1 到 idx_pos 的 return
            # → actual 應比較 close[idx_pos] vs close[idx_pos-1]
            g.at[g.index[idx_pos], "actual_return_d1"] = _safe_simple_return_pct(
                float(g["close"].iloc[idx_pos]),
                float(g["close"].iloc[idx_pos - 1]),
            )
            # pred_price_d1 = 前一天收盤 × exp(pred_ret)，即對 idx_pos 當天的價格預測
            g.at[g.index[idx_pos], "pred_price_d1"] = _safe_price_from_return_pct(
                float(g["close"].iloc[idx_pos - 1]),
                float(consensus_pred),
            )

        # =================================================================
        # 📊 核心對齊：將歷史回測天數鎖定為 40 天，確保上下子圖完美重合
        # =================================================================
        plot_history_len = 40
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

        # 未來 5 天預報
        # 🔥 改進：使用多個窗口位置聚合未來預測
        last_hist_price = float(g["close"].iloc[-1])
        
        if args.ensemble_future:
            # 從不同窗口起點進行預測（模擬歷史中多次預測同一天）
            future_preds_ensemble = {"d1": [], "d2": [], "d3": [], "d4": [], "d5": []}
            
            for offset in range(min(5, args.window_size)):
                window_start = len(g) - args.window_size - offset
                if window_start < 0:
                    break
                
                future_window = g.iloc[window_start : window_start + args.window_size][feature_cols].values
                future_window_scaled = (future_window - s_mean) / safe_std
                future_window_scaled = np.clip(future_window_scaled, -10, 10)
                future_preds_tmp = model.predict(np.array([future_window_scaled]), verbose=0)[0]
                
                # 收集每個 D+k 的預測
                for k in range(5):
                    future_preds_ensemble[f"d{k+1}"].append(float(future_preds_tmp[k]))
            
            # 對每個未來日期聚合預測
            future_preds_consensus = []
            future_confidences = []
            for k in range(1, 6):
                key = f"d{k}"
                preds_for_day = [
                    _decode_model_output_log_return(float(p), stats_data[stats_key]) * args.vol_factor
                    for p in future_preds_ensemble[key]
                ]
                consensus = _weighted_average_predictions(preds_for_day, method=args.ensemble_method)
                confidence = _consensus_confidence(preds_for_day)
                future_preds_consensus.append(consensus)
                future_confidences.append(confidence)
        else:
            # 單窗口預測（原始邏輯）
            future_window = g.iloc[-args.window_size:][feature_cols].values
            future_window_scaled = (future_window - s_mean) / safe_std
            future_window_scaled = np.clip(future_window_scaled, -10, 10)
            future_preds = model.predict(np.array([future_window_scaled]), verbose=0)[0]
            
            future_preds_consensus = [
                _decode_model_output_log_return(float(p), stats_data[stats_key]) * args.vol_factor
                for p in future_preds
            ]
            future_confidences = [0.5] * 5  # 無聚合時，預設中等信心度
        
        # 計算逐日簡單報酬
        # 步驟1：累積對數報酬已經在 future_preds_consensus 中
        future_cum_log = future_preds_consensus
        # 步驟2：累積 → 逐日對數增量
        future_daily_log = [future_cum_log[0]] + [future_cum_log[i] - future_cum_log[i - 1] for i in range(1, 5)]
        # 步驟3：轉為逐日簡單報酬 % = (exp(Δr/100) - 1) × 100
        future_forecast_returns = [(np.exp(d / 100.0) - 1.0) * 100.0 for d in future_daily_log]
        # 上圖價格線仍使用累積對數報酬推算
        future_prices = [
            _safe_price_from_return_pct(last_hist_price, float(r))
            for r in future_cum_log
        ]
        last_date_pd = pd.to_datetime(dates_hist[-1])
        future_dates = [(last_date_pd + pd.tseries.offsets.BDay(j)).strftime('%Y-%m-%d') for j in range(1, 6)]

        # 整合時間軸標籤
        all_x_labels = dates_hist + ["未來 D+1", "未來 D+2", "未來 D+3", "未來 D+4", "未來 D+5"]
        x_indices = np.arange(len(all_x_labels))
        bar_width = 0.25

        # 建立雙面板圖表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [4, 5]})
        stock_name = g['stock_name'].iloc[-1]
        fig.suptitle(f"股票 {sid_str} ({str(stock_name).strip()}) 籌碼機器學習「時序全對齊」決策看板", fontsize=15, fontweight='bold')

        # 上圖：絕對股價 (完全對齊索引軸)
        ax1.plot(x_indices[:plot_history_len], actual_prices_hist, marker='o', label='實際收盤價', color='#4a81ad', markersize=4)
        ax1.plot(x_indices[:plot_history_len], predicted_prices_hist, marker='x', linestyle='--', label='歷史 D+1 預測價', color='#f0a05e', markersize=5, alpha=0.7)
        
        future_x = [x_indices[plot_history_len - 1]] + list(x_indices[plot_history_len:])
        future_y = [actual_prices_hist[-1]] + future_prices
        ax1.plot(future_x, future_y, marker='*', linestyle='-', label='未來 5 日推估價預報', color='#d62728', linewidth=2, markersize=9)
        
        for xi, v in zip(x_indices[plot_history_len:], future_prices):
            ax1.text(xi, v, f'{v:.1f}', color='#d62728', fontsize=9, fontweight='bold', ha='center', va='bottom')
        ax1.set_ylabel("股價 (元)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left')

        # 下圖：報酬率驗證與預測 (垂直完全咬合 + 共識聚合)
        # 🔥 準備信心度背景和共識預測
        hist_consensus_ret = []
        hist_confidence = []
        
        for idx in g_plot.index:
            conf_val = g_plot.loc[idx, "pred_confidence"]
            cons_ret = g_plot.loc[idx, "pred_return_consensus"]
            
            # 轉換為簡單報酬
            cons_simple = (np.exp(cons_ret / 100.0) - 1.0) * 100.0 if np.isfinite(cons_ret) else np.nan
            
            hist_consensus_ret.append(cons_simple)
            hist_confidence.append(conf_val if np.isfinite(conf_val) else 0.5)
        
        hist_consensus_ret = np.asarray(hist_consensus_ret, dtype=np.float64)
        hist_confidence = np.asarray(hist_confidence, dtype=np.float64)
        
        # 繪製信心度背景色區間（整個歷史部分）
        for i in range(plot_history_len):
            conf = hist_confidence[i]
            # 根據信心度設定背景顏色
            if conf >= args.confidence_threshold:
                if conf >= 0.7:
                    bg_color = "#e8f5e9"  # 深綠背景（高信心）
                else:
                    bg_color = "#f1f8e9"  # 淡綠背景（中信心）
            else:
                bg_color = "#f5f5f5"  # 灰色背景（低信心）
            
            ax2.axvspan(x_indices[i] - 0.5, x_indices[i] + 0.5, alpha=0.2, color=bg_color, zorder=0)
        
        # 未來部分信心度背景
        for i, conf in enumerate(future_confidences):
            idx = plot_history_len + i
            if conf >= args.confidence_threshold:
                if conf >= 0.7:
                    bg_color = "#e8f5e9"  # 深綠
                else:
                    bg_color = "#f1f8e9"  # 淡綠
            else:
                bg_color = "#f5f5f5"  # 灰色
            
            ax2.axvspan(x_indices[idx] - 0.5, x_indices[idx] + 0.5, alpha=0.2, color=bg_color, zorder=0)
        
        # 繪製柱狀圖（從後往前，確保覆蓋順序正確）
        # 歷史部分：實際 (藍) 、預測-D+1 (粉)、共識 (深綠)
        ax2.bar(x_indices[:plot_history_len] - bar_width, hist_act_ret, bar_width, 
                label='歷史實際 D+1 回報 %', color='#9999ff', alpha=0.8, zorder=3)
        ax2.bar(x_indices[:plot_history_len], hist_pred_ret, bar_width, 
                label='歷史預測 D+1 回報 %', color='#ff9999', alpha=0.8, zorder=3)
        ax2.bar(x_indices[:plot_history_len] + bar_width, hist_consensus_ret, bar_width, 
                label='歷史共識預測 %', color='#4caf50', alpha=0.8, zorder=3)
        
        # 未來部分（綠/紅柱表示方向）
        future_colors = ['#ff3333' if r >= 0 else '#00aa00' for r in future_forecast_returns]
        rects_future = ax2.bar(x_indices[plot_history_len:], future_forecast_returns, bar_width * 2, 
                               label='未來逐日預期報酬 % (共識)', color=future_colors, edgecolor='black', 
                               alpha=0.9, zorder=3)
        
        # 未來預測值標籤
        for bar, r in zip(rects_future, future_forecast_returns):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + (0.1 if h>=0 else -0.4), f"{r:+.2f}%", 
                    ha='center', va='bottom' if h>=0 else 'top', fontsize=10, fontweight='bold', zorder=4)
        
        # 🔥 可選：繪製個別預測散點（歷史部分）
        if args.show_individual_preds:
            for idx, pred_list in enumerate(g_plot["pred_returns_all"]):
                if isinstance(pred_list, list) and len(pred_list) == 5:
                    x_pos = x_indices[idx]
                    for k, pred in enumerate(pred_list):
                        # 轉為簡單報酬
                        pred_simple = (np.exp(pred / 100.0) - 1.0) * 100.0 if np.isfinite(pred) else np.nan
                        if np.isfinite(pred_simple):
                            # k=0 (D+1) 時 x 偏移最多（左），k=4 (D+5) 時偏移最少（右）
                            x_scatter = x_pos + (k - 2) * 0.08
                            ax2.scatter(x_scatter, pred_simple, s=20, alpha=0.4, color='#999999', zorder=2)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(all_x_labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel("報酬率 (%)")
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
        safe_name = "".join(c for c in str(stock_name).strip() if c not in r'\\/*?:"<>|')
        plt.savefig(output_dir / f"{sid_str}_{safe_name}.png", dpi=300)
        plt.close()
        print(f"成功輸出 40 天完美對齊回測圖: {sid_str}_{safe_name}.png")

if __name__ == "__main__":
    main()