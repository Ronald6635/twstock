"""
使用訓練好的 Dilated CNN 模型進行 5 天視向預測與視覺化
====================================================
這個程式載入 12-feature 模型，並針對 targets.txt 中的股票進行預測。
預測結果會繪製成圖表並儲存在 predict_plot 目錄下。
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

# 法人權重設定 (需與訓練時一致)
DEFAULT_WEIGHTS = {
    "foreign": 1.0,
    "foreign_dealer": 1.0,
    "trust": 1.0,
    "dealer": 0.5,
    "dealer_hedge": 0.5,
    "retail": -0.2
}


def normalize_stock_id(value):
    """Normalize stock ID for cross-source matching."""
    sid = str(value).strip()
    if sid.endswith(".0"):
        sid = sid[:-2]
    return sid


def get_slope(y):
    if len(y) < 2: return 0.0
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.polyfit(np.arange(len(y_clean)), y_clean, 1)[0])

def main():
    parser = argparse.ArgumentParser(description="Dilated CNN 5-day Prediction Visualizer")
    parser.add_argument("--model-path", type=str, default="institutional_net_buy_v2_dilated.keras")
    parser.add_argument("--tfrecord-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str, help="Stats JSON path (defaults to auto-generated name)")
    parser.add_argument("--window-size", type=int, default=10)
    args = parser.parse_args()

    # 1. 載入模型
    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到模型 {args.model_path}")
        return
    model = keras.models.load_model(args.model_path)
    print(f"模型載入完成: {args.model_path}")

    # 2. 確定 Stats 路徑
    stats_path = args.stats_path or args.tfrecord_path.replace(".tfrecord", ".stats.json")
    if not os.path.exists(stats_path):
        print(f"錯誤：找不到統計檔案 {stats_path}，請先執行 ml.py 產生。")
        return
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)
    stats_index = {normalize_stock_id(k): k for k in stats_data.keys()}

    # 載入目標股號（提早載入，讓後續可先過濾再做特徵計算）
    target_ids = set()
    targets_file = Path(__file__).parent / "targets.txt"
    if targets_file.exists():
        with open(targets_file, "r", encoding="utf-8") as f:
            target_ids = {normalize_stock_id(line) for line in f if line.strip()}

    # 3. 讀取資料並轉換為 DataFrame 進行信號計算
    raw_ds = tf.data.TFRecordDataset(args.tfrecord_path, buffer_size=1024*1024*100) # 100MB buffer
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

    print("讀取 TFRecord 並計算即時信號（Batch 模式）...")
    parse_batch_size = 4096
    columns = list(feature_desc.keys())
    values_buffer = {col: [] for col in columns}
    count = 0
    for batch_raw in raw_ds.batch(parse_batch_size):
        parsed = tf.io.parse_example(batch_raw, feature_desc)
        batch_count = len(parsed["close"].numpy())
        count += batch_count
        if count % 5000 == 0:
            print(f"已讀取 {count} 筆資料...")

        for col in columns:
            values_buffer[col].append(parsed[col].numpy())

    if count == 0:
        print("錯誤：TFRecord 無資料")
        return

    data_dict = {}
    string_cols = {"date", "stock_id", "stock_name"}
    for col in columns:
        arr = np.concatenate(values_buffer[col], axis=0)

        if col in string_cols:
            # NOTE: tf.string 經過 batch parse 後可能是 bytes、unicode 或 object 混合陣列。
            # 直接對非字串 dtype 使用 np.char.decode 會觸發 TypeError。
            if arr.dtype.kind in ("S", "a"):
                data_dict[col] = np.char.decode(arr, "utf-8", errors="replace")
            elif arr.dtype.kind == "U":
                data_dict[col] = arr
            elif arr.dtype.kind == "O":
                data_dict[col] = np.array(
                    [
                        x.decode("utf-8", errors="replace")
                        if isinstance(x, (bytes, bytearray, np.bytes_))
                        else (x if isinstance(x, str) else ("" if x is None else str(x)))
                        for x in arr
                    ],
                    dtype=object,
                )
            else:
                data_dict[col] = arr.astype(str)
        else:
            data_dict[col] = arr

    df = pd.DataFrame(data_dict)
    df["stock_id"] = df["stock_id"].astype(str).map(normalize_stock_id)

    if target_ids:
        before_count = len(df)
        df = df[df["stock_id"].isin(target_ids)].copy().reset_index(drop=True)
        print(f"targets 預先過濾: {before_count} -> {len(df)} 筆")

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    
    # 信號計算 (與 ml.py 一致)
    df["retail_nb"] = -(df["foreign_net_buy"] + df["foreign_dealer_net_buy"] + df["trust_net_buy"] + df["dealer_net_buy"] + df["dealer_hedge_net_buy"])
    df["w_buy"] = (df["foreign_net_buy"] * DEFAULT_WEIGHTS["foreign"] + df["foreign_dealer_net_buy"] * DEFAULT_WEIGHTS["foreign_dealer"] +
                    df["trust_net_buy"] * DEFAULT_WEIGHTS["trust"] + df["dealer_net_buy"] * DEFAULT_WEIGHTS["dealer"] +
                    df["dealer_hedge_net_buy"] * DEFAULT_WEIGHTS["dealer_hedge"] + df["retail_nb"] * DEFAULT_WEIGHTS["retail"])
    
    df["sig_total"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(10).sum())
    df["sig_avg"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(10).mean())
    df["sig_pos_ratio"] = df.groupby("stock_id")["w_buy"].transform(lambda x: (x > 0).astype(float).rolling(10).mean())
    df["sig_slope"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.cumsum().rolling(10).apply(get_slope, raw=True))
    df["sig_baseline"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(20).mean().shift(10))
    df["sig_price_pct"] = df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(10))
    df["sig_price_slope"] = df.groupby("stock_id")["close"].transform(lambda x: x.rolling(10).apply(get_slope, raw=True))
    
    cost_cols = ["foreign_cost", "trust_cost", "dealer_cost", "dealer_hedge_cost", "foreign_dealer_cost"]
    feature_cols = ["w_buy", "sig_total", "sig_pos_ratio", "sig_slope", "sig_baseline", "sig_price_pct", "sig_price_slope"] + cost_cols

    # 處理 Infinity 並將 feature columns 的 NaN 填 0（保留早期行，避免 rolling warmup 丟棄過多資料）
    df = df.replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    # 只刪除 close 為 NaN 的行（真正無效資料）
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    print(f"資料清洗完成，剩餘 {len(df)} 筆資料，包含 {df['stock_id'].nunique()} 檔股票")

    output_dir = Path(__file__).parent / "predict_plot"
    output_dir.mkdir(exist_ok=True)

    print(f"對 {len(target_ids) if target_ids else '所有'} 股票進行預測並繪製比較圖...")
    skip_counts = {
        "not_in_targets": 0,
        "not_in_stats": 0,
        "too_short": 0,
        "no_window": 0,
    }
    seen_ids = set()
    processed_count = 0

    for sid, group in df.groupby("stock_id"):
        sid_str = normalize_stock_id(sid)
        seen_ids.add(sid_str)

        if target_ids and sid_str not in target_ids:
            skip_counts["not_in_targets"] += 1
            continue

        stats_key = stats_index.get(sid_str)
        if not stats_key:
            skip_counts["not_in_stats"] += 1
            continue
            
        g = group.dropna().copy().reset_index(drop=True)
        if len(g) <= args.window_size:
            skip_counts["too_short"] += 1
            continue
            
        # 歸一化參數
        s_min = np.array(stats_data[stats_key]["min"])
        s_max = np.array(stats_data[stats_key]["max"])
        s_denom = np.where(s_max - s_min == 0, 1.0, s_max - s_min)

        # 1. 準備預測視窗並執行 Batch 預測 (對該股票的所有可用日期進行預測)
        all_windows = []
        valid_indices = []
        for i in range(len(g)):
            if i < args.window_size:
                continue
            window_data = g.iloc[i - args.window_size : i][feature_cols].values
            norm_window = (window_data - s_min) / s_denom
            all_windows.append(norm_window)
            valid_indices.append(i)

        if not all_windows:
            skip_counts["no_window"] += 1
            continue

        batch_inp = np.array(all_windows)
        preds = model.predict(batch_inp, verbose=0)
        
        # ✅ 反標準化處理
        if "target_min" not in stats_data[stats_key]:
            print(f"警告：{stats_key} 缺少 target_min，請先重新執行 ml.py 訓練或更新 stats.json")
            continue
            
        t_min = np.array(stats_data[stats_key]["target_min"])
        t_max = np.array(stats_data[stats_key]["target_max"])
        t_denom = np.where(t_max - t_min == 0, 1.0, t_max - t_min)
        preds_denorm = preds * t_denom + t_min
        
        # 將 D+1 預測結果寫回 g 中，方便後續精確切片
        g["pred_d1"] = np.nan
        for idx, p_val in zip(valid_indices, preds_denorm):
            g.loc[g.index[idx], "pred_d1"] = float(p_val[0])
            
        # 2a. 藍線背景範圍：取最後 (window_size + 40) 筆，給足暖機前的歷史背景
        #     確保即使數據有限，藍線也能顯示盡可能多的歷史走勢
        g_blue = g.tail(args.window_size + 40).copy().reset_index(drop=True)
        if g_blue.empty:
            skip_counts["too_short"] += 1
            continue

        # 2b. 橘線預測範圍：跳過暖機期，只取有預測值的最後 40 天
        #     保證橘線永遠從圖表左端開始，不出現前段空白
        g_pred = g.iloc[args.window_size:].tail(40).copy().reset_index(drop=True)
        if g_pred.empty:
            skip_counts["too_short"] += 1
            continue

        dates_blue = g_blue['date'].values
        actual_prices = g_blue['close'].values
        dates_orange = g_pred['date'].values
        predicted_prices = g_pred['pred_d1'].values

        # 3. 提取最後一個觀測點的「未來 5 天」完整預報
        future_forecast = preds_denorm[-1].tolist()
        last_date_pd = pd.to_datetime(dates_blue[-1])
        # 使用 Pandas 的 BDay (Business Day) 計算未來 5 個交易日
        future_dates = [(last_date_pd + pd.tseries.offsets.BDay(i)).strftime('%Y-%m-%d') for i in range(1, 6)]

        # --- 繪圖設定 ---
        plt.figure(figsize=(15, 7))
        
        # A. 畫出實際價格 (藍色)，使用較長的背景窗
        plt.plot(dates_blue, actual_prices, marker='o', label='Actual Price', color='#4a81ad', markersize=4)
        
        # B. 畫出歷史對比預測線 (橘色虛線)，從第一個有預測值的點開始
        plt.plot(dates_orange, predicted_prices, marker='x', linestyle='--', label='Historical Pred (D+1)', color='#f0a05e', markersize=6, alpha=0.8)
        
        # C. 畫出「未來 5 天」預報線 (紅色實線，從最後一個藍線點延伸)
        forecast_dates = [dates_blue[-1]] + future_dates
        forecast_values = [actual_prices[-1]] + future_forecast
        plt.plot(forecast_dates, forecast_values, marker='*', linestyle='-', 
                 label='Future 5-Day Forecast', color='#d62728', linewidth=2, markersize=10)
        
        # 在未來的每個點上方標註價格
        for d, v in zip(future_dates, future_forecast):
            plt.text(d, v, f'{v:.1f}', color='#d62728', fontsize=10, fontweight='bold', ha='center', va='bottom')
        
        # 圖表修飾
        stock_name = g['stock_name'].iloc[-1]
        stock_name_str = str(stock_name).strip() if stock_name else "Unknown"
        plt.title(f"Stock {sid_str} ({stock_name_str}) - 歷史預測對比與未來 5 日預報", fontsize=14)
        plt.xlabel("日期")
        plt.ylabel("價格")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best')
        plt.tight_layout()

        # 儲存
        safe_stock_name = "".join(c for c in stock_name_str if c not in r'\\/*?:"<>|').strip()
        save_path = os.path.join(output_dir, f"{sid_str}_{safe_stock_name}.png")
        plt.savefig(save_path)
        plt.close()
        processed_count += 1
        print(f"已儲存預測比較圖: {sid_str}_{safe_stock_name}.png (股票名稱: {stock_name_str})")

    print("\n=== 執行摘要 ===")
    print(f"TFRecord 股票數: {len(seen_ids)}")
    print(f"targets 命中數: {len(seen_ids & target_ids) if target_ids else len(seen_ids)}")
    print(f"stats 正規化 key 數: {len(stats_index)}")
    print(f"成功輸出圖檔: {processed_count}")
    print(f"skip not_in_targets: {skip_counts['not_in_targets']}")
    print(f"skip not_in_stats: {skip_counts['not_in_stats']}")
    print(f"skip too_short: {skip_counts['too_short']}")
    print(f"skip no_window: {skip_counts['no_window']}")
    if target_ids and processed_count == 0:
        missing_targets = sorted(target_ids - seen_ids)
        print(f"警告：0 張圖。未出現在 TFRecord 的 targets 前 10 筆: {missing_targets[:10]}")

if __name__ == "__main__":
    main()
