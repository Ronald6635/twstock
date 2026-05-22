"""
使用 TFRecord 進行機器學習訓練的範例程式碼
====================================
這個程式建立了一個 8 層 Dilated CNN + GRU 模型，使用 12 個特徵 (7 個法人信號 + 5 個成本) 
來預測未來 5 天的收盤價。
"""

import os
# 設定 Keras 3 以 PyTorch 為後端
os.environ["KERAS_BACKEND"] = "torch"

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import torch

# 法人權重設定
DEFAULT_WEIGHTS = {
    "foreign": 1.0,
    "foreign_dealer": 1.0,
    "trust": 1.0,
    "dealer": 0.5,
    "dealer_hedge": 0.5,
    "retail": -0.2
}

def get_slope(y):
    """計算滾動視窗的線性斜率。"""
    if len(y) < 2: return 0.0
    # 確保值是有效的
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.polyfit(np.arange(len(y_clean)), y_clean, 1)[0])

def _make_windows(feature_values: np.ndarray, label_values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rolling windows from features and labels.

    Args:
        feature_values (np.ndarray): Feature matrix [N, F].
        label_values (np.ndarray): Label matrix [N, 5].
        window_size (int): Rolling window size.

    Returns:
        tuple[np.ndarray, np.ndarray]: Windowed feature and label arrays.
    """
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for i in range(len(feature_values) - window_size + 1):
        x_list.append(feature_values[i : i + window_size])
        y_list.append(label_values[i + window_size - 1])

    if not x_list:
        return np.empty((0, window_size, feature_values.shape[1]), dtype="float32"), np.empty((0, label_values.shape[1]), dtype="float32")

    return np.array(x_list, dtype="float32"), np.array(y_list, dtype="float32")


def _resolve_temporal_split_index(length: int, window_size: int, val_ratio: float) -> int:
    """
    Compute a valid chronological split index for a stock group.

    Args:
        length (int): Number of rows in the stock group.
        window_size (int): Rolling window length.
        val_ratio (float): Fraction of data reserved for validation.

    Returns:
        int: Training partition end index.
    """
    split_idx = int(length * (1.0 - val_ratio))
    split_idx = max(split_idx, window_size)
    split_idx = min(split_idx, length - window_size)
    return split_idx


def load_dataset(
    tfrecord_path: str,
    window_size: int = 10,
    use_cache: bool = True,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TFRecord data and build leakage-safe train/validation sets.

    Args:
        tfrecord_path (str): TFRecord file path.
        window_size (int): Rolling sequence length.
        use_cache (bool): Whether to use cache files.
        val_ratio (float): Validation ratio for each stock group.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_val, y_train, y_val.
    """
    cache_base = tfrecord_path.replace(".tfrecord", "")
    x_train_cache = f"{cache_base}_train_X.npy"
    x_val_cache = f"{cache_base}_val_X.npy"
    y_train_cache = f"{cache_base}_train_y.npy"
    y_val_cache = f"{cache_base}_val_Y.npy"
    stats_path = f"{cache_base}.leakage_fixed.stats.json"

    if use_cache and os.path.exists(x_train_cache) and os.path.exists(x_val_cache) and os.path.exists(y_train_cache) and os.path.exists(y_val_cache) and os.path.exists(stats_path):
        print("找到有效快取檔案，正在載入預先計算好的時序特徵...")
        return (
            np.load(x_train_cache),
            np.load(x_val_cache),
            np.load(y_train_cache),
            np.load(y_val_cache),
        )

    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    feature_desc = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "stock_id": tf.io.FixedLenFeature([], tf.string),
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

    print(f"正在讀取 TFRecord 並計算 12 個特徵信號...")
    rows = []
    for r in raw_ds:
        p = tf.io.parse_single_example(r, feature_desc)
        row = {}
        for k, v in p.items():
            val = v.numpy()
            row[k] = val.decode() if isinstance(val, bytes) else val
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("TFRecord 檔案為空，請確認資料來源。")

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    df["retail_nb"] = -(
        df["foreign_net_buy"]
        + df["foreign_dealer_net_buy"]
        + df["trust_net_buy"]
        + df["dealer_net_buy"]
        + df["dealer_hedge_net_buy"]
    )

    df["w_buy"] = (
        df["foreign_net_buy"] * DEFAULT_WEIGHTS["foreign"]
        + df["foreign_dealer_net_buy"] * DEFAULT_WEIGHTS["foreign_dealer"]
        + df["trust_net_buy"] * DEFAULT_WEIGHTS["trust"]
        + df["dealer_net_buy"] * DEFAULT_WEIGHTS["dealer"]
        + df["dealer_hedge_net_buy"] * DEFAULT_WEIGHTS["dealer_hedge"]
        + df["retail_nb"] * DEFAULT_WEIGHTS["retail"]
    )

    print("正在計算滾動時序信號 (Rolling Signals)...")
    df["sig_total"] = df.groupby("stock_id")["w_buy"].transform(lambda x: x.rolling(10).sum())
    df["sig_pos_ratio"] = df.groupby("stock_id")["w_buy"].transform(
        lambda x: (x > 0).astype(float).rolling(10).mean()
    )
    df["sig_slope"] = df.groupby("stock_id")["w_buy"].transform(
        lambda x: x.cumsum().rolling(10).apply(get_slope, raw=True)
    )
    df["sig_baseline"] = df.groupby("stock_id")["w_buy"].transform(
        lambda x: x.rolling(20).mean().shift(10)
    )
    df["sig_price_pct"] = df.groupby("stock_id")["close"].transform(lambda x: x.pct_change(10))
    df["sig_price_slope"] = df.groupby("stock_id")["close"].transform(
        lambda x: x.rolling(10).apply(get_slope, raw=True)
    )

    cost_cols = ["foreign_cost", "trust_cost", "dealer_cost", "dealer_hedge_cost", "foreign_dealer_cost"]
    feature_cols = [
        "w_buy",
        "sig_total",
        "sig_pos_ratio",
        "sig_slope",
        "sig_baseline",
        "sig_price_pct",
        "sig_price_slope",
    ] + cost_cols

    target_cols = [f"close_d+{i}" for i in range(1, 6)]
    for i in range(1, 6):
        df[f"close_d+{i}"] = df.groupby("stock_id")["close"].shift(-i)

    before_drop = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dropped = before_drop - len(df)
    print(f"特徵工程後移除 {dropped} 筆無效資料。")

    if df.empty:
        raise ValueError("特徵工程後無剩餘資料，可能是資料長度不足以計算滾動視窗。")

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []
    stats = {}
    skipped = 0

    groups = df.groupby("stock_id")
    total_groups = len(groups)
    for idx, (sid, group) in enumerate(groups):
        if idx % 100 == 0:
            print(f"處理中: {idx}/{total_groups}")

        g = group.sort_values("date").reset_index(drop=True)
        n_rows = len(g)
        split_idx = _resolve_temporal_split_index(n_rows, window_size, val_ratio)

        # 允許「剛好 == window_size」，因為可產生 1 個 window
        if split_idx < window_size or (n_rows - split_idx) < window_size:
            skipped += 1
            continue

        g_train = g.iloc[:split_idx].copy().reset_index(drop=True)
        g_val = g.iloc[split_idx:].copy().reset_index(drop=True)

        s_min = g_train[feature_cols].min()
        s_max = g_train[feature_cols].max()
        s_denom = (s_max - s_min).replace(0, 1.0)

        target_min = g_train[target_cols].min().to_numpy(dtype="float32")
        target_max = g_train[target_cols].max().to_numpy(dtype="float32")
        target_denom = np.where(target_max - target_min == 0, 1.0, target_max - target_min).astype("float32")

        g_train[feature_cols] = (g_train[feature_cols] - s_min) / s_denom
        g_val[feature_cols] = (g_val[feature_cols] - s_min) / s_denom
        g_train[target_cols] = (g_train[target_cols] - target_min) / target_denom
        g_val[target_cols] = (g_val[target_cols] - target_min) / target_denom

        x_tr, y_tr = _make_windows(g_train[feature_cols].values, g_train[target_cols].values, window_size)
        x_va, y_va = _make_windows(g_val[feature_cols].values, g_val[target_cols].values, window_size)

        if len(x_tr) == 0 or len(x_va) == 0:
            skipped += 1
            continue

        x_train_list.append(x_tr)
        y_train_list.append(y_tr)
        x_val_list.append(x_va)
        y_val_list.append(y_va)

        stats[str(sid)] = {
            "min": s_min.tolist(),
            "max": s_max.tolist(),
            "target_min": target_min.tolist(),
            "target_max": target_max.tolist(),
            "columns": feature_cols,
            "train_end_date": str(g_train["date"].iloc[-1]),
            "val_start_date": str(g_val["date"].iloc[0]),
        }

    if not x_train_list or not x_val_list:
        raise ValueError("切分後沒有可用資料。請調整 window_size 或 val_ratio。")

    x_train = np.concatenate(x_train_list, axis=0).astype("float32")
    y_train = np.concatenate(y_train_list, axis=0).astype("float32")
    x_val = np.concatenate(x_val_list, axis=0).astype("float32")
    y_val = np.concatenate(y_val_list, axis=0).astype("float32")

    print(f"略過資料不足或無法切分的股票數: {skipped}")
    print(f"訓練樣本數: {len(x_train)}, 驗證樣本數: {len(x_val)}")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)
    print(f"統計值已儲存至: {stats_path}")

    if use_cache:
        np.save(x_train_cache, x_train)
        np.save(x_val_cache, x_val)
        np.save(y_train_cache, y_train)
        np.save(y_val_cache, y_val)
        print("已儲存 cache 檔案。")

    return x_train, x_val, y_train, y_val

def build_model(input_shape: tuple):
    """
    建立 8 層 Dilated CNN + GRU 模型。
    輸出層為 Dense(5) 預測未來 5 天價格。
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # Dilation rates: 1, 2, 4, 8 重複兩次
    dilations = [1, 2, 4, 8, 1, 2, 4, 8]
    for i, d in enumerate(dilations):
        model.add(keras.layers.Conv1D(
            64, kernel_size=3, activation='relu', 
            padding='causal', dilation_rate=d, 
            name=f'dilated_conv_{i+1}'
        ))
        model.add(keras.layers.Dropout(0.2))
        
    # model.add(keras.layers.GRU(64, return_sequences=False, name='gru_core'))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.Dense(5, name='output_5day_forecast'))
    model.add(keras.layers.Conv1D(64, kernel_size=1, activation='relu', name='final_conv'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GlobalAveragePooling1D(name='global_avg_pool'))
    model.add(keras.layers.Dense(5, name='output_5day_forecast'))

    # model.add(keras.layers.Conv1D(5, kernel_size=1, activation="linear", name="output_5day_forecast_seq"))
    # model.add(keras.layers.Lambda(lambda t: t[:, -1, :], name="take_last_step"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="使用 12 特徵與 8 層 Dilated CNN 訓練模型")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="TFRecord 檔案路徑")
    parser.add_argument("--epochs", type=int, default=50, help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=512, help="批次大小 (建議 GPU 使用 512 以上)")
    parser.add_argument("--window-size", type=int, default=10, help="滑動視窗天數")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="每支股票驗證集比例")
    parser.add_argument("--no-cache", action="store_true", help="不使用快取，強制重新計算特徵")
    args = parser.parse_args()

    tf_path = Path(args.tfrecord_path)
    if not tf_path.exists():
        print(f"找不到檔案: {tf_path}")
        return

    print("====================================")
    print(f"Keras 後端: {keras.config.backend()}")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 載入資料 (包含快取與特徵工程)
    X_train, X_val, y_train, y_val = load_dataset(
        str(tf_path), 
        window_size=args.window_size,
        use_cache=not args.no_cache,
        val_ratio=args.val_ratio,
    )
    print(f"訓練樣本數: {len(X_train)}, 驗證樣本數: {len(X_val)}")
    
    # 2. 建立模型 (12 個特徵)
    model = build_model(input_shape=(args.window_size, 12))
    model.summary()
    
    # 4. 設定回調函數 (Callbacks) 以節省時間
    model_name = "institutional_net_buy_v2_dilated.keras"
    callbacks = [
        # 當驗證損失不再改善時停止訓練
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        # 當學習停滯時降低學習率
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            verbose=1
        ),
        # 儲存最佳模型
        keras.callbacks.ModelCheckpoint(
            model_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 5. 訓練
    print("\n開始訓練...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True
    )
    
    print(f"模型訓練完成。最佳權重已儲存至: {model_name}")

if __name__ == "__main__":
    main()
