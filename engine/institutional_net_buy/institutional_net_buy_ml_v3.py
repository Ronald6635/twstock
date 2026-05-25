"""
使用 TFRecord 進行機器學習訓練的範例程式碼 (V3: 動態特徵配置版)
================================================================
這個程式建立了一個 Dilated CNN / LSTM 模型，使用動態配置的特徵
(如籌碼純度、成本乖離率、淨買動能等) 來預測未來 5 天的對數報酬率。
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
}

# ==========================================
# 特徵配置中樞 (Feature Configuration Hub)
# ==========================================
# 透過此設定，你可以自由開啟/關閉要餵給神經網路的特徵

ALL_FEATURES = {
    # 1. 淨買超動能
    "net_buy": [
        "foreign_net_buy", "trust_net_buy", "dealer_net_buy",
        "foreign_dealer_net_buy", "dealer_hedge_net_buy"
    ],
    # 2. 籌碼進貨純度 (Conviction Intensity: 0~1)
    "intensity": [
        "conviction_intensity"
    ],
    # 3. 成本防守乖離率 (Cost Deviation: %)
    "cost_deviation": [
        "foreign_cost_dev", "trust_cost_dev", "dealer_cost_dev", 
        "foreign_dealer_cost_dev", "dealer_hedge_cost_dev"
    ],
    # 4. 價格與趨勢動能
    "price_action": [
        "vwap_change_pct"  # 近期 VWAP 實質變化率
    ]
}

# ⚡ 你可以隨時編輯這個 Tuple 來決定模型這回合要學什麼！
# 目前使用：Full Feature Set (全部塞進去)
SELECTED_FEATURES = tuple(
    ALL_FEATURES["intensity"] + 
    ALL_FEATURES["cost_deviation"] +
    ALL_FEATURES["net_buy"] +
    ALL_FEATURES["price_action"]
)

# 這個變數將決定你的 Keras 輸入層大小：(window_size, INPUT_DIM)
INPUT_DIM = len(SELECTED_FEATURES)


@keras.saving.register_keras_serializable(package="CustomLoss", name="quantile_loss_75")
def quantile_loss_75(y_true, y_pred):
    """
    Quantile loss for q=0.75, registered for model serialization.
    """
    q = 0.75
    error = y_true - y_pred
    return keras.ops.mean(keras.ops.maximum(q * error, (q - 1.0) * error), axis=-1)


def _build_loss(loss_name: str, quantile: float):
    """Build a configurable training loss."""
    if loss_name == "mse":
        return keras.losses.MeanSquaredError()
    if loss_name == "huber":
        return keras.losses.Huber(delta=0.5)
    if loss_name == "quantile":
        if abs(quantile - 0.75) < 1e-6:
            return quantile_loss_75
            
        def quantile_loss(y_true, y_pred):
            error = y_true - y_pred
            q = float(quantile)
            return keras.ops.mean(keras.ops.maximum(q * error, (q - 1.0) * error), axis=-1)

        return quantile_loss
    raise ValueError(f"不支援的 loss: {loss_name}")

def _make_windows(feature_values: np.ndarray, label_values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build rolling windows from features and labels.
    
    Skips windows where any feature in the window or the label contains NaN,
    preserving temporal continuity (no row removal before windowing).
    """
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for i in range(len(feature_values) - window_size + 1):
        x_win = feature_values[i : i + window_size]
        y_win = label_values[i + window_size - 1]
        # Skip windows with NaN in features or labels to preserve temporal continuity
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            continue
        x_list.append(x_win)
        y_list.append(y_win)

    if not x_list:
        return np.empty((0, window_size, feature_values.shape[1]), dtype="float32"), np.empty((0, label_values.shape[1]), dtype="float32")

    return np.array(x_list, dtype="float32"), np.array(y_list, dtype="float32")


def _resolve_temporal_split_index(length: int, window_size: int, val_ratio: float) -> int:
    """Compute a valid chronological split index for a stock group."""
    split_idx = int(length * (1.0 - val_ratio))
    split_idx = max(split_idx, window_size)
    split_idx = min(split_idx, length - window_size)
    return split_idx


def _transform_return_targets(target_values: pd.DataFrame, target_transform: str, predict_days: int) -> np.ndarray:
    """Transforms target prices into appropriate return forms.
    
    Stage 1: Preventive masking ensures only valid price pairs are used for log computation.
    """
    arr = target_values.values
    if target_transform == "cum_log":
        current_price = arr[:-predict_days]
        shifted_prices = [arr[i : len(arr) - predict_days + i] for i in range(1, predict_days + 1)]
        y_out = np.zeros((len(current_price), predict_days), dtype="float32")
        for step in range(predict_days):
            # Stage 1: Preventive masking - only compute log for valid price pairs
            # Check that both current and shifted prices are positive
            valid_mask = (current_price > 0.0) & (shifted_prices[step] > 0.0)
            # Initialize with NaN, only fill valid entries
            log_returns = np.full_like(y_out[:, step], np.nan, dtype="float32")
            log_returns[valid_mask] = (np.log(shifted_prices[step][valid_mask] / current_price[valid_mask]) * 100.0).astype("float32")
            y_out[:, step] = log_returns
        return y_out
    else:
        raise ValueError(f"未知的 target_transform: {target_transform}")

class DynamicStandardScaler:
    """A simple StandardScaler for the dynamically selected features."""
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, data: np.ndarray):
        self.mean = np.nanmean(data, axis=0)
        self.scale = np.nanstd(data, axis=0)
        # Stage 3: Standardization safety - clip std to prevent division by zero
        self.scale = np.clip(self.scale, a_min=1e-4, a_max=None)
        self.scale[self.scale == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.scale

    def save(self, filepath: str):
        np.savez(filepath, mean=self.mean, scale=self.scale)

    def load(self, filepath: str):
        npz = np.load(filepath)
        self.mean = npz['mean']
        self.scale = npz['scale']


def load_and_preprocess_data(
    tfrecord_path: str,
    window_size: int,
    predict_days: int,
    val_ratio: float,
    cache_dir: str = ".cache",
    target_transform: str = "cum_log"
):
    """Reads TFRecord, dynamically calculates features based on SELECTED_FEATURES, standardizes, and windows."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check cache (simplified for brevity, omitting specific target checking)
    x_train_cache = os.path.join(cache_dir, f"x_train_w{window_size}_p{predict_days}_dim{INPUT_DIM}.npy")
    x_val_cache = os.path.join(cache_dir, f"x_val_w{window_size}_p{predict_days}_dim{INPUT_DIM}.npy")
    y_train_cache = os.path.join(cache_dir, f"y_train_w{window_size}_p{predict_days}_dim{INPUT_DIM}.npy")
    y_val_cache = os.path.join(cache_dir, f"y_val_w{window_size}_p{predict_days}_dim{INPUT_DIM}.npy")

    if all(os.path.exists(f) for f in [x_train_cache, x_val_cache, y_train_cache, y_val_cache]):
        print("找到有效快取檔案，正在載入...")
        return np.load(x_train_cache), np.load(x_val_cache), np.load(y_train_cache), np.load(y_val_cache)

    raw_ds = tf.data.TFRecordDataset(tfrecord_path, buffer_size=1024*1024*100)
    
    feature_desc = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "stock_id": tf.io.FixedLenFeature([], tf.string),
        "close": tf.io.FixedLenFeature([], tf.float32),
        "vwap": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    
    for prefix in ["foreign", "trust", "dealer", "foreign_dealer", "dealer_hedge"]:
        feature_desc[f"{prefix}_net_buy"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        feature_desc[f"{prefix}_buy"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        feature_desc[f"{prefix}_sell"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
        feature_desc[f"{prefix}_cost"] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

    print(f"正在讀取 TFRecord...")
    rows = []
    for r in raw_ds:
        p = tf.io.parse_single_example(r, feature_desc)
        row = {k: v.numpy() for k, v in p.items()}
        row["date"] = row["date"].decode()
        row["stock_id"] = row["stock_id"].decode()
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("TFRecord 檔案為空。")

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    print("正在計算衍生特徵 (純度、乖離率、動能)...")
    
    # A. 籌碼參與純度 (Intensity)
    total_net = sum(df[f"{p}_net_buy"] * DEFAULT_WEIGHTS[p] for p in DEFAULT_WEIGHTS)
    total_vol = sum((df[f"{p}_buy"] + df[f"{p}_sell"]) * DEFAULT_WEIGHTS[p] for p in DEFAULT_WEIGHTS)
    df["conviction_intensity"] = total_net / (total_vol + 1e-5)

    # B. 成本防守乖離率 (Cost Deviation)
    for prefix in ["foreign", "trust", "dealer", "foreign_dealer", "dealer_hedge"]:
        df[f"{prefix}_cost_dev"] = (df["vwap"] - df[f"{prefix}_cost"]) / (df[f"{prefix}_cost"] + 1e-8)
        
    # C. VWAP 近日動能
    df["vwap_change_pct"] = df.groupby("stock_id")["vwap"].pct_change().fillna(0)

    missing_feats = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"缺少指定的特徵: {missing_feats}")

    # ========================================================================
    # Stage 2: Core cleanup after all feature engineering
    # ========================================================================
    # Replace all [inf, -inf] with NaN, then drop rows containing any NaN
    # This ensures data quality before windowing
    before_cleanup = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    cleaned_rows = before_cleanup - len(df)
    print(f"特徵工程後移除 {cleaned_rows} 筆無效資料 (NaN/Inf)。")
    
    if df.empty:
        raise ValueError("特徵工程後無剩餘資料，可能是資料長度不足或無效數值過多。")
    # ========================================================================

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    grouped = df.groupby("stock_id")
    for stock_id, group in grouped:
        if len(group) < window_size + predict_days:
            continue

        features = group[list(SELECTED_FEATURES)].values.astype("float32")
        targets = _transform_return_targets(group["close"], target_transform, predict_days)
        # features[i] corresponds to targets[i]: both represent date index i
        # Do NOT drop rows here — removing rows breaks temporal continuity.
        # NaN labels are handled inside _make_windows by skipping affected windows.
        features = features[:-predict_days]

        if len(features) < window_size + 1:
            continue

        split_idx = _resolve_temporal_split_index(len(features), window_size, val_ratio)
        
        feat_train = features[:split_idx]
        feat_val = features[split_idx:]
        tgt_train = targets[:split_idx]
        tgt_val = targets[split_idx:]

        X_t, y_t = _make_windows(feat_train, tgt_train, window_size)
        if len(X_t) > 0:
            X_train_list.append(X_t)
            y_train_list.append(y_t)
            
        X_v, y_v = _make_windows(feat_val, tgt_val, window_size)
        if len(X_v) > 0:
            X_val_list.append(X_v)
            y_val_list.append(y_v)

    if not X_train_list or not X_val_list:
        raise ValueError("無足夠資料可切分訓練/驗證集")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    print("正在進行 Z-Score 標準化...")
    # Flatten the window and sample dimensions for scaling
    orig_train_shape = X_train.shape
    orig_val_shape = X_val.shape
    
    scaler = DynamicStandardScaler()
    X_train_flat = X_train.reshape(-1, INPUT_DIM)
    scaler.fit(X_train_flat)
    
    X_train = scaler.transform(X_train_flat).reshape(orig_train_shape)
    X_val = scaler.transform(X_val.reshape(-1, INPUT_DIM)).reshape(orig_val_shape)
    
    scaler.save(os.path.join(cache_dir, f"scaler_dim{INPUT_DIM}.npz"))

    # Apply clipping robustly to prevent extreme outliers
    X_train = np.clip(X_train, -5.0, 5.0)
    X_val = np.clip(X_val, -5.0, 5.0)

    np.save(x_train_cache, X_train)
    np.save(x_val_cache, X_val)
    np.save(y_train_cache, y_train)
    np.save(y_val_cache, y_val)

    return X_train, X_val, y_train, y_val

# ==========================================
# 模型建構
# ==========================================
def build_model(window_size: int, predict_days: int) -> keras.Model:
    """
    Builds a flexible CNN/LSTM hybrid model adapting to INPUT_DIM.
    """
    inputs = keras.Input(shape=(window_size, INPUT_DIM))
    
    # Spatial feature extraction (1D Conv)
    x = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", dilation_rate=2)(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Temporal sequence modeling
    x = keras.layers.LSTM(64, return_sequences=False)(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(predict_days)(x)
    
    return keras.Model(inputs, outputs)

# ==========================================
# 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="動態特徵配置機器學習訓練 (V3)")
    parser.add_argument("--tfrecord-path", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--predict-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="quantile", choices=["mse", "huber", "quantile"])
    parser.add_argument("--quantile", type=float, default=0.75)
    args = parser.parse_args()

    print("\n=== 模型特徵配置清單 ===")
    print(f"目前啟用 {INPUT_DIM} 個特徵維度:")
    for i, f in enumerate(SELECTED_FEATURES, 1):
        print(f"  {i}. {f}")
    print("========================\n")

    X_train, X_val, y_train, y_val = load_and_preprocess_data(
        tfrecord_path=args.tfrecord_path,
        window_size=args.window_size,
        predict_days=args.predict_days,
        val_ratio=args.val_ratio
    )
    
    print(f"訓練集形狀: {X_train.shape}, 標籤形狀: {y_train.shape}")
    print(f"驗證集形狀: {X_val.shape}, 標籤形狀: {y_val.shape}")

    # Diagnostic: check for NaN/Inf in data before training
    print(f"X_train NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"y_train NaN: {np.isnan(y_train).sum()}, Inf: {np.isinf(y_train).sum()}")
    print(f"X_val NaN: {np.isnan(X_val).sum()}, Inf: {np.isinf(X_val).sum()}")
    print(f"y_val NaN: {np.isnan(y_val).sum()}, Inf: {np.isinf(y_val).sum()}")

    # 建立模型
    model = build_model(window_size=args.window_size, predict_days=args.predict_days)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=_build_loss(loss_name=args.loss, quantile=args.quantile),
    )

    model_name = f"model_w{args.window_size}_p{args.predict_days}_dim{INPUT_DIM}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_name,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n開始訓練...")
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=min(len(X_train), 50000))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"\n訓練完成，模型已儲存為 {model_name}")


if __name__ == "__main__":
    main()
