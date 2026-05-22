"""
Institutional Net Buy Attention LSTM Training Module

This module trains an Attention + LSTM model with 12 engineered features
(7 institutional signals + 5 cost features) to forecast the next 5 closing prices.

Key features:
- Per-stock temporal split to prevent time-series leakage
- Train-only min-max scaling for both features and targets
- Denormalized validation MAE and RMSE metrics for interpretability
"""

import os

# Set Keras 3 backend to PyTorch.
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import json
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

# Institutional weight settings.
DEFAULT_WEIGHTS = {
    "foreign": 1.25,
    "foreign_dealer": 1.0,
    "trust": 0.75,
    "dealer": 0.5,
    "dealer_hedge": 0.5,
    "retail": -0.2,
}

CACHE_SCHEMA_VERSION = "v3_lstm_attention_temporal_split"


def get_slope(y: np.ndarray) -> float:
    """
    Compute linear slope in a rolling window.

    Args:
        y (np.ndarray): Input window values.

    Returns:
        float: Slope of linear fit.
    """
    if len(y) < 2:
        return 0.0
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.polyfit(np.arange(len(y_clean)), y_clean, 1)[0])


def _cache_paths(tfrecord_path: str) -> dict[str, str]:
    """
    Build cache file paths for the Attention + LSTM pipeline.

    Args:
        tfrecord_path (str): TFRecord source path.

    Returns:
        dict[str, str]: Cache path mapping.
    """
    cache_base = tfrecord_path.replace(".tfrecord", "")
    return {
        "x_train": f"{cache_base}_lstmattn_X_train.npy",
        "x_val": f"{cache_base}_lstmattn_X_val.npy",
        "y_train": f"{cache_base}_lstmattn_y_train.npy",
        "y_val": f"{cache_base}_lstmattn_y_val.npy",
        "y_val_min": f"{cache_base}_lstmattn_y_val_min.npy",
        "y_val_denom": f"{cache_base}_lstmattn_y_val_denom.npy",
        "stats": f"{cache_base}_lstmattn.stats.json",
    }


def _is_cache_valid(paths: dict[str, str], use_cache: bool) -> bool:
    """
    Validate whether cache files are present and schema-compatible.

    Args:
        paths (dict[str, str]): Cache path mapping.
        use_cache (bool): Whether cache usage is enabled.

    Returns:
        bool: True if cache is valid.
    """
    if not use_cache:
        return False

    required = [
        paths["x_train"],
        paths["x_val"],
        paths["y_train"],
        paths["y_val"],
        paths["y_val_min"],
        paths["y_val_denom"],
        paths["stats"],
    ]
    if not all(os.path.exists(p) for p in required):
        return False

    try:
        with open(paths["stats"], "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("_meta", {}).get("schema_version") == CACHE_SCHEMA_VERSION
    except Exception as e:
        print(f"檢查快取時出錯: {e}")
        return False


def _make_windows(
    feature_values: np.ndarray,
    label_values: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert per-day rows into rolling windows.

    Args:
        feature_values (np.ndarray): Feature matrix [N, F].
        label_values (np.ndarray): Label matrix [N, 5].
        window_size (int): Rolling window length.

    Returns:
        tuple[np.ndarray, np.ndarray]: Windowed X and aligned y.
    """
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    total = len(feature_values)
    for i in range(total - window_size + 1):
        x_list.append(feature_values[i : i + window_size])
        y_list.append(label_values[i + window_size - 1])

    if not x_list:
        return (
            np.empty((0, window_size, feature_values.shape[1]), dtype="float32"),
            np.empty((0, label_values.shape[1]), dtype="float32"),
        )

    return np.array(x_list, dtype="float32"), np.array(y_list, dtype="float32")


def _resolve_temporal_split_index(length: int, window_size: int, val_ratio: float) -> int:
    """
    Determine a valid split index that preserves train/validation windows.

    Args:
        length (int): Total rows of one stock group.
        window_size (int): Rolling window size.
        val_ratio (float): Validation ratio.

    Returns:
        int: Split index between train and validation partitions.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TFRecord and build leakage-safe train/validation datasets.

    Args:
        tfrecord_path (str): TFRecord file path.
        window_size (int): Rolling sequence length.
        use_cache (bool): Whether to use precomputed cache.
        val_ratio (float): Validation ratio per stock in chronological order.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_val, y_train, y_val, y_val_min, y_val_denom.
    """
    if not (0.05 <= val_ratio <= 0.5):
        raise ValueError("val_ratio 必須在 0.05 到 0.5 之間。")

    paths = _cache_paths(tfrecord_path)

    if _is_cache_valid(paths, use_cache):
        print("找到有效快取，正在載入 Attention+LSTM 特徵資料...")
        return (
            np.load(paths["x_train"]),
            np.load(paths["x_val"]),
            np.load(paths["y_train"]),
            np.load(paths["y_val"]),
            np.load(paths["y_val_min"]),
            np.load(paths["y_val_denom"]),
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

    print("正在讀取 TFRecord 並計算 12 個特徵信號...")
    rows: list[dict[str, Any]] = []
    for record in raw_ds:
        parsed = tf.io.parse_single_example(record, feature_desc)
        row: dict[str, Any] = {}
        for key, value in parsed.items():
            val = value.numpy()
            row[key] = val.decode() if isinstance(val, bytes) else val
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("TFRecord 檔案為空，請確認資料來源。")

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # Retail estimate = inverse of the five institutional net buys.
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

    cost_cols = [
        "foreign_cost",
        "trust_cost",
        "dealer_cost",
        "dealer_hedge_cost",
        "foreign_dealer_cost",
    ]
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
    dropped_rows = before_drop - len(df)

    if df.empty:
        raise ValueError("特徵工程後無剩餘資料，可能是資料長度不足以計算滾動視窗。")

    print("正在執行每支股票獨立的時序切分與 Train-only Min-Max 標準化...")
    print(f"特徵工程清洗後共移除 {dropped_rows} 筆資料。")

    stats: dict[str, Any] = {
        "_meta": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "window_size": window_size,
            "val_ratio": val_ratio,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "dropped_rows_after_engineering": dropped_rows,
        }
    }

    x_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    x_val_list: list[np.ndarray] = []
    y_val_list: list[np.ndarray] = []
    y_val_min_list: list[np.ndarray] = []
    y_val_denom_list: list[np.ndarray] = []

    skipped_small_groups = 0
    skipped_invalid_split = 0

    groups = df.groupby("stock_id", sort=False)
    total_groups = len(groups)

    for idx, (sid, group) in enumerate(groups):
        if idx % 100 == 0:
            print(f"處理中: {idx}/{total_groups}")

        g = group.copy().sort_values("date").reset_index(drop=True)
        n_rows = len(g)

        if n_rows < window_size * 2:
            skipped_small_groups += 1
            continue

        split_idx = _resolve_temporal_split_index(n_rows, window_size, val_ratio)
        if split_idx < window_size or (n_rows - split_idx) < window_size:
            skipped_invalid_split += 1
            continue

        g_train = g.iloc[:split_idx].copy().reset_index(drop=True)
        g_val = g.iloc[split_idx:].copy().reset_index(drop=True)

        # Fit scalers on training partition only.
        s_min = g_train[feature_cols].min()
        s_max = g_train[feature_cols].max()
        s_denom = (s_max - s_min).replace(0, 1.0)

        target_min = g_train[target_cols].min().to_numpy(dtype="float32")
        target_max = g_train[target_cols].max().to_numpy(dtype="float32")
        target_denom = np.where(target_max - target_min == 0, 1.0, target_max - target_min).astype(
            "float32"
        )

        for partition in (g_train, g_val):
            partition[feature_cols] = (partition[feature_cols] - s_min) / s_denom
            partition[target_cols] = (partition[target_cols] - target_min) / target_denom

        feat_train = g_train[feature_cols].to_numpy(dtype="float32")
        label_train = g_train[target_cols].to_numpy(dtype="float32")
        feat_val = g_val[feature_cols].to_numpy(dtype="float32")
        label_val = g_val[target_cols].to_numpy(dtype="float32")

        x_tr, y_tr = _make_windows(feat_train, label_train, window_size)
        x_va, y_va = _make_windows(feat_val, label_val, window_size)

        if len(x_tr) == 0 or len(x_va) == 0:
            skipped_invalid_split += 1
            continue

        x_train_list.append(x_tr)
        y_train_list.append(y_tr)
        x_val_list.append(x_va)
        y_val_list.append(y_va)
        y_val_min_list.append(np.repeat(target_min[np.newaxis, :], len(y_va), axis=0))
        y_val_denom_list.append(np.repeat(target_denom[np.newaxis, :], len(y_va), axis=0))

        sid_str = str(sid)
        stats[sid_str] = {
            "min": s_min.tolist(),
            "max": s_max.tolist(),
            "target_min": target_min.tolist(),
            "target_max": target_max.tolist(),
            "columns": feature_cols,
            "train_rows": int(len(g_train)),
            "val_rows": int(len(g_val)),
            "train_end_date": str(g_train["date"].iloc[-1]),
            "val_start_date": str(g_val["date"].iloc[0]),
        }

    if not x_train_list or not x_val_list:
        raise ValueError("切分後沒有可用資料。請調整 window_size 或 val_ratio。")

    x_train = np.concatenate(x_train_list, axis=0).astype("float32")
    y_train = np.concatenate(y_train_list, axis=0).astype("float32")
    x_val = np.concatenate(x_val_list, axis=0).astype("float32")
    y_val = np.concatenate(y_val_list, axis=0).astype("float32")
    y_val_min = np.concatenate(y_val_min_list, axis=0).astype("float32")
    y_val_denom = np.concatenate(y_val_denom_list, axis=0).astype("float32")

    print(f"略過資料不足的股票數: {skipped_small_groups}")
    print(f"略過切分無法建窗的股票數: {skipped_invalid_split}")
    print(f"訓練樣本數: {len(x_train)}, 驗證樣本數: {len(x_val)}")

    with open(paths["stats"], "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"統計值已儲存至: {paths['stats']}")

    if use_cache:
        print("正在儲存快取檔案...")
        np.save(paths["x_train"], x_train)
        np.save(paths["x_val"], x_val)
        np.save(paths["y_train"], y_train)
        np.save(paths["y_val"], y_val)
        np.save(paths["y_val_min"], y_val_min)
        np.save(paths["y_val_denom"], y_val_denom)

    return x_train, x_val, y_train, y_val, y_val_min, y_val_denom


def build_model(
    input_shape: tuple[int, int],
    lstm_units: int = 64,
    attention_heads: int = 4,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Build an Attention + LSTM forecasting model.

    Args:
        input_shape (tuple[int, int]): Input shape as (window_size, feature_count).
        lstm_units (int): Hidden units in LSTM layers.
        attention_heads (int): Number of attention heads.
        dropout (float): Dropout ratio.

    Returns:
        keras.Model: Compiled Keras model.
    """
    key_dim = max(8, lstm_units // max(attention_heads, 1))

    inputs = keras.layers.Input(shape=input_shape, name="input_sequence")
    x = keras.layers.LayerNormalization(name="input_norm")(inputs)

    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="lstm_backbone")(x)
    x = keras.layers.Dropout(dropout, name="lstm_backbone_dropout")(x)

    attn_out = keras.layers.MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=key_dim,
        dropout=dropout,
        name="self_attention",
    )(x, x)

    x = keras.layers.Add(name="attn_residual")([x, attn_out])
    x = keras.layers.LayerNormalization(name="attn_norm")(x)

    x = keras.layers.LSTM(lstm_units, return_sequences=False, name="lstm_head")(x)
    x = keras.layers.Dropout(dropout, name="lstm_head_dropout")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_head")(x)
    x = keras.layers.Dropout(dropout, name="dense_head_dropout")(x)

    outputs = keras.layers.Dense(5, name="output_5day_forecast")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="attention_lstm_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


class DenormalizedMetricsCallback(keras.callbacks.Callback):
    """
    Compute denormalized validation metrics after each epoch.

    Args:
        x_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels in normalized space.
        y_val_min (np.ndarray): Per-sample target minimum for inverse scaling.
        y_val_denom (np.ndarray): Per-sample target scale denominator for inverse scaling.
        batch_size (int): Prediction batch size.
    """

    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        y_val_min: np.ndarray,
        y_val_denom: np.ndarray,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._x_val = x_val
        self._y_val = y_val
        self._y_val_min = y_val_min
        self._y_val_denom = y_val_denom
        self._batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """
        Add denormalized MAE/RMSE into training logs.

        Args:
            epoch (int): Current epoch index.
            logs (dict[str, float] | None): Mutable logs dictionary.
        """
        _ = epoch
        if logs is None:
            logs = {}

        if self.model is None:
            return

        y_pred = self.model.predict(self._x_val, batch_size=self._batch_size, verbose=0)

        y_true_denorm = self._y_val * self._y_val_denom + self._y_val_min
        y_pred_denorm = y_pred * self._y_val_denom + self._y_val_min

        denorm_mae = float(np.mean(np.abs(y_true_denorm - y_pred_denorm)))
        denorm_rmse = float(np.sqrt(np.mean((y_true_denorm - y_pred_denorm) ** 2)))

        logs["val_denorm_mae"] = denorm_mae
        logs["val_denorm_rmse"] = denorm_rmse

        print(f" - val_denorm_mae: {denorm_mae:.4f} - val_denorm_rmse: {denorm_rmse:.4f}")


def main() -> None:
    """
    Parse arguments, train the Attention + LSTM model, and save best checkpoint.
    """
    parser = argparse.ArgumentParser(description="使用 Attention+LSTM 與 12 特徵訓練模型")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="TFRecord 檔案路徑")
    parser.add_argument("--epochs", type=int, default=50, help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
    parser.add_argument("--window-size", type=int, default=10, help="滑動視窗天數")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="每檔股票時序驗證比例")
    parser.add_argument("--lstm-units", type=int, default=64, help="LSTM 隱藏層單元數")
    parser.add_argument("--attention-heads", type=int, default=4, help="注意力頭數")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout 比例")
    parser.add_argument("--no-cache", action="store_true", help="不使用快取，強制重算")
    args = parser.parse_args()

    tf_path = Path(args.tfrecord_path)
    if not tf_path.exists():
        print(f"找不到檔案: {tf_path}")
        return

    print("====================================")
    print(f"Keras 後端: {keras.config.backend()}")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

    x_train, x_val, y_train, y_val, y_val_min, y_val_denom = load_dataset(
        str(tf_path),
        window_size=args.window_size,
        use_cache=not args.no_cache,
        val_ratio=args.val_ratio,
    )

    model = build_model(
        input_shape=(args.window_size, 12),
        lstm_units=args.lstm_units,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    )
    model.summary()

    model_name = "institutional_net_buy_v3_lstm_attention.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_name,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        DenormalizedMetricsCallback(
            x_val=x_val,
            y_val=y_val,
            y_val_min=y_val_min,
            y_val_denom=y_val_denom,
            batch_size=args.batch_size,
        ),
    ]

    print("\n開始訓練 Attention+LSTM...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
    )

    print(f"模型訓練完成。最佳權重已儲存至: {model_name}")


if __name__ == "__main__":
    main()
