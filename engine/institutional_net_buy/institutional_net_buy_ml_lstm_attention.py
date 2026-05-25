"""
Institutional Net Buy Attention LSTM Training Module (升級版：報酬率與成本乖離架構)

This module trains an Attention + LSTM model with 11 engineered features
(6 institutional signals + 5 cost deviation features) to forecast the next 5-day log returns (in %).

Key features:
- Per-stock temporal split to prevent time-series leakage
- Train-only min-max scaling for features; Global clipping for return targets
- Direct return MAE and RMSE metrics for precise trading interpretability
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
    "foreign": 1.0,
    "foreign_dealer": 1.0,
    "trust": 0.75,
    "dealer": 0.5,
    "dealer_hedge": 0.5,
    "retail": -0.2,
}

# 升級快取架構版本號
CACHE_SCHEMA_VERSION = "v6_lstm_attention_zscore_return_no_price_slope"


@keras.saving.register_keras_serializable(package="CustomLoss", name="quantile_loss_75")
def quantile_loss_75(y_true, y_pred):
    """
    Quantile loss for q=0.75, registered for model serialization.
    """
    q = 0.75
    error = y_true - y_pred
    return keras.ops.mean(keras.ops.maximum(q * error, (q - 1.0) * error), axis=-1)


def _build_loss(loss_name: str, quantile: float):
    if loss_name == "mse":
        return keras.losses.MeanSquaredError()
    if loss_name == "huber":
        return keras.losses.Huber(delta=1.0)
    if loss_name == "quantile":
        if abs(quantile - 0.75) < 1e-6:
            return quantile_loss_75
        def q_loss(y_true, y_pred):
            error = y_true - y_pred
            return keras.ops.mean(keras.ops.maximum(quantile * error, (quantile - 1.0) * error), axis=-1)
        return q_loss
    return keras.losses.MeanSquaredError()


def _tfrecord_fingerprint(tfrecord_path: str) -> dict[str, Any]:
    """
    Build lightweight TFRecord fingerprint for cache validation.
    """
    stat = os.stat(tfrecord_path)
    return {
        "path": str(Path(tfrecord_path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def get_slope(y: np.ndarray) -> float:
    """
    Compute linear slope in a rolling window.
    """
    if len(y) < 2:
        return 0.0
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.polyfit(np.arange(len(y_clean)), y_clean, 1)[0])


def _cache_paths(tfrecord_path: str, window_size: int) -> dict[str, str]:
    """
    Build cache file paths for the Attention + LSTM pipeline.
    window_size is embedded in filenames so different window sizes never share cache.
    """
    cache_base = tfrecord_path.replace(".tfrecord", "")
    return {
        "x_train": f"{cache_base}_lstmattn_w{window_size}_X_train.npy",
        "x_val": f"{cache_base}_lstmattn_w{window_size}_X_val.npy",
        "y_train": f"{cache_base}_lstmattn_w{window_size}_y_train.npy",
        "y_val": f"{cache_base}_lstmattn_w{window_size}_y_val.npy",
        "stats": f"{cache_base}_lstmattn_w{window_size}.stats.json",
    }


def _is_cache_valid(
    paths: dict[str, str],
    use_cache: bool,
    tfrecord_path: str,
    window_size: int,
    val_ratio: float,
) -> bool:
    """
    Validate whether cache files are present and schema-compatible.
    """
    if not use_cache:
        return False

    required = [
        paths["x_train"],
        paths["x_val"],
        paths["y_train"],
        paths["y_val"],
        paths["stats"],
    ]
    if not all(os.path.exists(p) for p in required):
        return False

    try:
        with open(paths["stats"], "r", encoding="utf-8") as f:
            stats = json.load(f)
        meta = stats.get("_meta", {})
        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False

        if int(meta.get("window_size", -1)) != int(window_size):
            return False

        if float(meta.get("val_ratio", -1.0)) != float(val_ratio):
            return False

        expected_fp = _tfrecord_fingerprint(tfrecord_path)
        cached_fp = meta.get("tfrecord_fingerprint", {})
        return (
            cached_fp.get("path") == expected_fp["path"]
            and int(cached_fp.get("size", -1)) == expected_fp["size"]
            and int(cached_fp.get("mtime_ns", -1)) == expected_fp["mtime_ns"]
        )
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
    Load TFRecord and build leakage-safe train/validation datasets with return targets.
    """
    if not (0.05 <= val_ratio <= 0.5):
        raise ValueError("val_ratio 必須在 0.05 到 0.5 之間。")

    paths = _cache_paths(tfrecord_path, window_size)

    if _is_cache_valid(paths, use_cache, tfrecord_path, window_size, val_ratio):
        print("找到有效快取，正在載入升級版 Attention+LSTM 特徵資料...")
        return (
            np.load(paths["x_train"]),
            np.load(paths["x_val"]),
            np.load(paths["y_train"]),
            np.load(paths["y_val"]),
        )

    raw_ds = tf.data.TFRecordDataset(tfrecord_path, buffer_size=1024*1024*100)  # 100MB buffer for efficiency
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

    print("正在讀取 TFRecord 並計算 11 個特徵信號...")
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
    # 將價格斜率除以收盤價，轉為「百分比斜率」，確保高低價股基準統一
    # 【核心更新 1】成本特徵乖離化 (去除價格絕對值)
    cost_cols = [
        "foreign_cost",
        "trust_cost",
        "dealer_cost",
        "dealer_hedge_cost",
        "foreign_dealer_cost",
    ]
    for c in cost_cols:
        df[c] = (df[c] - df["close"]) / (df["close"] + 1e-8)

    feature_cols = [
        "w_buy",
        "sig_total",
        "sig_pos_ratio",
        "sig_slope",
        "sig_baseline",
        "sig_price_pct",
    ] + cost_cols

    # 【核心更新 2】預測目標改為對數報酬率百分比 (%)
    target_cols = [f"return_d+{i}" for i in range(1, 6)]
    for i in range(1, 6):
        forward_price = df.groupby("stock_id")["close"].shift(-i)
        valid_price_mask = (forward_price > 0.0) & (df["close"] > 0.0)
        safe_ratio = pd.Series(np.nan, index=df.index, dtype="float64")
        safe_ratio.loc[valid_price_mask] = (
            forward_price.loc[valid_price_mask] / df["close"].loc[valid_price_mask]
        )
        df[f"return_d+{i}"] = np.log(safe_ratio) * 100.0

    before_drop = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dropped_rows = before_drop - len(df)

    if df.empty:
        raise ValueError("特徵工程後無剩餘資料，可能是資料長度不足以計算滾動視窗。")

    print("正在執行每支股票獨立的時序切分與特徵 MinMax 標準化...")
    print(f"特徵工程清洗後共移除 {dropped_rows} 筆資料。")

    stats: dict[str, Any] = {
        "_meta": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "window_size": window_size,
            "val_ratio": val_ratio,
            "tfrecord_fingerprint": _tfrecord_fingerprint(tfrecord_path),
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "dropped_rows_after_engineering": dropped_rows,
        }
    }

    x_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    x_val_list: list[np.ndarray] = []
    y_val_list: list[np.ndarray] = []

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

        # 【核心更新 3】限制報酬率目標在台股波段極限內，完全移除 Targets 的 MinMaxScaler 縮放
        g_train[target_cols] = g_train[target_cols].clip(lower=-50.0, upper=70.0)
        g_val[target_cols] = g_val[target_cols].clip(lower=-50.0, upper=70.0)

        # 升級：更換為 Z-Score Normalization
        # Z-Score 不會鎖定上下界，對噴發行情（離群值）更具耐受力
        s_mean = g_train[feature_cols].mean()
        s_std = g_train[feature_cols].std().replace(0, 1.0) # 防止除以零

        g_train[feature_cols] = (g_train[feature_cols] - s_mean) / s_std
        # 驗證集使用訓練集標準進行標準化，並加上寬鬆 Clip 防止極端值
        s_mean_arr = s_mean.values
        s_std_arr = s_std.values
        g_val[feature_cols] = (g_val[feature_cols].to_numpy() - s_mean_arr) / s_std_arr
        g_val[feature_cols] = np.clip(g_val[feature_cols], -10.0, 10.0) # Z-Score 截斷

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

        sid_str = str(sid)
        stats[sid_str] = {
            "mean": s_mean.tolist(),
            "std": s_std.tolist(),
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

    print(f"略過資料不足的股票數: {skipped_small_groups}")
    print(f"略過切分無法建窗的股票數: {skipped_invalid_split}")
    print(f"訓練樣本數: {len(x_train)}, 驗證樣本數: {len(x_val)}")

    with open(paths["stats"], "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"統計值已儲存至: {paths['stats']}")

    if use_cache:
        print("正在儲存優化後的快取檔案...")
        np.save(paths["x_train"], x_train)
        np.save(paths["x_val"], x_val)
        np.save(paths["y_train"], y_train)
        np.save(paths["y_val"], y_val)

    return x_train, x_val, y_train, y_val


def build_model(
    input_shape: tuple[int, int],
    lstm_units: int = 64,
    attention_heads: int = 4,
    dropout: float = 0.2,
    loss_name: str = "huber",
    quantile: float = 0.75,
) -> keras.Model:
    """
    Build an Attention + LSTM forecasting model.
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
        loss=_build_loss(loss_name=loss_name, quantile=quantile),
        metrics=["mae"],
    )
    return model


class ReturnMetricsCallback(keras.callbacks.Callback):
    """
    【核心更新 4】簡化指標計算。
    因為預測目標直接就是報酬率百分比，不再需要逆標準化，直接計算真實的 MAE 與 RMSE 百分比誤差。
    """

    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int,
        every_n_epochs: int = 1,
        val_fraction: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self._x_val = x_val
        self._y_val = y_val
        self._batch_size = batch_size
        self._every_n_epochs = max(1, int(every_n_epochs))
        self._val_fraction = float(np.clip(val_fraction, 0.05, 1.0))
        self._random_seed = int(random_seed)

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        if logs is None:
            logs = {}
        if self.model is None:
            return

        if (epoch + 1) % self._every_n_epochs != 0:
            return

        x_eval = self._x_val
        y_eval = self._y_val
        if self._val_fraction < 1.0:
            sample_size = max(1, int(len(self._x_val) * self._val_fraction))
            rng = np.random.default_rng(self._random_seed + int(epoch) + 1)
            sample_idx = rng.choice(len(self._x_val), size=sample_size, replace=False)
            x_eval = self._x_val[sample_idx]
            y_eval = self._y_val[sample_idx]

        y_pred = self.model.predict(x_eval, batch_size=self._batch_size, verbose=0)

        # 這裡算出的數值直接代表「相差幾個百分點 (%)」
        return_mae = float(np.mean(np.abs(y_eval - y_pred)))
        return_rmse = float(np.sqrt(np.mean((y_eval - y_pred) ** 2)))

        logs["val_return_mae_pct"] = return_mae
        logs["val_return_rmse_pct"] = return_rmse

        print(f" - val_return_mae: {return_mae:.4f}% - val_return_rmse: {return_rmse:.4f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Attention+LSTM 與 11 乖離特徵訓練報酬率模型")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="TFRecord 檔案路徑")
    parser.add_argument("--epochs", type=int, default=50, help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
    parser.add_argument("--window-size", type=int, default=20, help="滑動視窗天數")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="每檔股票時序驗證比例")
    parser.add_argument("--lstm-units", type=int, default=64, help="LSTM 隱藏層單元數")
    parser.add_argument("--attention-heads", type=int, default=4, help="注意力頭數")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout 比例")
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber", "quantile"], help="訓練 loss 函數")
    parser.add_argument("--quantile", type=float, default=0.75, help="quantile loss 參數")
    parser.add_argument("--metrics-every", type=int, default=3, help="每隔 N 個 epoch 才計算一次 return 指標")
    parser.add_argument("--metrics-val-fraction", type=float, default=0.3, help="計算 return 指標時使用的驗證集抽樣比例 (0.05~1.0)")
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

    x_train, x_val, y_train, y_val = load_dataset(
        str(tf_path),
        window_size=args.window_size,
        use_cache=not args.no_cache,
        val_ratio=args.val_ratio,
    )

    model = build_model(
        input_shape=(args.window_size, 11),
        lstm_units=args.lstm_units,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        loss_name=args.loss,
        quantile=args.quantile,
    )
    model.summary()

    model_name = "institutional_net_buy_v3_lstm_attention.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            model_name,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReturnMetricsCallback(
            x_val=x_val,
            y_val=y_val,
            batch_size=args.batch_size,
            every_n_epochs=args.metrics_every,
            val_fraction=args.metrics_val_fraction,
        ),
    ]

    print("\n開始訓練 Attention+LSTM...")
    
    # 使用 tf.data.Dataset 构建高性能 Shuffle 管道
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=min(len(x_train), 50000)) \
                       .batch(args.batch_size) \
                       .prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
                     .batch(args.batch_size) \
                     .prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"模型訓練完成。最佳權重已儲存至: {model_name}")


if __name__ == "__main__":
    main()