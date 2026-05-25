"""
使用 TFRecord 進行機器學習訓練的範例程式碼 (升級版：報酬率預測架構)
================================================================
這個程式建立了一個 4 層 Dilated CNN 模型，使用 11 個特徵 (6 個籌碼信號 + 5 個成本乖離率) 
來預測未來 5 天的對數報酬率 (放大 100 倍，單位為百分比 %)。
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


@keras.saving.register_keras_serializable(package="CustomLoss", name="quantile_loss_75")
def quantile_loss_75(y_true, y_pred):
    """
    Quantile loss for q=0.75, registered for model serialization.
    """
    q = 0.75
    error = y_true - y_pred
    return keras.ops.mean(keras.ops.maximum(q * error, (q - 1.0) * error), axis=-1)


def _build_loss(loss_name: str, quantile: float):
    """
    Build a configurable training loss.

    Args:
        loss_name: One of "mse", "huber", "quantile".
        quantile: Quantile parameter used when loss_name is "quantile".

    Returns:
        Keras loss object/callable.
    """
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
            # Pinball loss encourages directional robustness on asymmetric tails.
            return keras.ops.mean(keras.ops.maximum(q * error, (q - 1.0) * error), axis=-1)

        return quantile_loss
    raise ValueError(f"不支援的 loss: {loss_name}")

def get_slope(y):
    """計算滾動視窗的線性斜率。"""
    if len(y) < 2: return 0.0
    # 確保值是有效的
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.polyfit(np.arange(len(y_clean)), y_clean, 1)[0])

def _make_windows(feature_values: np.ndarray, label_values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rolling windows from features and labels.
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
    """
    split_idx = int(length * (1.0 - val_ratio))
    split_idx = max(split_idx, window_size)
    split_idx = min(split_idx, length - window_size)
    return split_idx


def _transform_return_targets(
    target_values: pd.DataFrame,
    target_transform: str,
    clip_lower: float,
    clip_upper: float,
    tanh_scale: float,
) -> pd.DataFrame:
    """
    Transform return targets for stable training.

    Args:
        target_values: Raw log-return targets (percent).
        target_transform: One of "clip", "tanh", "none".
        clip_lower: Lower bound used by clip transform.
        clip_upper: Upper bound used by clip transform.
        tanh_scale: Scale used by tanh saturation transform.

    Returns:
        Transformed targets in the same shape as input.
    """
    if target_transform == "clip":
        return target_values.clip(lower=clip_lower, upper=clip_upper)

    if target_transform == "tanh":
        target_array = target_values.to_numpy(dtype="float32")
        transformed_array = np.tanh(target_array / tanh_scale) * tanh_scale
        return pd.DataFrame(
            transformed_array,
            index=target_values.index,
            columns=target_values.columns,
        )

    if target_transform == "none":
        return target_values

    raise ValueError(f"不支援的 target_transform: {target_transform}")


def _cache_matches_target_config(
    stats_path: str,
    target_transform: str,
    clip_lower: float,
    clip_upper: float,
    tanh_scale: float,
    expected_feature_cols: list[str],
) -> bool:
    """
    Validate whether existing cache files were generated with same target config.
    """
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
    except Exception:
        return False

    if not stats_data:
        return False

    first_key = next(iter(stats_data.keys()))
    first_stat = stats_data.get(first_key, {})
    cached_transform = first_stat.get("target_transform", "clip")
    cached_clip_lower = float(first_stat.get("target_clip_lower", -50.0))
    cached_clip_upper = float(first_stat.get("target_clip_upper", 70.0))
    cached_tanh_scale = float(first_stat.get("target_tanh_scale", 50.0))
    cached_feature_cols = first_stat.get("columns", [])

    if cached_transform != target_transform:
        return False

    if abs(cached_clip_lower - clip_lower) > 1e-8:
        return False
    if abs(cached_clip_upper - clip_upper) > 1e-8:
        return False
    if abs(cached_tanh_scale - tanh_scale) > 1e-8:
        return False

    if cached_feature_cols != expected_feature_cols:
        return False

    return True


def load_dataset(
    tfrecord_path: str,
    window_size: int = 10,
    use_cache: bool = True,
    val_ratio: float = 0.2,
    target_transform: str = "tanh",
    clip_lower: float = -50.0,
    clip_upper: float = 70.0,
    tanh_scale: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TFRecord data and build leakage-safe train/validation sets with return targets.
    """
    cache_base = tfrecord_path.replace(".tfrecord", "")
    x_train_cache = f"{cache_base}_train_X.npy"
    x_val_cache = f"{cache_base}_val_X.npy"
    y_train_cache = f"{cache_base}_train_y.npy"
    y_val_cache = f"{cache_base}_val_Y.npy"
    stats_path = f"{cache_base}.leakage_fixed.stats.json"

    has_all_cache_files = (
        os.path.exists(x_train_cache)
        and os.path.exists(x_val_cache)
        and os.path.exists(y_train_cache)
        and os.path.exists(y_val_cache)
        and os.path.exists(stats_path)
    )
    if use_cache and has_all_cache_files:
        if _cache_matches_target_config(
            stats_path=stats_path,
            target_transform=target_transform,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            tanh_scale=tanh_scale,
            expected_feature_cols=[
                "w_buy",
                "sig_total",
                "sig_pos_ratio",
                "sig_slope",
                "sig_baseline",
                "sig_price_pct",
                "foreign_cost",
                "trust_cost",
                "dealer_cost",
                "dealer_hedge_cost",
                "foreign_dealer_cost",
            ],
        ):
            print("找到有效快取檔案，正在載入預先計算好的時序特徵...")
            return (
                np.load(x_train_cache),
                np.load(x_val_cache),
                np.load(y_train_cache),
                np.load(y_val_cache),
            )
        print("偵測到快取與當前 target 參數不一致，將重新計算特徵。")

    raw_ds = tf.data.TFRecordDataset(tfrecord_path, buffer_size=1024*1024*100)  # 100MB buffer
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

    print(f"正在讀取 TFRecord 並計算 11 個特徵信號...")
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

    # 計算散戶動向與綜合買超力道 [cite: 243]
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
    # 【升級：特徵定常化】將絕對價格的法人成本線，轉為與當日收盤價的相對乖離率 [cite: 2, 276]
    cost_cols = ["foreign_cost", "trust_cost", "dealer_cost", "dealer_hedge_cost", "foreign_dealer_cost"]
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

    # 【升級：目標值計算】計算未來 d+1 到 d+5 天相對於當天的對數報酬率，並乘以 100 轉為百分比單位 
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

        if split_idx < window_size or (n_rows - split_idx) < window_size:
            skipped += 1
            continue

        g_train = g.iloc[:split_idx].copy().reset_index(drop=True)
        g_val = g.iloc[split_idx:].copy().reset_index(drop=True)

        # 標籤轉換：可在 clip/tanh/none 間切換，降低尾部噪音同時保留較多振幅資訊。
        g_train[target_cols] = _transform_return_targets(
            target_values=g_train[target_cols],
            target_transform=target_transform,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            tanh_scale=tanh_scale,
        )
        g_val[target_cols] = _transform_return_targets(
            target_values=g_val[target_cols],
            target_transform=target_transform,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
            tanh_scale=tanh_scale,
        )

        # 【升級：更換為 Z-Score Normalization】
        # Z-Score 不會鎖定上下界，對噴發行情（離群值）更具耐受力
        s_mean = g_train[feature_cols].mean()
        s_std = g_train[feature_cols].std().clip(lower=1e-4) # 避免除以零

        # 訓練集標準化，且clip於 mean ± 10*std 範圍內，保留極端行情資訊同時減少離群值影響
        g_train[feature_cols] = ((g_train[feature_cols] - s_mean) / s_std).clip(-10, 10)
        
        # 驗證集「跟隨」訓練集標準，且不再嚴格 Clip
        # 只有在 Z-Score 極端到離譜時 (例如 10 個標準差以上) 才微調，保留絕大部分行情資訊
        s_mean_arr = s_mean.to_numpy()
        s_std_arr = s_std.to_numpy()
        g_val[feature_cols] = (g_val[feature_cols].to_numpy() - s_mean_arr) / s_std_arr

        x_tr, y_tr = _make_windows(g_train[feature_cols].values, g_train[target_cols].values, window_size)
        x_va, y_va = _make_windows(g_val[feature_cols].values, g_val[target_cols].values, window_size)

        if len(x_tr) == 0 or len(x_va) == 0:
            skipped += 1
            continue

        x_train_list.append(x_tr)
        y_train_list.append(y_tr)
        x_val_list.append(x_va)
        y_val_list.append(y_va)

        # 寫入 json 統計字典（記錄 mean/std 以備推論使用）
        stats[str(sid)] = {
            "mean": s_mean.tolist(),
            "std": s_std.tolist(),
            "target_transform": target_transform,
            "target_clip_lower": float(clip_lower),
            "target_clip_upper": float(clip_upper),
            "target_tanh_scale": float(tanh_scale),
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

def build_model(input_shape: tuple, loss_name: str = "mse", quantile: float = 0.75):
    """
    建立 4 層 Dilated CNN 模型。
    輸出層為 Dense(5) 一次預測未來 5 天的報酬率百分比。

    容量設計原則：
    - 64 filters（非 128）：避免在跨股 MinMax 混合資料上過擬合
    - Dropout 0.3（非 0.2）：增強正則化，補償 LayerNorm 無正則化效果
    - 不在 Dilated Conv 加 LayerNorm：小批次 / 短序列時 LN 易加速過擬合
    - 僅在最終 1x1 Conv 後加單層 LayerNorm：穩定 feature aggregation
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    dilations = [1, 2, 4, 8]
    for i, d in enumerate(dilations):
        model.add(keras.layers.Conv1D(
            32,
            kernel_size=3, activation='relu',
            padding='causal', dilation_rate=d,
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f'dilated_conv_{i+1}'
        ))
        model.add(keras.layers.Dropout(0.4, name=f'dilated_drop_{i+1}'))

    model.add(keras.layers.Conv1D(
        32,
        kernel_size=1, activation='relu',
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name='final_conv'
    ))
    model.add(keras.layers.LayerNormalization(name='final_conv_ln'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.GlobalAveragePooling1D(name='global_avg_pool'))

    model.add(keras.layers.Dense(32, activation='relu', name='hidden'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(5, name='output_5day_forecast'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=_build_loss(loss_name=loss_name, quantile=quantile),
        metrics=['mae']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="使用 12 特徵與 4 層 Dilated CNN 訓練報酬率模型")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="TFRecord 檔案路徑")
    parser.add_argument("--epochs", type=int, default=50, help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=512, help="批次大小 (建議使用 256 以上)")
    parser.add_argument("--window-size", type=int, default=10, help="滑動視窗天數 (搭配 Dilated CNN 建議 >= 10)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="每支股票驗證集比例")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber", "quantile"], help="訓練 loss 函數")
    parser.add_argument("--quantile", type=float, default=0.75, help="quantile loss 參數，僅在 --loss quantile 時生效")
    parser.add_argument("--target-transform", type=str, default="tanh", choices=["clip", "tanh", "none"], help="目標值轉換策略")
    parser.add_argument("--clip-lower", type=float, default=-50.0, help="clip 下界，僅在 --target-transform clip 時主要生效")
    parser.add_argument("--clip-upper", type=float, default=70.0, help="clip 上界，僅在 --target-transform clip 時主要生效")
    parser.add_argument("--tanh-scale", type=float, default=50.0, help="tanh 飽和尺度，僅在 --target-transform tanh 時生效")
    parser.add_argument("--no-cache", action="store_true", help="不使用快取，強制重新計算特徵")
    args = parser.parse_args()

    if not (0.0 < args.quantile < 1.0):
        raise ValueError("--quantile 必須在 (0, 1) 範圍內")
    if not (args.clip_lower < args.clip_upper):
        raise ValueError("--clip-lower 必須小於 --clip-upper")
    if args.tanh_scale <= 0.0:
        raise ValueError("--tanh-scale 必須大於 0")

    tf_path = Path(args.tfrecord_path)
    if not tf_path.exists():
        print(f"找不到檔案: {tf_path}")
        return

    print("====================================")
    print(f"Keras 後端: {keras.config.backend()}")
    print(f"Loss 設定: {args.loss}, quantile={args.quantile:.2f}")
    print(
        "Target 設定: "
        f"transform={args.target_transform}, "
        f"clip=({args.clip_lower:.1f}, {args.clip_upper:.1f}), "
        f"tanh_scale={args.tanh_scale:.1f}"
    )
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 載入資料 (包含快取與特徵工程)
    X_train, X_val, y_train, y_val = load_dataset(
        str(tf_path), 
        window_size=args.window_size,
        use_cache=not args.no_cache,
        val_ratio=args.val_ratio,
        target_transform=args.target_transform,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        tanh_scale=args.tanh_scale,
    )
    print(f"訓練樣本數: {len(X_train)}, 驗證樣本數: {len(X_val)}")
    
    # 2. 建立模型 (11 個特徵)
    model = build_model(
        input_shape=(args.window_size, 11),
        loss_name=args.loss,
        quantile=args.quantile,
    )
    model.summary()
    
    # 4. 設定回調函數 (Callbacks)
    model_name = "institutional_net_buy_v2_dilated.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,  # 5 → 10，給模型更多時間
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5, 
            patience=5,   # 3 → 5
            min_lr=1e-6,  # 新增：防止 LR 降到無效值
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_name,
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 5. 訓練
    print("\n開始訓練...")

    # 使用 tf.data.Dataset 以實現更好的 Shuffle 與效能
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=min(len(X_train), 50000)) \
                       .batch(args.batch_size) \
                       .prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
                     .batch(args.batch_size) \
                     .prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 快速驗證訊號品質：回報 D+1 方向命中率（在目前 target 空間）。
    val_preds = model.predict(X_val, verbose=0)
    val_direction_acc = float(np.mean(np.sign(val_preds[:, 0]) == np.sign(y_val[:, 0])))
    print(f"驗證集 D+1 方向命中率: {val_direction_acc:.4f}")
    
    print(f"模型訓練完成。最佳權重已儲存至: {model_name}")

if __name__ == "__main__":
    main()