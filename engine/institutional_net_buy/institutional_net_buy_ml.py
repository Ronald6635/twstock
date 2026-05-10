"""
使用 TFRecord 進行機器學習訓練的範例程式碼
====================================
這個程式示範了如何使用 TensorFlow 的 tf.data API 來讀取 TFRecord 格式的資料，
並在高效率的 Keras 3 + PyTorch 後端環境下，訓練混合 1D CNN 與 GRU 的時序模型來預測收盤價 (close)。

請確保已經使用 institutional_net_buy_fetcher.py 產生了 TFRecord 檔案，並將路徑傳入 --tfrecord-path 參數。

使用說明
----------------
1. 產生 TFRecord 檔案
   請先使用 institutional_net_buy_fetcher.py 來產生 TFRecord檔案，例如：
   python institutional_net_buy_fetcher.py `
       --stock-source list `
       --stocks 2330,2317 `
       --days 90 `
       --export-tfrecord

2. 執行訓練程式
   python institutional_net_buy_ml.py --tfrecord-path path/to/your/data.tfrecord --epochs 20 --batch-size 64
"""

import os
# 設定 Keras 3 以 PyTorch 為後端 (確保在 Windows 順利讀取 GPU)
os.environ["KERAS_BACKEND"] = "torch"

import argparse
from pathlib import Path
import tensorflow as tf
import keras
import torch

def parse_tfrecord_fn(example):
    """定義 TFRecord 的資料解析格式 (與 Fetcher 匯出格式對應)"""
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
        "foreign_cost": tf.io.FixedLenFeature([], tf.float32),
        "foreign_dealer_cost": tf.io.FixedLenFeature([], tf.float32),
        "trust_cost": tf.io.FixedLenFeature([], tf.float32),
        "dealer_hedge_cost": tf.io.FixedLenFeature([], tf.float32),
        "dealer_cost": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    
    # 擷取特徵 (Features) 與目標 (Label)
    # 將五大法人淨買超取出並轉型
    foreign = tf.cast(parsed["foreign_net_buy"], tf.float32)
    foreign_dealer = tf.cast(parsed["foreign_dealer_net_buy"], tf.float32)
    trust = tf.cast(parsed["trust_net_buy"], tf.float32)
    dealer_hedge = tf.cast(parsed["dealer_hedge_net_buy"], tf.float32)
    dealer = tf.cast(parsed["dealer_net_buy"], tf.float32)

    c1 = parsed["foreign_cost"] / 100.0
    c2 = parsed["foreign_dealer_cost"] / 100.0
    c3 = parsed["trust_cost"] / 100.0
    c4 = parsed["dealer_hedge_cost"] / 100.0
    c5 = parsed["dealer_cost"] / 100.0
    
    # 標準化或縮放特徵 (例如除以 1000) 和加入 cost 特徵
    features = tf.stack([
        foreign / 1000.0, 
        foreign_dealer / 1000.0, 
        trust / 1000.0, 
        dealer_hedge / 1000.0, 
        dealer / 1000.0,
        c1, c2, c3, c4, c5
    ])
    
    label = parsed["close"]
    
    return features, label

def load_dataset(tfrecord_path: str, batch_size: int = 32, epochs: int = 1, window_size: int = 1):
    """讀取 TFRecord 並轉為 tf.data.Dataset 交給 Keras 訓練，支援滑動視窗 (Sliding Window)"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # 資料解析與前處理 (解析出 features, label)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if window_size > 1:
        # 實作滑動視窗 (Sliding Window)
        # 1. 將資料打包成視窗。window() 會產生 dataset of datasets
        # 2. flat_map 將其展平。batch(window_size) 會將連續的資料組成一組
        dataset = dataset.window(size=window_size, shift=1, drop_remainder=True)
        
        # 重新結構化：將 [(f1, l1), (f2, l2), ...] 轉為 ([f1, f2, ...], l_target)
        # 這裡我們預測「最後一天」的價格作為 Label
        def pack_window(feat_ds, label_ds):
            features_window = feat_ds.batch(window_size)
            # 取最後一個時間點的 label 作為預測目標
            label_target = label_ds.skip(window_size - 1).take(1)
            return tf.data.Dataset.zip((features_window, label_target))
        
        dataset = dataset.flat_map(pack_window)

    # 隨機打亂與批次處理
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_model(input_shape=(1, 10)):
    """建構混合 1D CNN 與 GRU (Gated Recurrent Unit) 的時序模型
    
    Args:
        input_shape: 格式為 (time_steps, features)。
                     單日預測時為 (1, 10)，十日視窗時為 (10, 10)
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        
        # 1D CNN 層：用於局部特徵萃取 (kernel_size 可捕捉短天期趨勢)
        # 當 time_steps > 1 時，kernel_size=3 可以學習三天的局部相關性
        keras.layers.Conv1D(filters=32, kernel_size=min(3, input_shape[0]), 
                           activation='relu', padding='causal'),
        
        # GRU 層：序列與時序記憶模組
        keras.layers.GRU(64, activation='tanh', return_sequences=False),
        
        # 可以加上 Dropout 防止過擬合
        keras.layers.Dropout(0.2),
        
        # 輸出層：輸出收盤價等連續數值預測
        keras.layers.Dense(1) 
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="使用 TFRecord 進行機器學習訓練")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="TFRecord 檔案路徑")
    parser.add_argument("--epochs", type=int, default=10, help="訓練週期")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小")
    parser.add_argument("--window-size", type=int, default=1, help="滑動視窗大小 (天數)，用來捕捉時序特徵")
    args = parser.parse_args()

    tfrecord_path = Path(args.tfrecord_path)
    if not tfrecord_path.exists():
        print(f"找不到檔案: {tfrecord_path}")
        return

    print("====================================")
    print(f"Keras Backend: {keras.config.backend()}")
    if torch.cuda.is_available():
        print(f"啟動 GPU 訓練 - 偵測到裝置: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未偵測到預期可用的 NVIDIA GPU，將降級使用 CPU 訓練！")
    print(f"載入資料集: {tfrecord_path}")
    print(f"滑動視窗天數: {args.window_size}")
    print("====================================")
    
    # 建立與編譯模型 (input_shape 為 [time_steps, features])
    model = build_model(input_shape=(args.window_size, 10))
    model.summary()
    
    # 建立 Dataset
    train_dataset = load_dataset(
        str(tfrecord_path), 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        window_size=args.window_size
    )

    print("\n開始訓練模型...")
    # 由於在 load_dataset 已經有 .repeat(args.epochs)，我們直接執行 fit
    history = model.fit(
        train_dataset
    )
    
    # 儲存模型
    model_save_path = "institutional_net_buy_model.keras"
    model.save(model_save_path)
    print(f"\n訓練完成！模型已儲存至: {model_save_path}")

if __name__ == "__main__":
    main()