"""
使用訓練好的模型進行預測 (Prediction)
====================================
這個程式示範了如何載入訓練完成的 .keras 模型，
並針對指定的 TFRecord 資料或新資料點進行股價預測。

使用說明
----------------
python institutional_net_buy_predict.py --model-path institutional_net_buy_model.keras --tfrecord-path path/to/your/data.tfrecord
"""

import os
# 設定 Keras 3 以 PyTorch 為後端
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import tensorflow as tf
import keras
import numpy as np
from pathlib import Path

def parse_tfrecord_fn(example):
    """定義 TFRecord 的資料解析格式 (需與訓練時一致)"""
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
    
    # 提取特徵
    f1 = tf.cast(parsed["foreign_net_buy"], tf.float32) / 1000.0
    f2 = tf.cast(parsed["foreign_dealer_net_buy"], tf.float32) / 1000.0
    f3 = tf.cast(parsed["trust_net_buy"], tf.float32) / 1000.0
    f4 = tf.cast(parsed["dealer_hedge_net_buy"], tf.float32) / 1000.0
    f5 = tf.cast(parsed["dealer_net_buy"], tf.float32) / 1000.0
    
    c1 = parsed["foreign_cost"] / 100.0
    c2 = parsed["foreign_dealer_cost"] / 100.0
    c3 = parsed["trust_cost"] / 100.0
    c4 = parsed["dealer_hedge_cost"] / 100.0
    c5 = parsed["dealer_cost"] / 100.0

    features = tf.stack([f1, f2, f3, f4, f5, c1, c2, c3, c4, c5])
    
    return {
        "date": parsed["date"],
        "stock_id": parsed["stock_id"],
        "stock_name": parsed["stock_name"],
        "features": features,
        "actual_close": parsed["close"]
    }

def main():
    parser = argparse.ArgumentParser(description="載入模型進行預測")
    parser.add_argument("--model-path", type=str, default="institutional_net_buy_model.keras", help="模型路徑")
    parser.add_argument("--tfrecord-path", type=str, required=True, help="要預測的 TFRecord 路徑")
    parser.add_argument("--window-size", type=int, default=1, help="滑動視窗大小 (需與訓練時一致)")
    parser.add_argument("--limit", type=int, default=10, help="顯示前 N 筆預測結果")
    args = parser.parse_args()

    # 1. 載入模型
    if not os.path.exists(args.model_path):
        print(f"錯誤：找不到模型檔案 {args.model_path}")
        return
    
    print(f"載入模型中: {args.model_path}...")
    model = keras.models.load_model(args.model_path)
    
    # 2. 準備預測資料集
    raw_dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    # 如果有 window_size，我們手動處理以便對齊資訊
    print(f"進行預測中 (Window Size: {args.window_size})...\n")
    print(f"{'日期':<12} | {'代號':<6} | {'名稱':<8} | {'實際股價':>10} | {'預測股價':>10} | {'誤差'}")
    print("-" * 80)
    
    # 這裡示範如何從 Dataset 中取出資料並餵給模型
    # 注意：若是 window_size > 1，需要收集連續資料
    feature_buffer = []
    info_buffer = []
    
    count = 0
    for item in parsed_dataset:
        feature_buffer.append(item["features"].numpy())
        info_buffer.append({
            "date": item["date"].numpy().decode('utf-8'),
            "stock_id": item["stock_id"].numpy().decode('utf-8'),
            "stock_name": item["stock_name"].numpy().decode('utf-8'),
            "actual": item["actual_close"].numpy()
        })
        
        if len(feature_buffer) == args.window_size:
            # 準備輸入 (Batch=1, TimeSteps=window_size, Features=5)
            X_input = np.array([feature_buffer]) 
            
            # 模型預測
            pred = model.predict(X_input, verbose=0)[0][0]
            
            # 取得最後一天的資訊
            last_info = info_buffer[-1]
            diff = pred - last_info["actual"]
            
            print(f"{last_info['date']:<12} | {last_info['stock_id']:<6} | {last_info['stock_name']:<8} | {last_info['actual']:10.2f} | {pred:10.2f} | {diff:+.2f}")
            
            # 移除最舊的一筆，維持視窗
            feature_buffer.pop(0)
            info_buffer.pop(0)
            
            count += 1
            if count >= args.limit:
                break

if __name__ == "__main__":
    main()
