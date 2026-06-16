#!/usr/bin/env python3
"""Test script to verify load_dataset() produces correct label shapes for multi-step prediction"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, 'engine/institutional_net_buy')

import tensorflow as tf
from institutional_net_buy_ml import load_dataset

print("=" * 80)
print("CREATING SAMPLE TFRECORD FOR TESTING")
print("=" * 80)

# Create sample data: 10 days of data for testing
sample_data = []
for day in range(10):
    features = [float(day)] * 10  # 10 features per day
    closing_price = 100.0 + day  # Simple incrementing prices
    sample_data.append((features, closing_price))

# Create temporary TFRecord file
with tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False) as f:
    tfrecord_path = f.name

print(f"Writing {len(sample_data)} days of sample data to {tfrecord_path}")

# Write TFRecord
with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for features, closing_price in sample_data:
        # Create simplified example (just features and close price)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "close": tf.train.Feature(float_list=tf.train.FloatList(value=[closing_price])),
                    "foreign_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    "foreign_dealer_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    "trust_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    "dealer_hedge_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    "dealer_net_buy": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                    "foreign_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "foreign_dealer_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "trust_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "dealer_hedge_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "dealer_cost": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "date": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"2026-05-01"])),
                    "stock_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"2330"])),
                    "stock_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"TSMC"])),
                }
            )
        )
        writer.write(example.SerializeToString())

print(f"✓ Sample TFRecord created with {len(sample_data)} examples\n")

print("=" * 80)
print("TESTING LOAD_DATASET() WITH window_size=1 (MULTI-STEP PREDICTION)")
print("=" * 80)

# Test load_dataset with window_size=1 (should use 6-day sliding window for 5-day labels)
dataset = load_dataset(tfrecord_path, batch_size=2, epochs=1, window_size=1)

# Take first few batches
print(f"\nDataset created. Taking first 2 batches to inspect shapes...\n")

batch_count = 0
for features_batch, labels_batch in dataset.take(2):
    batch_count += 1
    print(f"Batch {batch_count}:")
    print(f"  Features shape: {features_batch.shape}")
    print(f"  Labels shape: {labels_batch.shape}")
    print(f"  Expected features shape: (batch_size, 1, 10)")
    print(f"  Expected labels shape: (batch_size, 5)")
    
    # Verify shapes
    features_check = len(features_batch.shape) == 3 and features_batch.shape[1] == 1 and features_batch.shape[2] == 10
    labels_check = len(labels_batch.shape) == 2 and labels_batch.shape[1] == 5
    
    print(f"  Features shape correct: {features_check}")
    print(f"  Labels shape correct: {labels_check}")
    
    # Show label values to verify they're consecutive closing prices
    print(f"  Sample label values (should be [101, 102, 103, 104, 105] pattern): {labels_batch[0].numpy()}")
    print()

print("=" * 80)
print("CLEANUP")
print("=" * 80)
Path(tfrecord_path).unlink()
print(f"✓ Cleaned up temporary file: {tfrecord_path}")
