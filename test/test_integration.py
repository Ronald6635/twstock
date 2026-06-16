#!/usr/bin/env python3
"""Integration test: verify model can train with the multi-step dataset pipeline"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, 'engine/institutional_net_buy')

import tensorflow as tf
from institutional_net_buy_ml import build_model, load_dataset

print("=" * 80)
print("INTEGRATION TEST: MODEL TRAINING WITH MULTI-STEP DATASET")
print("=" * 80)

# Create sample data: 20 days of data for testing
print("\nCreating sample TFRecord with 20 days of data...")
with tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False) as f:
    tfrecord_path = f.name

with tf.io.TFRecordWriter(tfrecord_path) as writer:
    for day in range(20):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "close": tf.train.Feature(float_list=tf.train.FloatList(value=[100.0 + day])),
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

print(f"✓ Sample TFRecord created with 20 days of data\n")

print("=" * 80)
print("BUILDING MODEL")
print("=" * 80)
model = build_model(input_shape=(1, 10))
print(f"Model built successfully")
print(f"Output shape: {model.output_shape}\n")

print("=" * 80)
print("CREATING DATASET")
print("=" * 80)
dataset = load_dataset(tfrecord_path, batch_size=4, epochs=1, window_size=1)
print(f"✓ Dataset created\n")

print("=" * 80)
print("TRAINING MODEL FOR 2 EPOCHS (INTEGRATION TEST)")
print("=" * 80)

try:
    # Train for 2 iterations to verify no shape mismatches
    history = model.fit(dataset, epochs=2, verbose=1)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final loss: {history.history['loss'][-1]:.6f}")
    print(f"Final MAE: {history.history['mae'][-1]:.6f}")
    
    print("\n✓ INTEGRATION TEST PASSED")
    print("  - Model instantiation: OK")
    print("  - Dataset pipeline: OK")
    print("  - Training: OK (no shape mismatches)")
    
except Exception as e:
    print(f"\n✗ INTEGRATION TEST FAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    Path(tfrecord_path).unlink()
    print(f"✓ Cleaned up temporary file")
