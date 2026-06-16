#!/usr/bin/env python3
"""Test script to verify build_model() produces correct architecture"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
sys.path.insert(0, 'engine/institutional_net_buy')

from institutional_net_buy_ml import build_model

print("=" * 80)
print("VERIFYING MODEL ARCHITECTURE WITH INPUT_SHAPE=(1, 10)")
print("=" * 80)
model = build_model((1, 10))
model.summary()

print("\n" + "=" * 80)
print("LAYER COUNT AND VERIFICATION")
print("=" * 80)
print(f"Total number of layers: {len(model.layers)}")
print(f"Expected: 18 (InputLayer + 8 Conv1D + 8 Dropout + GRU + Dropout + Dense)")

# Count layer types
conv_count = sum(1 for layer in model.layers if 'Conv1D' in layer.__class__.__name__)
dropout_count = sum(1 for layer in model.layers if 'Dropout' in layer.__class__.__name__)
gru_count = sum(1 for layer in model.layers if 'GRU' in layer.__class__.__name__)
dense_count = sum(1 for layer in model.layers if 'Dense' in layer.__class__.__name__)

print(f"\nLayer breakdown:")
print(f"  - Conv1D layers: {conv_count} (expected: 8)")
print(f"  - Dropout layers: {dropout_count} (expected: 9: 8 after conv + 1 before output)")
print(f"  - GRU layers: {gru_count} (expected: 1)")
print(f"  - Dense layers: {dense_count} (expected: 1)")
print(f"  - Output shape: {model.output_shape} (expected: (None, 5))")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
checks = [
    ("Conv1D count", conv_count == 8, f"{conv_count} == 8"),
    ("Dropout count", dropout_count == 9, f"{dropout_count} == 9"),
    ("GRU count", gru_count == 1, f"{gru_count} == 1"),
    ("Dense count", dense_count == 1, f"{dense_count} == 1"),
    ("Output shape", model.output_shape == (None, 5), f"{model.output_shape} == (None, 5)"),
]

all_pass = True
for check_name, result, details in checks:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status}: {check_name} ({details})")
    all_pass = all_pass and result

print("\n" + "=" * 80)
if all_pass:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED!")
print("=" * 80)
