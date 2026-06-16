"""
GPU Configuration Verification Module

This module provides diagnostic tools to verify that the deep learning
environment correctly detects and utilizes NVIDIA GPU hardware via 
the Keras 3 and PyTorch backend stack.

Key features:
- Backend configuration verification
- CUDA availability check
- GPU hardware metadata reporting
- Functional execution test on GPU device
"""

import os
# CRITICAL: Backend must be set before importing any Keras components
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
from typing import Dict, Any, Union, Optional

# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def verify_gpu_status() -> Dict[str, Union[str, bool, float]]:
    """
    Perform a comprehensive check of the GPU environment.
    
    This function queries both the Keras configuration and the underlying 
    PyTorch CUDA state to ensure hardware acceleration is active.
    
    Returns:
        Dictionary containing:
            - 'backend' (str): The active Keras backend name.
            - 'is_cuda_active' (bool): Whether CUDA is detectable by Torch.
            - 'device_name' (str): The name of the primary GPU device.
            - 'total_memory_gb' (float): Total VRAM available in Gigabytes.
            
    Example:
        >>> results = verify_gpu_status()
        >>> if results['is_cuda_active']:
        ...     print(f"Running on {results['device_name']}")
    """
    status: Dict[str, Any] = {
        "backend": keras.config.backend(),
        "is_cuda_active": False,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
        "cuda_version": None,
        "error": None,
    }

    try:
        status["is_cuda_active"] = torch.cuda.is_available()
        if status["is_cuda_active"]:
            status["device_name"] = torch.cuda.get_device_name(0)
            status["cuda_version"] = torch.version.cuda or "unknown"
            # Convert bytes -> GiB using 1024**3 for accuracy
            status["total_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
    except Exception as exc:  # pragma: no cover - diagnostic helper
        status["error"] = str(exc)

    return status

# =============================================================================
# REFINED DIAGNOSTIC EXECUTION
# =============================================================================

def execute_gpu_test(results: Dict[str, Any]) -> None:
    """
    Execute a functional deep learning diagnostic.
    
    This function creates a minimal neural network to verify that tensors 
    are correctly dispatched to the GPU device by the Torch backend.
    
    Args:
        results (Dict[str, Any]): Metadata dictionary from verify_gpu_status()
        
    Returns:
        None
        
    Note:
        Success is indicated by a non-crashing forward pass and the 
        presence of a torch.Tensor output.
    """
    print(f"Keras Backend: {results['backend']}")
    print(f"PyTorch CUDA Available: {results['is_cuda_active']}")

    if results.get("is_cuda_active"):
        print(f"GPU Device: {results.get('device_name')}")
        print(f"CUDA Runtime: {results.get('cuda_version')}")
        print(f"GPU Memory: {results.get('total_memory_gb'):.2f} GiB")
    else:
        print("WARNING: CUDA is not detected. Training will be slow on CPU.")

    # Extra system diagnostic (best-effort)
    try:  # pragma: no cover - informational
        import subprocess

        completed = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if completed.returncode == 0:
            print("\n--- nvidia-smi output (snippet) ---")
            print("\n".join(completed.stdout.splitlines()[:12]))
            print("--- end snippet ---\n")
    except Exception:
        pass

    # Determine device and create tensors on it explicitly
    device = torch.device("cuda:0" if results.get("is_cuda_active") else "cpu")
    X_test: torch.Tensor = torch.randn(10, 5, device=device)

    # Pure PyTorch functional check (most reliable indicator of GPU use)
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(5, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).to(device)

    with torch.no_grad():
        torch_out = torch_model(X_test)

    print(f"PyTorch model output device: {torch_out.device}, shape: {torch_out.shape}")

    # Keras test: try using the same input (may require CPU numpy fallback)
    model: keras.Sequential = keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')

    try:
        # Attempt to pass the torch tensor (backend='torch' may accept it)
        prediction = model(X_test)
        print(f"\nKeras Model Test Output Shape: {prediction.shape}")
    except Exception:
        # Fallback: pass CPU numpy array to Keras
        prediction = model(X_test.cpu().numpy())
        print(f"\nKeras Model (CPU fallback) Output Shape: {prediction.shape}")

    print("✓ Diagnostic complete!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("GPU Setup Verification")
    print("=" * 50)
    
    diagnostic_results = verify_gpu_status()
    execute_gpu_test(diagnostic_results)