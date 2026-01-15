"""
Virtual Environment Cleanup Utility

Removes corrupted package metadata directories that block imports.
Safe to run repeatedly without side effects.
"""

import os
import shutil
from pathlib import Path
from typing import List

def find_corrupted_packages(venv_path: str) -> List[Path]:
    """
    Find corrupted package directories with tilde prefixes.
    
    Args:
        venv_path: Path to virtual environment directory
        
    Returns:
        List of corrupted dist-info paths
    """
    site_packages = Path(venv_path) / "Lib" / "site-packages"
    
    if not site_packages.exists():
        print(f"ERROR: site-packages not found at {site_packages}")
        return []
    
    corrupted: List[Path] = []
    for item in site_packages.iterdir():
        # Match tilde-prefixed dist-info directories
        if item.is_dir() and item.name.startswith("~") and item.name.endswith(".dist-info"):
            corrupted.append(item)
    
    return corrupted

def cleanup_venv(venv_path: str) -> None:
    """
    Remove all corrupted package metadata from virtual environment.
    
    Args:
        venv_path: Path to virtual environment directory
    """
    corrupted_packages = find_corrupted_packages(venv_path)
    
    if not corrupted_packages:
        print("✓ No corrupted packages found. Virtual environment is clean.")
        return
    
    print(f"Found {len(corrupted_packages)} corrupted package(s):")
    for pkg_path in corrupted_packages:
        print(f"  - {pkg_path.name}")
    
    print("\nRemoving corrupted packages...")
    for pkg_path in corrupted_packages:
        try:
            shutil.rmtree(pkg_path)
            print(f"✓ Removed: {pkg_path.name}")
        except Exception as e:
            print(f"✗ Failed to remove {pkg_path.name}: {e}")
    
    print("\n✓ Cleanup complete!")

if __name__ == "__main__":
    import sys
    
    venv_path = sys.argv[1] if len(sys.argv) > 1 else ".venv"
    cleanup_venv(venv_path)