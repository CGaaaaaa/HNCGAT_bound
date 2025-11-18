#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment verification script for HNCGAT
Run this script to verify all required packages are installed correctly
"""

import sys

def check_package(name, import_name=None, version_attr='__version__'):
    """Check if a package is installed and print version"""
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, 'unknown')
        print(f"✓ {name} {version}")
        return True
    except ImportError as e:
        print(f"✗ {name} not installed: {e}")
        return False

def main():
    print("=" * 60)
    print("HNCGAT Environment Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9) or sys.version_info >= (3, 10):
        print("⚠ Warning: Python 3.9.13 is recommended")
    print()
    
    # Check PyTorch
    print("Checking PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU count: {torch.cuda.device_count()}")
        else:
            print("  ⚠ CUDA not available, will use CPU (slower)")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        all_ok = False
    print()
    
    # Check other packages
    print("Checking other dependencies...")
    all_ok &= check_package("NumPy", "numpy")
    all_ok &= check_package("SciPy", "scipy")
    all_ok &= check_package("scikit-learn", "sklearn")
    print()
    
    # Check if HNCGAT modules can be imported
    print("Checking HNCGAT modules...")
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from utils import readListfile, get_train_index, calculateauc
        print("✓ utils module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        all_ok = False
    
    try:
        from loss import multi_contrastive_loss
        print("✓ loss module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import loss: {e}")
        all_ok = False
    
    print()
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! Environment is ready.")
        return 0
    else:
        print("✗ Some checks failed. Please install missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

