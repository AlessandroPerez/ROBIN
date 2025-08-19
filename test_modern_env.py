#!/usr/bin/env python3
"""
Test script for ROBIN watermarking with modern Python 3.10 environment
"""

import torch
import os
import sys
import numpy as np
from PIL import Image

print("🔧 Testing ROBIN with Python 3.10 environment...")
print(f"✅ Python version: {sys.version}")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

try:
    import diffusers
    print(f"✅ Diffusers version: {diffusers.__version__}")
except ImportError as e:
    print(f"❌ Diffusers import failed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")
    sys.exit(1)

# Test if we can import the ROBIN modules
try:
    from stable_diffusion_robin import ROBINStableDiffusionPipeline
    print("✅ ROBIN pipeline imported successfully")
except ImportError as e:
    print(f"❌ ROBIN pipeline import failed: {e}")
    print("This might be due to compatibility issues with newer diffusers version")
    print("Let's try to diagnose...")
    
    # Try importing diffusers components
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline imported successfully")
    except ImportError as e:
        print(f"❌ StableDiffusionPipeline import failed: {e}")
    
    try:
        from diffusers import UNet2DConditionModel
        print("✅ UNet2DConditionModel imported successfully")
    except ImportError as e:
        print(f"❌ UNet2DConditionModel import failed: {e}")
    
    # Let's see what's available in diffusers
    print("\n📋 Available diffusers modules:")
    import diffusers
    for attr in sorted(dir(diffusers)):
        if not attr.startswith('_'):
            print(f"   - {attr}")

print("\n🎯 Testing basic PyTorch functionality...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# Test basic tensor operations
x = torch.randn(2, 3, 256, 256).to(device)
print(f"✅ Created tensor on {device}: {x.shape}")

print("\n✅ Environment test completed!")
print("\nNext steps:")
print("1. Check ROBIN code compatibility with diffusers 0.35.0")
print("2. Update imports if necessary")
print("3. Run watermarking test")
