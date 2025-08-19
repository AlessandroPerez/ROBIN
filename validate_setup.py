#!/usr/bin/env python3
"""
ROBIN Testing Pipeline Validation Script
========================================

This script validates that all dependencies and files are properly set up
for running the comprehensive testing pipeline.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_conda_environment():
    """Check if running in correct conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"Conda environment: {conda_env}")
    
    if conda_env == 'robin':
        print("✅ Correct conda environment")
        return True
    else:
        print("❌ Please activate robin environment: conda activate robin")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'torchvision', 
        'diffusers',
        'transformers',
        'datasets',
        'PIL',
        'cv2',
        'sklearn',
        'numpy',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages installed")
        return True

def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
            print(f"   Current device: {torch.cuda.get_device_name()}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'prompts_1000.json',
        'comprehensive_testing_pipeline.py',
        'optim_utils.py',
        'inverse_stable_diffusion.py',
        'stable_diffusion_robin.py',
        'io_utils.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_prompts_file():
    """Validate prompts file format"""
    try:
        with open('prompts_1000.json', 'r') as f:
            data = json.load(f)
        
        if 'prompts' in data and isinstance(data['prompts'], list):
            num_prompts = len(data['prompts'])
            print(f"✅ Prompts file valid - {num_prompts} prompts")
            
            if num_prompts >= 50:
                print(f"✅ Sufficient prompts for testing")
                return True
            else:
                print(f"⚠️  Only {num_prompts} prompts - may limit testing")
                return True
        else:
            print("❌ Invalid prompts file format")
            return False
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error reading prompts file: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        stat = os.statvfs('.')
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"Available disk space: {available_gb:.1f} GB")
        
        if available_gb >= 5.0:
            print("✅ Sufficient disk space")
            return True
        else:
            print("⚠️  Low disk space - may need more for image storage")
            return True
    except:
        print("⚠️  Could not check disk space")
        return True

def run_quick_test():
    """Run a quick functionality test"""
    print("\nRunning quick functionality test...")
    
    try:
        # Test basic imports
        import torch
        from diffusers import DPMSolverMultistepScheduler
        import PIL.Image
        import numpy as np
        
        # Test basic tensor operations
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_tensor = torch.randn(1, 3, 64, 64).to(device)
        result = torch.nn.functional.relu(test_tensor)
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("ROBIN Testing Pipeline Validation")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Conda Environment", check_conda_environment), 
        ("Required Packages", check_required_packages),
        ("PyTorch & CUDA", check_torch_cuda),
        ("Required Files", check_required_files),
        ("Prompts File", check_prompts_file),
        ("Disk Space", check_disk_space),
        ("Quick Test", run_quick_test)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * len(check_name))
        
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    
    if all_passed:
        print("🎉 All checks passed! Ready to run the testing pipeline.")
        print("\nTo start testing, run:")
        print("bash run_testing.sh")
    else:
        print("❌ Some checks failed. Please fix the issues above before running.")
        print("\nCommon solutions:")
        print("1. Activate environment: conda activate robin")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run setup: bash setup_environment.sh")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
