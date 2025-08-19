#!/usr/bin/env python3
"""
Simplified ROBIN test script
Run this after installing packages: pip install torch torchvision diffusers transformers
"""

import torch
import json
import os
from PIL import Image

def main():
    print("=== SIMPLIFIED ROBIN TEST ===")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA not available - will use CPU")
    
    # Load prompts
    with open('prompts_1000.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"✅ Loaded {len(prompts)} prompts")
    
    # Test with first prompt
    test_prompt = prompts[0]
    print(f"Test prompt: {test_prompt}")
    
    # Create test directory
    os.makedirs("simple_test_results", exist_ok=True)
    
    # Here you would add the actual ROBIN pipeline testing
    # For now, just create a placeholder result
    result = {
        "test_prompt": test_prompt,
        "status": "setup_complete",
        "message": "Environment ready for ROBIN testing"
    }
    
    with open("simple_test_results/test_log.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("✅ Simple test completed - check simple_test_results/test_log.json")

if __name__ == "__main__":
    main()
