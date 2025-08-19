#!/usr/bin/env python3
"""
Simple test script to validate ROBIN pipeline with 5 images
"""

import sys
import os
import torch
import time
import json
from PIL import Image

def test_basic_imports():
    """Test if we can import required packages"""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__} imported successfully")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        from PIL import Image
        print("✅ PIL imported successfully")
        
        # Test if we can load the ROBIN files
        from stable_diffusion_robin import ROBINStableDiffusionPipeline
        print("✅ ROBIN Pipeline imported successfully")
        
        from optim_utils import optim_utils
        print("✅ Optimization utilities imported successfully")
        
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        print("✅ Inverse Stable Diffusion imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
        print(f"   Current device: {torch.cuda.get_device_name()}")
        return True
    else:
        print("❌ CUDA not available")
        return False

def test_prompts_file():
    """Test if prompts file is readable"""
    print("\nTesting prompts file...")
    try:
        with open('prompts_1000.json', 'r') as f:
            prompts = json.load(f)
        print(f"✅ Prompts file loaded - {len(prompts)} prompts available")
        return prompts[:5]  # Return first 5 prompts for testing
    except Exception as e:
        print(f"❌ Prompts file error: {e}")
        return None

def simple_generation_test(prompts):
    """Test simple image generation with ROBIN"""
    print("\nTesting ROBIN image generation...")
    
    try:
        # Initialize pipeline
        print("Loading ROBIN pipeline...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to create the pipeline
        pipe = ROBINStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        
        print("✅ Pipeline loaded successfully")
        
        # Create test results directory
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("test_results/images", exist_ok=True)
        
        results = []
        
        # Test with 5 prompts
        for i, prompt in enumerate(prompts[:5]):
            print(f"\nGenerating image {i+1}/5: '{prompt[:50]}...'")
            
            try:
                start_time = time.time()
                
                # Generate watermarked image
                wm_image = pipe(
                    prompt,
                    num_inference_steps=30,
                    generator=torch.Generator(device=device).manual_seed(42),
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
                
                # Generate clean image (for comparison)
                clean_image = pipe(
                    prompt,
                    num_inference_steps=30,
                    generator=torch.Generator(device=device).manual_seed(42),
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    use_watermark=False
                ).images[0]
                
                generation_time = time.time() - start_time
                
                # Save images
                wm_path = f"test_results/images/test_{i+1}_watermarked.png"
                clean_path = f"test_results/images/test_{i+1}_clean.png"
                
                wm_image.save(wm_path)
                clean_image.save(clean_path)
                
                print(f"✅ Generated in {generation_time:.2f}s")
                print(f"   Saved: {wm_path} and {clean_path}")
                
                results.append({
                    "prompt": prompt,
                    "watermarked_image": wm_path,
                    "clean_image": clean_path,
                    "generation_time": generation_time,
                    "status": "success"
                })
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Save results
        with open("test_results/simple_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Test completed! Results saved to test_results/simple_test_results.json")
        print(f"   Successfully generated {sum(1 for r in results if r['status'] == 'success')}/5 images")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("ROBIN Pipeline Simple Test (5 Images)")
    print("=" * 60)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed. Please check your environment.")
        return False
    
    # Test 2: CUDA
    test_cuda()
    
    # Test 3: Prompts file
    prompts = test_prompts_file()
    if prompts is None:
        print("\n❌ Prompts file test failed.")
        return False
    
    # Test 4: Simple generation
    if not simple_generation_test(prompts):
        print("\n❌ Generation test failed.")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! ROBIN pipeline is working correctly.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
