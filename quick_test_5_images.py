#!/usr/bin/env python3
"""
Quick 5-image test for ROBIN watermarking
This script tests the basic functionality with minimal dependencies
"""

import os
import json
import time
import sys

def load_prompts():
    """Load prompts from file"""
    try:
        with open('prompts_1000.json', 'r') as f:
            data = json.load(f)
        return data['prompts'][:5]  # Get first 5 prompts
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return None

def test_basic_setup():
    """Test basic setup"""
    print("=" * 60)
    print("ROBIN 5-IMAGE VALIDATION TEST")
    print("=" * 60)
    
    # Check files
    required_files = [
        'stable_diffusion_robin.py',
        'optim_utils.py', 
        'inverse_stable_diffusion.py',
        'comprehensive_testing_pipeline.py',
        'prompts_1000.json'
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
    
    # Test imports
    print(f"\nTesting imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available, will use CPU")
        
        from PIL import Image
        print("✅ PIL")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Load prompts
    prompts = load_prompts()
    if prompts is None:
        return False
    
    print(f"✅ Loaded {len(prompts)} test prompts")
    
    return True, prompts

def run_comprehensive_test(prompts):
    """Run the comprehensive testing pipeline with 5 images"""
    print(f"\nRunning comprehensive test with 5 images...")
    
    try:
        # Import the comprehensive pipeline
        from comprehensive_testing_pipeline import WatermarkTester
        
        print("✅ Successfully imported WatermarkTester")
        
        # Create test configuration
        config = {
            'num_images': 5,
            'output_dir': 'validation_test_results',
            'model_name': 'stabilityai/stable-diffusion-2-1-base',
            'save_images': True,
            'run_attacks': True
        }
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        print(f"Created output directory: {config['output_dir']}")
        
        # Initialize tester
        print("Initializing ROBIN watermark tester...")
        tester = WatermarkTester(config)
        
        print("✅ WatermarkTester initialized")
        
        # Run test with 5 prompts
        print(f"\nStarting generation and testing with {len(prompts)} prompts...")
        start_time = time.time()
        
        results = tester.run_comprehensive_test(prompts)
        
        total_time = time.time() - start_time
        
        print(f"✅ Test completed in {total_time:.2f} seconds")
        
        # Save results
        results_path = os.path.join(config['output_dir'], 'validation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {results_path}")
        
        # Print summary
        print(f"\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Total images generated: {summary.get('total_images', 0)}")
            print(f"Successful generations: {summary.get('successful_generations', 0)}")
            print(f"Failed generations: {summary.get('failed_generations', 0)}")
            print(f"Total attacks tested: {summary.get('total_attacks_tested', 0)}")
            print(f"Average generation time: {summary.get('avg_generation_time', 0):.2f}s")
        
        print(f"\nImages saved in: {config['output_dir']}/images/")
        print(f"Results saved in: {results_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    # Basic setup test
    setup_result = test_basic_setup()
    if isinstance(setup_result, tuple):
        success, prompts = setup_result
    else:
        success = setup_result
        prompts = None
    
    if not success:
        print("\n❌ Basic setup failed. Please check the issues above.")
        return False
    
    # Run comprehensive test
    if prompts:
        if run_comprehensive_test(prompts):
            print(f"\n✅ ALL TESTS PASSED! ROBIN pipeline is working correctly.")
            return True
        else:
            print(f"\n❌ Comprehensive test failed.")
            return False
    else:
        print(f"\n❌ No prompts available for testing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
