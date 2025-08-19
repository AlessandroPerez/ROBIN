#!/usr/bin/env python3
"""
Minimal ROBIN test - tests core functionality without complex dependencies
"""

import sys
import os
import json

def test_core_imports():
    """Test core imports needed for ROBIN"""
    print("Testing core imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        from PIL import Image
        print("✅ PIL")
        
        # Test ROBIN core files
        if os.path.exists('stable_diffusion_robin.py'):
            print("✅ stable_diffusion_robin.py found")
        else:
            print("❌ stable_diffusion_robin.py missing")
            return False
            
        if os.path.exists('optim_utils.py'):
            print("✅ optim_utils.py found")
        else:
            print("❌ optim_utils.py missing")
            return False
            
        if os.path.exists('inverse_stable_diffusion.py'):
            print("✅ inverse_stable_diffusion.py found")
        else:
            print("❌ inverse_stable_diffusion.py missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_prompts():
    """Test prompts file"""
    print("\nTesting prompts file...")
    try:
        with open('prompts_1000.json', 'r') as f:
            data = json.load(f)
        
        if 'prompts' in data:
            prompts = data['prompts']
            print(f"✅ {len(prompts)} prompts loaded")
            
            # Show first 5 prompts
            print("\nFirst 5 prompts:")
            for i, prompt in enumerate(prompts[:5]):
                print(f"  {i+1}. {prompt[:50]}...")
            
            return prompts[:5]
        else:
            print(f"❌ No 'prompts' key found in file")
            return None
    except Exception as e:
        print(f"❌ Prompts error: {e}")
        return None

def test_comprehensive_pipeline():
    """Test if our comprehensive pipeline can be imported"""
    print("\nTesting comprehensive pipeline...")
    try:
        # Try to import our test file
        if os.path.exists('comprehensive_testing_pipeline.py'):
            print("✅ comprehensive_testing_pipeline.py found")
            
            # Try to parse it (basic syntax check)
            with open('comprehensive_testing_pipeline.py', 'r') as f:
                content = f.read()
            
            # Check for key classes
            if 'class WatermarkTester' in content:
                print("✅ WatermarkTester class found")
            else:
                print("❌ WatermarkTester class missing")
                return False
                
            if 'class ImageAttacks' in content:
                print("✅ ImageAttacks class found")
            else:
                print("❌ ImageAttacks class missing")
                return False
                
            if 'class AttackConfig' in content:
                print("✅ AttackConfig class found")
            else:
                print("❌ AttackConfig class missing")
                return False
            
            print("✅ Comprehensive pipeline structure looks correct")
            return True
        else:
            print("❌ comprehensive_testing_pipeline.py missing")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_results_format():
    """Test if we have a results.json file to match format"""
    print("\nTesting results format...")
    try:
        if os.path.exists('results.json'):
            with open('results.json', 'r') as f:
                results = json.load(f)
            
            print(f"✅ Results format template found")
            
            # Check structure
            if isinstance(results, dict):
                attack_count = len([k for k in results.keys() if k.startswith('attack_')])
                print(f"✅ {attack_count} attack configurations found")
                
                # Show some attack types
                attack_types = set()
                for key, value in results.items():
                    if key.startswith('attack_') and isinstance(value, dict):
                        attack_type = value.get('attack_type', 'unknown')
                        attack_types.add(attack_type)
                
                print(f"✅ Attack types: {', '.join(sorted(attack_types))}")
                return True
            else:
                print("❌ Results format structure unexpected")
                return False
        else:
            print("❌ results.json missing")
            return False
            
    except Exception as e:
        print(f"❌ Results format test failed: {e}")
        return False

def create_simple_test_script():
    """Create a simple test script that can run with minimal dependencies"""
    print("\nCreating simplified test script...")
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    with open('simple_robin_test.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created simple_robin_test.py")
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("ROBIN PIPELINE VALIDATION - MINIMAL TEST")
    print("=" * 60)
    
    success = True
    
    # Test 1: Core imports
    if not test_core_imports():
        success = False
    
    # Test 2: Prompts
    prompts = test_prompts()
    if prompts is None:
        success = False
    
    # Test 3: Pipeline structure
    if not test_comprehensive_pipeline():
        success = False
    
    # Test 4: Results format
    if not test_results_format():
        success = False
    
    # Create simple test script
    create_simple_test_script()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ BASIC VALIDATION PASSED")
        print("\nNEXT STEPS:")
        print("1. Install missing packages: pip install diffusers transformers")
        print("2. Run: python simple_robin_test.py")
        print("3. For full testing: python comprehensive_testing_pipeline.py --num_images 5")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\nPlease check the issues above before proceeding.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
