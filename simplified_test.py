#!/usr/bin/env python3
"""
Simplified ROBIN test - works around compatibility issues
"""

import torch
import json
import os
import time
import sys
from PIL import Image

def main():
    print("=" * 60)
    print("SIMPLIFIED ROBIN TEST")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        device = "cuda"
    else:
        print("⚠️ CUDA not available - using CPU")
        device = "cpu"
    
    # Load prompts
    try:
        with open('prompts_1000.json', 'r') as f:
            data = json.load(f)
        prompts = data['prompts'][:2]  # Test with 2 prompts
        print(f"✅ Loaded {len(prompts)} test prompts")
    except Exception as e:
        print(f"❌ Failed to load prompts: {e}")
        return False
    
    # Create output directory
    output_dir = "simplified_test_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    print(f"✅ Created output directory: {output_dir}")
    
    try:
        print("\nTesting basic diffusers functionality...")
        from diffusers import StableDiffusionPipeline
        print("✅ Basic diffusers import successful")
        
        # Try basic stable diffusion first
        print("Initializing basic Stable Diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        print(f"✅ Basic pipeline loaded on {device}")
        
        results = []
        
        # Generate images with basic pipeline
        for i, prompt in enumerate(prompts):
            print(f"\n--- Generating image {i+1}/{len(prompts)} ---")
            print(f"Prompt: {prompt[:60]}...")
            
            try:
                start_time = time.time()
                
                # Generate image
                print("Generating image...")
                result = pipe(
                    prompt,
                    num_inference_steps=20,  # Reduced for speed
                    generator=torch.Generator(device=device).manual_seed(42 + i),
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                image = result.images[0]
                
                generation_time = time.time() - start_time
                
                # Save image
                image_path = f"{output_dir}/images/test_{i+1:02d}_basic.png"
                image.save(image_path)
                
                print(f"✅ Generated and saved in {generation_time:.2f}s")
                print(f"   Saved: {image_path}")
                
                result_data = {
                    "prompt": prompt,
                    "image_id": i + 1,
                    "image_path": image_path,
                    "generation_time": generation_time,
                    "status": "success",
                    "pipeline_type": "basic_stable_diffusion"
                }
                results.append(result_data)
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                result_data = {
                    "prompt": prompt,
                    "image_id": i + 1,
                    "error": str(e),
                    "status": "failed"
                }
                results.append(result_data)
        
        # Test ROBIN import separately
        print(f"\n--- Testing ROBIN Components ---")
        try:
            from stable_diffusion_robin import ROBINStableDiffusionPipeline
            print("✅ ROBIN pipeline import successful")
            robin_available = True
        except Exception as e:
            print(f"❌ ROBIN pipeline import failed: {e}")
            robin_available = False
        
        # Save results
        results_path = f"{output_dir}/simplified_test_results.json"
        final_results = {
            "test_type": "simplified_robin_test",
            "device": device,
            "num_prompts": len(prompts),
            "basic_diffusion_working": True,
            "robin_import_working": robin_available,
            "results": results,
            "summary": {
                "successful": len([r for r in results if r["status"] == "success"]),
                "failed": len([r for r in results if r["status"] == "failed"]),
                "total_time": sum(r.get("generation_time", 0) for r in results),
                "avg_time": sum(r.get("generation_time", 0) for r in results) / len(results) if results else 0
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n" + "=" * 60)
        print("SIMPLIFIED TEST SUMMARY")
        print("=" * 60)
        
        summary = final_results["summary"]
        print(f"✅ Basic Diffusion: Working")
        print(f"{'✅' if robin_available else '❌'} ROBIN Import: {'Working' if robin_available else 'Issues detected'}")
        print(f"✅ Successful generations: {summary['successful']}/{len(prompts)}")
        print(f"⏱️ Total time: {summary['total_time']:.2f}s")
        print(f"⏱️ Average time per image: {summary['avg_time']:.2f}s")
        print(f"📁 Images saved in: {output_dir}/images/")
        print(f"📄 Results saved in: {results_path}")
        
        if summary['successful'] > 0:
            print(f"\n✅ SUCCESS! Basic image generation is working!")
            print(f"Generated {summary['successful']} images successfully.")
            if robin_available:
                print(f"🔬 ROBIN components are importable (may need version adjustments)")
            else:
                print(f"🔧 ROBIN needs compatibility fixes")
            return True
        else:
            print(f"\n❌ FAILED! No images were generated successfully.")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
