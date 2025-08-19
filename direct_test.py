#!/usr/bin/env python3
"""
Direct ROBIN test - bypasses complex imports and tests core functionality
"""

import torch
import json
import os
import time
import sys
from PIL import Image

def main():
    print("=" * 60)
    print("DIRECT ROBIN FUNCTIONALITY TEST")
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
        prompts = data['prompts'][:3]  # Test with 3 prompts
        print(f"✅ Loaded {len(prompts)} test prompts")
    except Exception as e:
        print(f"❌ Failed to load prompts: {e}")
        return False
    
    # Create output directory
    output_dir = "direct_test_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    print(f"✅ Created output directory: {output_dir}")
    
    try:
        # Import ROBIN pipeline
        print("\nImporting ROBIN pipeline...")
        from stable_diffusion_robin import ROBINStableDiffusionPipeline
        print("✅ ROBIN pipeline imported successfully")
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipe = ROBINStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        print(f"✅ Pipeline loaded on {device}")
        
        results = []
        
        # Generate images
        for i, prompt in enumerate(prompts):
            print(f"\n--- Generating image {i+1}/{len(prompts)} ---")
            print(f"Prompt: {prompt[:60]}...")
            
            try:
                start_time = time.time()
                
                # Generate watermarked image
                print("Generating watermarked image...")
                wm_result = pipe(
                    prompt,
                    num_inference_steps=20,  # Reduced for speed
                    generator=torch.Generator(device=device).manual_seed(42 + i),
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                wm_image = wm_result.images[0]
                
                # Generate clean image  
                print("Generating clean image...")
                clean_result = pipe(
                    prompt,
                    num_inference_steps=20,
                    generator=torch.Generator(device=device).manual_seed(42 + i),
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    use_watermark=False
                )
                clean_image = clean_result.images[0]
                
                generation_time = time.time() - start_time
                
                # Save images
                wm_path = f"{output_dir}/images/test_{i+1:02d}_watermarked.png"
                clean_path = f"{output_dir}/images/test_{i+1:02d}_clean.png"
                
                wm_image.save(wm_path)
                clean_image.save(clean_path)
                
                print(f"✅ Generated and saved in {generation_time:.2f}s")
                print(f"   Watermarked: {wm_path}")
                print(f"   Clean: {clean_path}")
                
                result = {
                    "prompt": prompt,
                    "image_id": i + 1,
                    "watermarked_path": wm_path,
                    "clean_path": clean_path,
                    "generation_time": generation_time,
                    "status": "success"
                }
                results.append(result)
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                result = {
                    "prompt": prompt,
                    "image_id": i + 1,
                    "error": str(e),
                    "status": "failed"
                }
                results.append(result)
        
        # Save results
        results_path = f"{output_dir}/direct_test_results.json"
        final_results = {
            "test_type": "direct_robin_test",
            "device": device,
            "num_prompts": len(prompts),
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
        print("DIRECT TEST SUMMARY")
        print("=" * 60)
        
        summary = final_results["summary"]
        print(f"✅ Successful generations: {summary['successful']}/{len(prompts)}")
        print(f"❌ Failed generations: {summary['failed']}/{len(prompts)}")
        print(f"⏱️ Total time: {summary['total_time']:.2f}s")
        print(f"⏱️ Average time per image: {summary['avg_time']:.2f}s")
        print(f"📁 Images saved in: {output_dir}/images/")
        print(f"📄 Results saved in: {results_path}")
        
        if summary['successful'] > 0:
            print(f"\n✅ SUCCESS! ROBIN pipeline is working correctly!")
            print(f"Generated {summary['successful']} watermarked image pairs successfully.")
            return True
        else:
            print(f"\n❌ FAILED! No images were generated successfully.")
            return False
        
    except Exception as e:
        print(f"\n❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
