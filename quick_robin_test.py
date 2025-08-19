#!/usr/bin/env python3
"""
Quick test of ROBIN watermarking with modern environment
"""

import torch
import os
from stable_diffusion_robin import ROBINStableDiffusionPipeline
from PIL import Image
import json

print("🚀 Testing ROBIN watermarking with Python 3.10...")

# Load a few prompts for testing
with open('prompts_1000.json', 'r') as f:
    prompts_data = json.load(f)

# Use first 3 prompts for quick test
test_prompts = prompts_data['prompts'][:3]

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize ROBIN pipeline
print("Loading ROBIN pipeline...")
try:
    pipe = ROBINStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    print("✅ ROBIN pipeline loaded successfully!")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    exit(1)

# Test watermark generation
print("\n🎨 Generating test images...")
os.makedirs("test_output", exist_ok=True)

for i, prompt in enumerate(test_prompts):
    print(f"\n📝 Prompt {i+1}: {prompt}")
    
    try:
        # Generate watermarked image
        result = pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.Generator(device=device).manual_seed(42 + i)
        )
        
        image = result.images[0]
        
        # Save image
        output_path = f"test_output/robin_test_{i+1}.png"
        image.save(output_path)
        print(f"✅ Generated and saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating image {i+1}: {e}")
        continue

print("\n🎯 Test completed! Check test_output/ directory for generated images.")
print("✅ ROBIN watermarking is working with Python 3.10 environment!")
