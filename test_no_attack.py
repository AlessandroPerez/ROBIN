#!/usr/bin/env python3
"""
ROBIN Watermark Detection Test - No Attacks
Tests the core watermark embedding and detection logic without any attacks.
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the actual ROBIN pipeline
try:
    from actual_robin_pipeline import ActualROBINPipeline, ActualROBINDetector, SimpleROBINWatermarkEmbedder
    ROBIN_AVAILABLE = True
    logger.info("✅ ROBIN pipeline components available")
except ImportError as e:
    ROBIN_AVAILABLE = False
    logger.error(f"❌ ROBIN components not available: {e}")

def test_no_attack_detection():
    """Test watermark detection without any attacks"""
    
    if not ROBIN_AVAILABLE:
        logger.error("❌ Cannot run test - ROBIN components not available")
        return
    
    # Initialize pipeline
    logger.info("Initializing ROBIN pipeline...")
    pipeline = ActualROBINPipeline()
    detector = ActualROBINDetector(pipeline.embedder)
    
    # Test prompts
    test_prompts = [
        "A beautiful sunset over a mountain lake",
        "A field of colorful wildflowers in spring", 
        "A serene forest path with dappled sunlight"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"Testing prompt {i+1}/3: {prompt}")
        
        pattern_idx = i % len(pipeline.embedder.watermark_patterns)
        
        # Generate watermarked image
        logger.info("  Generating watermarked image...")
        watermarked_image, watermark_info = pipeline.generate_watermarked_image(
            prompt=prompt,
            pattern_idx=pattern_idx,
            seed=42 + i
        )
        
        # Generate clean image (no watermark)
        logger.info("  Generating clean image...")
        clean_image, _ = pipeline.generate_watermarked_image(
            prompt=prompt,
            pattern_idx=-1,  # No watermark
            seed=42 + i
        )
        
        # Test watermarked image detection
        logger.info("  Testing watermarked image detection...")
        wm_detection = detector.detect_watermark(watermarked_image, pattern_idx)
        
        # Test clean image detection
        logger.info("  Testing clean image detection...")
        clean_detection = detector.detect_watermark(clean_image, None)
        
        # Store results
        test_result = {
            "prompt_id": i,
            "prompt": prompt,
            "expected_pattern": watermark_info['pattern_id'] if watermark_info else None,
            "watermarked_test": {
                "detected": wm_detection["watermark_detected"],
                "detected_pattern": wm_detection["detected_pattern"],
                "confidence": wm_detection["confidence"],
                "attribution_correct": wm_detection["attribution_correct"],
                "all_scores": wm_detection["all_scores"]
            },
            "clean_test": {
                "detected": clean_detection["watermark_detected"],
                "detected_pattern": clean_detection["detected_pattern"],
                "confidence": clean_detection["confidence"],
                "all_scores": clean_detection["all_scores"]
            }
        }
        
        results.append(test_result)
        
        # Print immediate results
        print(f"\n{'='*60}")
        print(f"PROMPT {i+1}: {prompt}")
        print(f"{'='*60}")
        print(f"Expected Pattern: {watermark_info['pattern_id'] if watermark_info else 'None'}")
        print(f"\n🎯 WATERMARKED IMAGE:")
        print(f"   Detected: {'✅ YES' if wm_detection['watermark_detected'] else '❌ NO'}")
        print(f"   Pattern: {wm_detection['detected_pattern']}")
        print(f"   Confidence: {wm_detection['confidence']:.4f}")
        print(f"   Attribution: {'✅ CORRECT' if wm_detection['attribution_correct'] else '❌ WRONG'}")
        
        print(f"\n🔍 CLEAN IMAGE:")
        print(f"   Detected: {'❌ FALSE POSITIVE' if clean_detection['watermark_detected'] else '✅ CORRECT REJECTION'}")
        print(f"   Pattern: {clean_detection['detected_pattern']}")
        print(f"   Confidence: {clean_detection['confidence']:.4f}")
        
        # Show all pattern scores for watermarked image
        print(f"\n📊 ALL PATTERN SCORES (Watermarked):")
        for pattern_id, score in wm_detection['all_scores'].items():
            marker = "👑" if pattern_id == wm_detection['detected_pattern'] else "  "
            expected_marker = "🎯" if pattern_id == (watermark_info['pattern_id'] if watermark_info else None) else "  "
            print(f"   {marker}{expected_marker} {pattern_id}: {score:.4f}")
    
    # Calculate summary statistics
    watermarked_detections = [r["watermarked_test"]["detected"] for r in results]
    clean_detections = [r["clean_test"]["detected"] for r in results]
    correct_attributions = [r["watermarked_test"]["attribution_correct"] for r in results]
    
    wm_detection_rate = sum(watermarked_detections) / len(watermarked_detections)
    clean_false_positive_rate = sum(clean_detections) / len(clean_detections)
    attribution_accuracy = sum(correct_attributions) / len(correct_attributions)
    
    print(f"\n{'='*80}")
    print("📈 SUMMARY STATISTICS (NO ATTACKS)")
    print(f"{'='*80}")
    print(f"🎯 Watermarked Image Detection Rate: {wm_detection_rate:.1%} ({sum(watermarked_detections)}/{len(watermarked_detections)})")
    print(f"🔍 Clean Image False Positive Rate: {clean_false_positive_rate:.1%} ({sum(clean_detections)}/{len(clean_detections)})")
    print(f"🎪 Attribution Accuracy: {attribution_accuracy:.1%} ({sum(correct_attributions)}/{len(correct_attributions)})")
    
    # Evaluation
    print(f"\n{'='*80}")
    print("🔬 EVALUATION")
    print(f"{'='*80}")
    
    if wm_detection_rate >= 0.8:
        print("✅ EXCELLENT: Watermark detection rate >= 80%")
    elif wm_detection_rate >= 0.6:
        print("⚠️  GOOD: Watermark detection rate >= 60%")
    else:
        print("❌ POOR: Watermark detection rate < 60%")
    
    if clean_false_positive_rate <= 0.2:
        print("✅ EXCELLENT: Clean image false positive rate <= 20%")
    elif clean_false_positive_rate <= 0.4:
        print("⚠️  ACCEPTABLE: Clean image false positive rate <= 40%")
    else:
        print("❌ POOR: Clean image false positive rate > 40%")
    
    if attribution_accuracy >= 0.7:
        print("✅ EXCELLENT: Attribution accuracy >= 70%")
    elif attribution_accuracy >= 0.5:
        print("⚠️  ACCEPTABLE: Attribution accuracy >= 50%")
    else:
        print("❌ POOR: Attribution accuracy < 50%")
    
    # Save detailed results
    output_dir = Path("no_attack_test_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "no_attack_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    summary = {
        "watermarked_detection_rate": wm_detection_rate,
        "clean_false_positive_rate": clean_false_positive_rate,
        "attribution_accuracy": attribution_accuracy,
        "total_tests": len(results),
        "detailed_results": results
    }
    
    with open(output_dir / "no_attack_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    return summary

def main():
    """Main function"""
    try:
        logger.info("Starting ROBIN No-Attack Detection Test")
        summary = test_no_attack_detection()
        
        if summary:
            print(f"\n✅ Test completed successfully!")
            print(f"📁 Results saved to: no_attack_test_results/")
        else:
            print(f"\n❌ Test failed!")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
