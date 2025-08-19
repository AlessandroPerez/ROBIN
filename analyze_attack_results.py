#!/usr/bin/env python3
"""
Analyze ROBIN results by attack type
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_by_attack_type(results_file):
    """Analyze results breakdown by attack type"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Group by attack type
    watermarked_by_attack = defaultdict(list)
    clean_by_attack = defaultdict(list)
    
    for result in results:
        attack_name = result["attack_name"]
        detection_results = result["detection_results"]
        
        if result["image_type"] == "watermarked":
            watermarked_by_attack[attack_name].append({
                'detected': detection_results["watermark_detected"],
                'attribution_correct': detection_results["attribution_correct"],
                'confidence': detection_results["confidence"],
                'detection_rate': detection_results["detection_rate"],
                'f1_score': detection_results["f1_score"]
            })
        else:  # clean
            clean_by_attack[attack_name].append({
                'detected': detection_results["watermark_detected"],
                'confidence': detection_results["confidence"],
                'detection_rate': detection_results["detection_rate"]
            })
    
    print("="*80)
    print("🔍 ROBIN ATTACK ANALYSIS - WATERMARK SURVIVAL RATES")
    print("="*80)
    
    # Analyze watermarked images
    print("\n🎯 WATERMARKED IMAGE PERFORMANCE BY ATTACK:")
    print("-" * 60)
    
    attack_performance = []
    
    for attack_name in sorted(watermarked_by_attack.keys()):
        results_list = watermarked_by_attack[attack_name]
        
        detection_rate = sum(r['detected'] for r in results_list) / len(results_list)
        attribution_rate = sum(r['attribution_correct'] for r in results_list) / len(results_list)
        avg_confidence = sum(r['confidence'] for r in results_list) / len(results_list)
        avg_f1 = sum(r['f1_score'] for r in results_list) / len(results_list)
        
        attack_performance.append((attack_name, detection_rate, attribution_rate, avg_confidence, avg_f1))
        
        print(f"{attack_name:15s}: Detection={detection_rate:5.1%}, Attribution={attribution_rate:5.1%}, "
              f"Confidence={avg_confidence:.3f}, F1={avg_f1:.3f}")
    
    print("\n🔍 CLEAN IMAGE PERFORMANCE BY ATTACK:")
    print("-" * 60)
    
    for attack_name in sorted(clean_by_attack.keys()):
        results_list = clean_by_attack[attack_name]
        
        false_positive_rate = sum(r['detected'] for r in results_list) / len(results_list)
        avg_confidence = sum(r['confidence'] for r in results_list) / len(results_list)
        
        print(f"{attack_name:15s}: False Positive={false_positive_rate:5.1%}, "
              f"Confidence={avg_confidence:.3f}")
    
    # Rank attacks by difficulty
    print("\n📊 ATTACK DIFFICULTY RANKING (Hardest to Easiest for Watermarks):")
    print("-" * 60)
    
    attack_performance.sort(key=lambda x: x[1])  # Sort by detection rate
    
    for i, (attack_name, detection_rate, attribution_rate, avg_confidence, avg_f1) in enumerate(attack_performance):
        difficulty = "🔴 HARD" if detection_rate < 0.4 else "🟡 MEDIUM" if detection_rate < 0.7 else "🟢 EASY"
        print(f"{i+1:2d}. {attack_name:15s}: {detection_rate:5.1%} {difficulty}")
    
    # Attack category analysis
    print("\n🏷️  ATTACK CATEGORY ANALYSIS:")
    print("-" * 60)
    
    category_stats = defaultdict(lambda: {'detections': [], 'attributions': []})
    
    for attack_name, detection_rate, attribution_rate, _, _ in attack_performance:
        if 'blur' in attack_name:
            category = 'BLUR'
        elif 'jpeg' in attack_name:
            category = 'JPEG'
        elif 'noise' in attack_name:
            category = 'NOISE'
        else:
            category = 'OTHER'
        
        category_stats[category]['detections'].append(detection_rate)
        category_stats[category]['attributions'].append(attribution_rate)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        avg_detection = sum(stats['detections']) / len(stats['detections'])
        avg_attribution = sum(stats['attributions']) / len(stats['attributions'])
        
        print(f"{category:8s}: Avg Detection={avg_detection:5.1%}, Avg Attribution={avg_attribution:5.1%}")
    
    print("\n" + "="*80)
    
    return attack_performance, category_stats

if __name__ == "__main__":
    results_file = "final_robin_results/actual_robin_detailed.json"
    
    if Path(results_file).exists():
        analyze_by_attack_type(results_file)
    else:
        print(f"❌ Results file not found: {results_file}")
