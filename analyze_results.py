#!/usr/bin/env python3
"""
ROBIN Results Analysis Script
============================

This script analyzes the comprehensive test results and generates
summary reports and visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def load_results(results_path):
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)

def create_summary_table(results):
    """Create a summary table of all attack results"""
    attack_data = []
    
    for attack_name, attack_results in results.items():
        if attack_name == 'benchmark_info':
            continue
            
        attack_data.append({
            'Attack': attack_name,
            'Type': attack_results['attack_config']['type'],
            'Intensity': attack_results['attack_config']['intensity'],
            'Detection F1': attack_results['detection_metrics']['f1_score'],
            'Detection Precision': attack_results['detection_metrics']['precision'],
            'Detection Recall': attack_results['detection_metrics']['recall'],
            'Attribution Accuracy': attack_results['attribution_accuracy'],
            'Attribution F1 Macro': attack_results['attribution_metrics']['f1_score_macro'],
            'Avg Confidence': attack_results['avg_confidence'],
            'Avg Time (s)': attack_results['avg_time'],
            'True Positives': attack_results['true_positives'],
            'False Positives': attack_results['false_positives'],
            'True Negatives': attack_results['true_negatives'],
            'False Negatives': attack_results['false_negatives']
        })
    
    return pd.DataFrame(attack_data)

def plot_detection_performance(df, output_dir):
    """Plot detection performance across attacks"""
    plt.figure(figsize=(15, 10))
    
    # Group by attack type
    attack_types = df['Type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(attack_types)))
    
    for i, attack_type in enumerate(attack_types):
        type_data = df[df['Type'] == attack_type]
        plt.scatter(type_data['Detection F1'], type_data['Attribution Accuracy'], 
                   color=colors[i], label=attack_type, s=60, alpha=0.7)
    
    plt.xlabel('Detection F1 Score')
    plt.ylabel('Attribution Accuracy') 
    plt.title('Detection vs Attribution Performance by Attack Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detection_attribution_scatter.png", dpi=300)
    plt.close()

def plot_performance_by_intensity(df, output_dir):
    """Plot performance degradation by attack intensity"""
    intensity_order = ['none', 'mild', 'moderate', 'strong', 'extreme']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Detection F1 by intensity
    intensity_data = df.groupby('Intensity').agg({
        'Detection F1': 'mean',
        'Attribution Accuracy': 'mean',
        'Avg Confidence': 'mean',
        'Avg Time (s)': 'mean'
    }).reindex(intensity_order)
    
    # Detection F1
    axes[0, 0].bar(intensity_data.index, intensity_data['Detection F1'])
    axes[0, 0].set_title('Detection F1 Score by Attack Intensity')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Attribution Accuracy
    axes[0, 1].bar(intensity_data.index, intensity_data['Attribution Accuracy'])
    axes[0, 1].set_title('Attribution Accuracy by Attack Intensity')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Confidence
    axes[1, 0].bar(intensity_data.index, intensity_data['Avg Confidence'])
    axes[1, 0].set_title('Average Confidence by Attack Intensity')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Processing Time
    axes[1, 1].bar(intensity_data.index, intensity_data['Avg Time (s)'])
    axes[1, 1].set_title('Average Processing Time by Attack Intensity')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_by_intensity.png", dpi=300)
    plt.close()

def plot_attack_type_heatmap(df, output_dir):
    """Create heatmap of performance by attack type and intensity"""
    # Create pivot table
    pivot_data = df.pivot_table(values='Detection F1', 
                               index='Type', 
                               columns='Intensity',
                               aggfunc='mean')
    
    # Reorder columns by intensity
    intensity_order = ['none', 'mild', 'moderate', 'strong', 'extreme']
    available_intensities = [i for i in intensity_order if i in pivot_data.columns]
    pivot_data = pivot_data.reindex(columns=available_intensities)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', fmt='.3f',
                cbar_kws={'label': 'Detection F1 Score'})
    plt.title('Detection Performance Heatmap\n(Attack Type vs Intensity)')
    plt.ylabel('Attack Type')
    plt.xlabel('Attack Intensity')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attack_heatmap.png", dpi=300)
    plt.close()

def generate_text_report(df, results, output_dir):
    """Generate a comprehensive text report"""
    with open(f"{output_dir}/analysis_report.txt", 'w') as f:
        f.write("ROBIN Watermarking Test Results Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Benchmark info
        if 'benchmark_info' in results:
            info = results['benchmark_info']
            f.write("Test Configuration:\n")
            f.write(f"- Total Images: {info.get('total_images', 'N/A')}\n")
            f.write(f"- Watermarked Images: {info.get('watermarked_images', 'N/A')}\n")
            f.write(f"- Clean Images: {info.get('clean_images', 'N/A')}\n")
            f.write(f"- Attacks Tested: {info.get('attacks_tested', 'N/A')}\n")
            f.write(f"- Test Duration: {info.get('duration_seconds', 0):.1f} seconds\n")
            f.write(f"- Timestamp: {info.get('timestamp', 'N/A')}\n\n")
        
        # Overall performance summary
        f.write("Overall Performance Summary:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Detection F1: {df['Detection F1'].mean():.3f}\n")
        f.write(f"Average Attribution Accuracy: {df['Attribution Accuracy'].mean():.3f}\n")
        f.write(f"Average Confidence: {df['Avg Confidence'].mean():.3f}\n")
        f.write(f"Average Processing Time: {df['Avg Time (s)'].mean():.3f}s\n\n")
        
        # Best and worst performing attacks
        f.write("Best Performing Attacks (Detection F1):\n")
        f.write("-" * 35 + "\n")
        best_attacks = df.nlargest(5, 'Detection F1')[['Attack', 'Detection F1', 'Attribution Accuracy']]
        for _, row in best_attacks.iterrows():
            f.write(f"- {row['Attack']}: F1={row['Detection F1']:.3f}, Attr={row['Attribution Accuracy']:.3f}\n")
        
        f.write("\nWorst Performing Attacks (Detection F1):\n")
        f.write("-" * 36 + "\n")
        worst_attacks = df.nsmallest(5, 'Detection F1')[['Attack', 'Detection F1', 'Attribution Accuracy']]
        for _, row in worst_attacks.iterrows():
            f.write(f"- {row['Attack']}: F1={row['Detection F1']:.3f}, Attr={row['Attribution Accuracy']:.3f}\n")
        
        # Performance by attack type
        f.write("\nPerformance by Attack Type:\n")
        f.write("-" * 28 + "\n")
        type_summary = df.groupby('Type').agg({
            'Detection F1': 'mean',
            'Attribution Accuracy': 'mean',
            'Avg Confidence': 'mean'
        }).round(3)
        
        for attack_type, row in type_summary.iterrows():
            f.write(f"- {attack_type}: F1={row['Detection F1']:.3f}, "
                   f"Attr={row['Attribution Accuracy']:.3f}, "
                   f"Conf={row['Avg Confidence']:.3f}\n")
        
        # Performance by intensity
        f.write("\nPerformance by Attack Intensity:\n")
        f.write("-" * 32 + "\n")
        intensity_summary = df.groupby('Intensity').agg({
            'Detection F1': 'mean',
            'Attribution Accuracy': 'mean',
            'Avg Confidence': 'mean'
        }).round(3)
        
        intensity_order = ['none', 'mild', 'moderate', 'strong', 'extreme']
        for intensity in intensity_order:
            if intensity in intensity_summary.index:
                row = intensity_summary.loc[intensity]
                f.write(f"- {intensity}: F1={row['Detection F1']:.3f}, "
                       f"Attr={row['Attribution Accuracy']:.3f}, "
                       f"Conf={row['Avg Confidence']:.3f}\n")

def export_to_csv(df, output_dir):
    """Export results to CSV for further analysis"""
    df.to_csv(f"{output_dir}/comprehensive_results.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description='Analyze ROBIN test results')
    parser.add_argument('--results_file', default='test_results/comprehensive_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', default='test_results/analysis',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and analyze results
    print("Loading results...")
    results = load_results(args.results_file)
    
    print("Creating summary table...")
    df = create_summary_table(results)
    
    print("Generating visualizations...")
    plot_detection_performance(df, args.output_dir)
    plot_performance_by_intensity(df, args.output_dir)
    plot_attack_type_heatmap(df, args.output_dir)
    
    print("Generating text report...")
    generate_text_report(df, results, args.output_dir)
    
    print("Exporting CSV...")
    export_to_csv(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("Generated files:")
    print("- analysis_report.txt: Comprehensive text summary")
    print("- comprehensive_results.csv: Data table for further analysis")
    print("- detection_attribution_scatter.png: Performance scatter plot")
    print("- performance_by_intensity.png: Performance vs intensity")
    print("- attack_heatmap.png: Attack type vs intensity heatmap")

if __name__ == '__main__':
    main()
