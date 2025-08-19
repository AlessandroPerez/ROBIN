# ROBIN Watermarking - Quick Start Guide

## 🚀 Quick Setup and Testing (5 minutes)

### 1. Setup Environment
```bash
# Clone/navigate to ROBIN directory
cd /path/to/ROBIN

# Create conda environment (choose one method)
conda env create -f environment.yml          # Recommended
# OR
bash setup_environment.sh                    # Alternative

# Activate environment  
conda activate robin
```

### 2. Validate Setup
```bash
# Check if everything is properly installed
python validate_setup.py
```

### 3. Run Comprehensive Testing
```bash
# Run the complete testing pipeline
bash run_testing.sh

# This will:
# ✅ Generate 50 test images (25 clean + 25 watermarked)
# ✅ Apply 30 different attacks with proper intensities
# ✅ Calculate detection and attribution metrics
# ✅ Save results in results.json format
# ✅ Organize all images by attack type
```

### 4. View Results
```bash
# Check results
cat test_results/comprehensive_results.json

# Generate analysis report
python analyze_results.py

# View analysis
cat test_results/analysis/analysis_report.txt
```

## 📊 Expected Results Structure

Your results will match the format in the provided `results.json`:

```json
{
  "clean": {
    "attack_config": {"type": "none", "intensity": "none"},
    "detection_metrics": {"f1_score": 1.0, "precision": 1.0, "recall": 1.0},
    "attribution_accuracy": 1.0,
    "avg_confidence": 0.45,
    "avg_time": 0.02
  },
  "blur_mild": {
    "attack_config": {"type": "blur", "intensity": "mild", "params": {"sigma": 0.5}},
    // ... metrics
  },
  // ... 30 attack types total
}
```

## 🎯 Key Features

- **Balanced Dataset**: Equal numbers of watermarked and clean images
- **Complete Attack Suite**: All 30 attack types from original results  
- **Exact Parameters**: Matching intensities and parameters
- **Automated Metrics**: F1, precision, recall, attribution accuracy
- **Image Organization**: All generated/attacked images saved and organized
- **Conda Environment**: Reproducible setup with exact dependency versions

## ⚙️ Customization

Edit `run_testing.sh` to customize:
- `NUM_TEST_IMAGES=50` → Change number of test images
- `MODEL_ID` → Use different Stable Diffusion model
- `WM_PATH` → Use pre-optimized watermark

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `conda activate robin` |
| CUDA out of memory | Reduce `NUM_TEST_IMAGES` to 20-30 |
| Slow generation | Ensure GPU is available |
| Permission denied | `chmod +x *.sh` |

## 📈 Performance Tips

- **GPU Required**: Use CUDA-enabled GPU for reasonable performance
- **Start Small**: Begin with 20-30 images for initial testing
- **Pre-optimize**: Run `bash optimize_watermark.sh` for better results
- **SSD Storage**: Use fast storage for image I/O operations

## 🎉 What You Get

After completion:
- Comprehensive metrics matching the provided results.json
- 1500+ images organized by attack type  
- Analysis reports and visualizations
- CSV data for custom analysis
- Complete reproducible testing environment

**Estimated Runtime**: 30-60 minutes for 50 images (with GPU)
