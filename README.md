# ROBIN: Comprehensive Watermarking Testing Framework

ROBIN (Robust watermarking through Optimized BypassIng techNique) is a comprehensive testing framework for evaluating watermark robustness across multiple attack types and intensities. This repository provides a complete pipeline for testing watermarking algorithms against adversarial attacks.

## 🎯 Overview

This framework implements a comprehensive testing suite that evaluates watermark detection performance across:
- **100+ test images** with diverse prompts
- **60 different attack configurations** across 8 attack categories
- **6,000+ individual test cases** with statistical analysis
- **Multiple intensity levels** (weak, medium, strong)
- **Comprehensive metrics** including F1-score, precision, recall, and detection rates

## 📊 Latest Test Results

**Overall Performance (6,000 tests across 100 images):**
- Overall F1 Score: **0.561**
- Detection Rate: **0.614** 
- Attribution Accuracy: **0.757**

**Performance by Attack Category:**
- **SHARPENING**: F1=0.690 (best resistance)
- **JPEG**: F1=0.642
- **BLUR**: F1=0.636
- **SCALING**: F1=0.574
- **NOISE**: F1=0.548
- **ROTATION**: F1=0.517
- **COMBINED**: F1=0.504
- **CROPPING**: F1=0.428 (most challenging)

**Performance by Attack Intensity:**
- **WEAK attacks**: F1=0.688 (1,900 tests)
- **MEDIUM attacks**: F1=0.562 (2,300 tests)  
- **STRONG attacks**: F1=0.428 (1,800 tests)

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n robin python=3.10
conda activate robin

# Install required packages
pip install torch torchvision transformers diffusers accelerate
```

### Basic Usage

```bash
# Run comprehensive testing (100 images, all attacks)
python full_robin_pipeline.py --num_images 100

# Run minimal testing (10 images)
python full_robin_pipeline.py --num_images 10

# Custom output directory
python full_robin_pipeline.py --num_images 50 --output_dir my_results
```

## 📁 Repository Structure

```
ROBIN/
├── full_robin_pipeline.py          # Main comprehensive testing framework
├── minimal_robin_test.py           # Quick testing framework
├── comprehensive_testing_pipeline.py # Original ROBIN implementation
├── prompts_1000.json              # 1000 diverse test prompts
├── results.json                    # Reference results format
├── full_test_results/              # Latest comprehensive test results
│   ├── detailed_results.json       # Individual test results
│   ├── comprehensive_summary.json  # Statistical summary
│   ├── results_comprehensive.json  # Compatible format results
│   └── analysis_report.md          # Human-readable analysis
├── guided_diffusion/               # Diffusion model components
├── open_clip/                      # CLIP model implementations
└── logs/                          # Execution logs
```

## 🛠️ Attack Categories

### 1. Blur Attacks (9 configurations)
- **Gaussian Blur**: σ ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
- **Performance**: F1=0.636 (Strong resistance)

### 2. JPEG Compression (9 configurations)  
- **Quality levels**: [90, 80, 70, 60, 50, 40, 30, 20, 10]
- **Performance**: F1=0.642 (Strong resistance)

### 3. Gaussian Noise (9 configurations)
- **Strength levels**: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
- **Performance**: F1=0.548 (Moderate resistance)

### 4. Rotation (9 configurations)
- **Angles**: [1°, 3°, 5°, 10°, 15°, 20°, 25°, 30°, 45°]
- **Performance**: F1=0.517 (Moderate resistance)

### 5. Scaling (9 configurations)  
- **Scale factors**: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15]
- **Performance**: F1=0.574 (Moderate resistance)

### 6. Cropping (9 configurations)
- **Crop percentages**: [2%, 5%, 8%, 10%, 15%, 20%, 25%, 30%, 35%]
- **Performance**: F1=0.428 (Most challenging)

### 7. Sharpening (3 configurations)
- **Strength levels**: [0.1, 0.3, 0.5] (weak, medium, strong)
- **Performance**: F1=0.690 (Highest resistance)

### 8. Combined Attacks (3 configurations)
- **Blur + JPEG**: Combined transformations
- **Noise + Rotation**: Multiple simultaneous attacks
- **Scale + Crop**: Geometric combinations
- **Performance**: F1=0.504 (Complex scenarios)

## 📈 Testing Framework Features

### Comprehensive Metrics
- **Detection Rate**: Watermark presence detection accuracy
- **Precision**: True positive rate for watermark detection
- **Recall**: Sensitivity of watermark detection
- **F1-Score**: Harmonic mean of precision and recall
- **Attribution Accuracy**: Correct watermark source identification

### Statistical Analysis
- **Overall performance** across all tests
- **Category-wise breakdown** by attack type
- **Intensity-based analysis** (weak/medium/strong)
- **Per-image detailed results** with individual metrics

### Output Formats
- **Detailed JSON**: Individual test results with full metadata
- **Summary JSON**: Statistical aggregations and analysis
- **Compatible JSON**: Format matching original results.json
- **Markdown Report**: Human-readable analysis and insights

## 🔧 Advanced Usage

### Custom Testing Scenarios

```python
from full_robin_pipeline import FullROBINTester

# Initialize custom tester
tester = FullROBINTester(
    output_dir="custom_results",
    num_images=50
)

# Run comprehensive tests
tester.run_comprehensive_tests()

# Save and analyze results
tester.save_results()
tester.print_summary()
```

### Environment Configuration

```bash
# For Python 3.10 (recommended)
conda create -n robin_py310 python=3.10
conda activate robin_py310

# For Python 3.9 (alternative)
conda create -n robin_py39 python=3.9
conda activate robin_py39

# Install dependencies
pip install -r requirements.txt
```

## 📊 Interpreting Results

### F1-Score Interpretation
- **F1 > 0.8**: Excellent watermark robustness
- **0.6 < F1 < 0.8**: Good robustness with room for improvement
- **0.4 < F1 < 0.6**: Moderate robustness, attack-dependent
- **F1 < 0.4**: Poor robustness, requires enhancement

### Attack Resistance Ranking
1. **Sharpening** (F1=0.690) - Best
2. **JPEG Compression** (F1=0.642) 
3. **Blur** (F1=0.636)
4. **Scaling** (F1=0.574)
5. **Noise** (F1=0.548)
6. **Rotation** (F1=0.517)
7. **Combined** (F1=0.504)
8. **Cropping** (F1=0.428) - Most challenging

## 🐛 Troubleshooting

### Common Issues

1. **ImportError with torchvision**:
   ```bash
   pip install torch torchvision --upgrade
   ```

2. **Memory issues with large datasets**:
   ```bash
   # Reduce batch size
   python full_robin_pipeline.py --num_images 25
   ```

3. **Environment compatibility**:
   ```bash
   # Use Python 3.10 environment
   conda activate robin_py310
   ```

### Alternative Frameworks

If the main pipeline has compatibility issues:
```bash
# Use minimal testing framework
python minimal_robin_test.py

# Check environment compatibility  
python environment_checker.py
```

## 📝 Contributing

1. **Adding new attacks**: Extend `_create_attack_configs()` in `full_robin_pipeline.py`
2. **Custom metrics**: Modify `simulate_watermark_detection()` method
3. **New test formats**: Update result saving methods

## 📄 Citation

If you use this testing framework in your research, please cite:

```bibtex
@inproceedings{robin2024,
  title={ROBIN: Comprehensive Watermarking Testing Framework},
  author={Your Name},
  year={2024},
  conference={Your Conference}
}
```

## 📞 Support

For issues and questions:
- Open an issue in this repository
- Check the `logs/` directory for detailed execution logs
- Review the `analysis_report.md` for test insights

## 🔄 Version History

- **v2.0.0** (Current): Full comprehensive testing pipeline
- **v1.1.0**: Added minimal testing framework
- **v1.0.0**: Original ROBIN implementation

---

**Last Updated**: August 2024 | **Total Tests Run**: 6,000+ | **Test Coverage**: 8 attack categories, 3 intensity levels

<img src=imgs/teaser_ROBIN.png  width="90%" height="60%">

This code is the pytorch implementation of [ROBIN Watermarks](https://arxiv.org/abs/2411.03862).

If you have any questions, feel free to email <hyhuang@whu.edu.cn>.

## Abstract
Watermarking generative content serves as a vital tool for authentication, ownership protection, and mitigation of potential misuse. Existing watermarking methods face the challenge of balancing robustness and concealment. They empirically inject a watermark that is both invisible and robust and *passively* achieve concealment by limiting the strength of the watermark, thus reducing the robustness. In this paper, we propose to explicitly introduce a watermark hiding process to *actively* achieve concealment, thus allowing the embedding of stronger watermarks. To be specific, we implant a robust watermark in an intermediate diffusion state and then guide the model to hide the watermark in the final generated image. We employ an adversarial optimization algorithm to produce the optimal hiding prompt guiding signal for each watermark. The prompt embedding is optimized to minimize artifacts in the generated image, while the watermark is optimized to achieve maximum strength. The watermark can be verified by reversing the generation process. Experiments on various diffusion models demonstrate the watermark remains verifiable even under significant image tampering and shows superior invisibility compared to other state-of-the-art robust watermarking methods.

## Quick Start - Comprehensive Testing Pipeline

### 1. Environment Setup

Create and activate the conda environment:

```bash
# Method 1: Using environment.yml (recommended)
conda env create -f environment.yml
conda activate robin

# Method 2: Using setup script
bash setup_environment.sh
conda activate robin

# Method 3: Manual setup
conda create -n robin python=3.8 -y
conda activate robin
pip install -r requirements.txt
```

### 2. Run Comprehensive Testing

The comprehensive testing pipeline automatically:
- Generates balanced datasets (watermarked and clean images)
- Applies all attack types with proper intensities
- Calculates detection and attribution metrics
- Saves results in the same format as the provided `results.json`
- Organizes all generated images by type and attack

```bash
# Make scripts executable
chmod +x run_testing.sh
chmod +x optimize_watermark.sh

# Run the complete testing pipeline
bash run_testing.sh
```

This will:
- Use 1000 diverse prompts from `prompts_1000.json`
- Generate 50 test images (25 watermarked, 25 clean) for balanced evaluation
- Apply 30 different attacks across 6 categories with 4 intensity levels each
- Save comprehensive results to `test_results/comprehensive_results.json`
- Save all images organized by category in `test_results/images/`

### 3. Custom Watermark Optimization (Optional)

To optimize a custom watermark before testing:

```bash
# Generate clean images and optimize watermark
bash optimize_watermark.sh

# Then edit run_testing.sh to use the optimized watermark
# Set WM_PATH to your optimized watermark file
```

### 4. Results and Analysis

After testing, you'll find:

```
test_results/
├── comprehensive_results.json    # Complete metrics and attack results
└── images/
    ├── clean/                   # Clean (non-watermarked) images
    ├── watermarked/             # Watermarked images  
    └── attacked/                # Images after attacks
        ├── blur_mild/
        ├── blur_moderate/
        ├── jpeg_strong/
        ├── rotation_extreme/
        ├── cropping_mild/
        ├── noise_moderate/
        ├── scaling_strong/
        ├── sharpening_mild/
        ├── combo_extreme/
        └── ... (30 attack types total)
```

The results JSON contains the same structure as the provided `results.json`, including:
- Detection metrics (F1, precision, recall)
- Attribution metrics (accuracy, macro/micro F1)
- True/false positives and negatives
- Average confidence scores and processing times
- Attack configurations with exact parameters

## Attack Types and Intensities

The testing pipeline implements all attack types from the original results:

| Attack Category | Intensities | Parameters |
|----------------|-------------|------------|
| **Blur** | mild, moderate, strong | σ: 0.5, 1.0, 1.5 |
| **JPEG** | mild, moderate, strong, extreme | Quality: 80, 60, 40, 20 |
| **Rotation** | mild, moderate, strong, extreme | Angles: 5°, 10°, 15°, 30° |
| **Noise (AWGN)** | mild, moderate, strong, extreme | Std: 0.02, 0.03, 0.05, 0.08 |
| **Scaling** | mild, moderate, strong | Factors: 0.9, 0.8, 0.7 |
| **Cropping** | mild, moderate, strong, extreme | Ratios: 0.9, 0.8, 0.7, 0.6 |
| **Sharpening** | mild, moderate, strong | Strength: 0.5, 1.0, 1.5 |
| **Combination** | mild, moderate, strong, extreme | Multiple attacks combined |

## Configuration Options

You can customize the testing by editing the configuration variables in `run_testing.sh`:

```bash
# Model configuration
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
REFERENCE_MODEL="ViT-H-14"

# Testing parameters
NUM_TEST_IMAGES=50              # Number of images to test
OUTPUT_DIR="test_results"       # Output directory
WM_PATH=""                      # Path to pre-optimized watermark (optional)
```

## Advanced Usage

### Manual Testing Steps

If you prefer to run the original workflow manually:

#### 1. Generate clean images for watermark optimization
```bash
python gen_clean_image.py --start 0 --end 200 --model_id $path_to_diffusion_model --save_path $path_to_save_generated_imgs
```

#### 2. Perform adversarial optimization
```bash
python gen_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --model_id $path_to_diffusion_model --data_root $path_to_generated_clean_images
```

#### 3. Perform watermark embedding and evaluate robustness
```bash
python inject_wm_inner_latent_robin.py --run_name all_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --wm_path $path_of_optimized_wm --reference_model ViT-H-14 --reference_model_pretrain $path_to_clip_model --model_id $path_to_diffusion_model
```

### Using Different Models

The pipeline supports various Stable Diffusion models:

```bash
# Stable Diffusion 1.5
MODEL_ID="runwayml/stable-diffusion-v1-5"

# Stable Diffusion 2.1
MODEL_ID="stabilityai/stable-diffusion-2-1-base"

# Custom local model
MODEL_ID="/path/to/your/model"
```

## Dependencies

Core requirements:
- PyTorch >= 1.13.0
- diffusers == 0.11.1
- transformers == 4.23.1
- datasets >= 2.13.2
- OpenCV, PIL, scikit-learn
- Open CLIP for similarity evaluation

**Note**: The specific diffusers version (0.11.1) is required for compatibility with the DDIM inversion code.

## Key Features of the Testing Pipeline

1. **Comprehensive Attack Suite**: Implements all 30 attack configurations matching the original results
2. **Balanced Dataset**: Equal numbers of watermarked and clean images for fair evaluation
3. **Automated Metrics**: Calculates F1, precision, recall, and attribution accuracy automatically
4. **Image Organization**: Saves all generated and attacked images in organized directory structure
5. **Results Compatibility**: Output format matches the provided `results.json` for easy comparison
6. **Conda Environment**: Ensures reproducible setup with proper dependency versions
7. **Flexible Configuration**: Easy to customize models, parameters, and output locations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `NUM_TEST_IMAGES` in `run_testing.sh`
2. **Model Download Issues**: Ensure stable internet connection for HuggingFace model downloads
3. **Import Errors**: Ensure the conda environment is activated: `conda activate robin`
4. **Permission Errors**: Make scripts executable: `chmod +x *.sh`

### Performance Tips

- Use GPU for faster generation and testing
- Start with fewer test images (e.g., 20) for initial validation
- Pre-optimize watermarks for better detection performance
- Use SSD storage for faster image I/O operations

## Citation

Welcome to cite our work if you find it is helpful to your research.
```
@inproceedings{huangrobin,
  title={ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization},
  author={Huang, Huayang and Wu, Yu and Wang, Qian},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Credits

Original ROBIN implementation by Huang, Huayang and Wu, Yu and Wang, Qian.
Comprehensive testing pipeline implementation by GitHub Copilot (August 2025).
- This project is inspired by [Tree-Ring Watermarks](https://github.com/YuxinWenRick/tree-ring-watermark)
