#!/bin/bash
# Optimize Watermark for ROBIN Testing Pipeline
# =============================================

set -e

# Configuration
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
CLEAN_IMAGES_DIR="clean_images"
NUM_CLEAN_IMAGES=200
WM_OUTPUT_DIR="ckpts"

echo "ROBIN Watermark Optimization Pipeline"
echo "====================================="
echo "This script will:"
echo "1. Generate clean images for watermark optimization"
echo "2. Optimize the watermark using adversarial training"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "robin" ]]; then
    echo "WARNING: Please activate the robin conda environment first:"
    echo "conda activate robin"
    echo ""
    echo "Or run the setup script first:"
    echo "bash setup_environment.sh"
    exit 1
fi

# Step 1: Generate clean images
echo "Step 1: Generating clean images for optimization..."
mkdir -p "$CLEAN_IMAGES_DIR"

python gen_clean_image.py \
    --start 0 \
    --end $NUM_CLEAN_IMAGES \
    --model_id "$MODEL_ID" \
    --save_path "$CLEAN_IMAGES_DIR" \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --image_length 512

echo "Clean images generated in: $CLEAN_IMAGES_DIR"

# Step 2: Optimize watermark
echo ""
echo "Step 2: Optimizing watermark..."
mkdir -p "$WM_OUTPUT_DIR"

python gen_watermark.py \
    --run_name watermark_optimization \
    --w_channel 3 \
    --w_pattern ring \
    --model_id "$MODEL_ID" \
    --data_root "$CLEAN_IMAGES_DIR"

echo ""
echo "Watermark optimization completed!"
echo "Optimized watermark saved in: $WM_OUTPUT_DIR"
echo ""
echo "You can now run the comprehensive testing pipeline with:"
echo "bash run_testing.sh"
echo ""
echo "Or manually specify the watermark path in run_testing.sh by setting:"
echo "WM_PATH=\"ckpts/your_watermark_file.pth\""
