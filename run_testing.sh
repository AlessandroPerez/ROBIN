#!/bin/bash
# Run ROBIN Watermarking Testing Pipeline
# ======================================

set -e

# Configuration
MODEL_ID="stabilityai/stable-diffusion-2-1-base"
PROMPTS_FILE="prompts_1000.json"
OUTPUT_DIR="test_results"
NUM_TEST_IMAGES=50
REFERENCE_MODEL="ViT-H-14"
REFERENCE_MODEL_PRETRAIN="laion2b_s32b_b79k"

# Optional: Path to pre-trained watermark (leave empty to generate random)
WM_PATH=""  # Set to path of .pth file if available, e.g., "ckpts/watermark.pth"

echo "Starting ROBIN Watermarking Testing Pipeline..."
echo "================================================"
echo "Model ID: $MODEL_ID"
echo "Prompts file: $PROMPTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Number of test images: $NUM_TEST_IMAGES"
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

# Check if prompts file exists
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "Error: Prompts file '$PROMPTS_FILE' not found!"
    echo "Please ensure the prompts_1000.json file is in the current directory."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the command arguments
CMD_ARGS="--model_id $MODEL_ID"
CMD_ARGS="$CMD_ARGS --prompts_file $PROMPTS_FILE"
CMD_ARGS="$CMD_ARGS --output_dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --num_test_images $NUM_TEST_IMAGES"
CMD_ARGS="$CMD_ARGS --reference_model $REFERENCE_MODEL"
CMD_ARGS="$CMD_ARGS --reference_model_pretrain $REFERENCE_MODEL_PRETRAIN"

# Add watermark path if specified
if [[ -n "$WM_PATH" && -f "$WM_PATH" ]]; then
    CMD_ARGS="$CMD_ARGS --wm_path $WM_PATH"
    echo "Using pre-trained watermark: $WM_PATH"
else
    echo "Using randomly generated watermark"
fi

echo "Running comprehensive testing pipeline..."
echo "Command: python comprehensive_testing_pipeline.py $CMD_ARGS"
echo ""

# Run the pipeline
python comprehensive_testing_pipeline.py $CMD_ARGS

echo ""
echo "Testing pipeline completed!"
echo "Results available in: $OUTPUT_DIR/"
echo "- comprehensive_results.json: Detailed metrics and attack results"
echo "- images/: Generated and attacked images organized by type"
echo "  - clean/: Clean (non-watermarked) images"
echo "  - watermarked/: Watermarked images"
echo "  - attacked/: Images after various attacks, organized by attack type"
