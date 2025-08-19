#!/bin/bash
# Activate ROBIN conda environment
# ===============================

# This script helps activate the robin conda environment properly

echo "Activating ROBIN conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if robin environment exists
if conda env list | grep -q "^robin "; then
    echo "✅ Found 'robin' environment"
    echo "To activate the environment, run:"
    echo ""
    echo "conda activate robin"
    echo ""
    echo "Then you can run:"
    echo "python validate_setup.py     # Validate installation"
    echo "bash run_testing.sh         # Run comprehensive testing"
    echo "python analyze_results.py   # Analyze results"
else
    echo "❌ 'robin' environment not found"
    echo ""
    echo "To create the environment, run:"
    echo "bash setup_environment.sh"
    echo ""
    echo "Or create manually with:"
    echo "conda env create -f environment.yml"
    exit 1
fi
