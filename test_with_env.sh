#!/bin/bash
# Test script that ensures we're in the robin environment

echo "=== ROBIN ENVIRONMENT TEST ==="
echo "Activating robin environment..."

# Activate the environment
source ~/miniconda3/bin/activate robin

# Verify environment
echo "Current environment: $(conda info --envs | grep '*')"
echo "Python path: $(which python)"
echo "Diffusers installed: $(python -c 'import diffusers; print(diffusers.__version__)' 2>/dev/null || echo 'NOT FOUND')"

echo ""
echo "=== RUNNING DIRECT TEST ==="
python direct_test.py
