#!/bin/bash
# Conda Environment Setup for ROBIN Watermarking Testing Pipeline
# =============================================================

set -e  # Exit on any error

echo "Creating conda environment for ROBIN watermarking testing..."

# Method 1: Create from environment.yml (recommended)
if [[ -f "environment.yml" ]]; then
    echo "Using environment.yml for setup..."
    conda env create -f environment.yml
else
    # Method 2: Manual setup (fallback)
    echo "Using manual setup..."
    
    # Create conda environment
    conda create -n robin python=3.8 -y

    # Activate environment
    source activate robin

    # Install PyTorch with CUDA support
    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

    # Install core dependencies from package.txt requirements
    pip install accelerate==0.20.3
    pip install diffusers==0.11.1
    pip install transformers==4.23.1
    pip install datasets==2.13.2

    # Install additional required packages
    pip install opencv-python==4.9.0.80
    pip install pytorch-msssim==1.0.0
    pip install scikit-learn==1.0.2
    pip install Pillow==9.5.0
    pip install numpy==1.21.6
    pip install pandas==1.3.5

    # Install CLIP related packages  
    pip install ftfy==6.1.1
    pip install regex==2023.8.8

    # Install other utilities
    pip install tqdm
    pip install requests==2.31.0
    pip install safetensors==0.4.0

    # Install jupyter for analysis (optional)
    pip install jupyter
    pip install matplotlib
    pip install seaborn
fi

echo "Conda environment 'robin' created successfully!"
echo ""
echo "To activate the environment, run:"
echo "conda activate robin"
echo ""
echo "To test the installation, you can run:"
echo "python -c \"import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\""
