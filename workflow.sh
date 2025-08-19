#!/bin/bash
# Complete ROBIN Testing Workflow
# ===============================

echo "🚀 ROBIN Watermarking Complete Testing Workflow"
echo "=============================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "📋 Step-by-Step Instructions:"
echo ""

echo "1️⃣  Setup Conda Environment:"
echo "   conda env create -f environment.yml"
echo "   conda activate robin"
echo ""

echo "2️⃣  Validate Setup:"
echo "   python validate_setup.py"
echo ""

echo "3️⃣  Run Comprehensive Testing:"
echo "   bash run_testing.sh"
echo ""

echo "4️⃣  Analyze Results:"
echo "   python analyze_results.py"
echo ""

echo "🎯 What You'll Get:"
echo "   ✅ Balanced dataset (25 clean + 25 watermarked images)"
echo "   ✅ 30 attack types with proper intensities"
echo "   ✅ Detection metrics (F1, precision, recall)"
echo "   ✅ Attribution accuracy scores"
echo "   ✅ Results in same format as provided results.json"
echo "   ✅ 1500+ organized images by attack type"
echo "   ✅ Analysis reports and visualizations"
echo ""

echo "⚙️  Optional - Optimize Custom Watermark:"
echo "   bash optimize_watermark.sh"
echo "   # Then edit run_testing.sh to set WM_PATH"
echo ""

echo "🔧 Quick Start Commands:"
echo "   chmod +x *.sh                    # Make scripts executable"
echo "   conda env create -f environment.yml  # Create environment"
echo "   conda activate robin             # Activate environment"
echo "   python validate_setup.py         # Check everything works"
echo "   bash run_testing.sh             # Run full pipeline"
echo ""

echo "📁 Output Structure:"
echo "   test_results/"
echo "   ├── comprehensive_results.json  # Main results file"
echo "   └── images/"
echo "       ├── clean/                  # Clean images"
echo "       ├── watermarked/            # Watermarked images"
echo "       └── attacked/               # Images after attacks"
echo "           ├── blur_mild/"
echo "           ├── jpeg_strong/"
echo "           ├── rotation_extreme/"
echo "           └── ... (30 attack types)"
echo ""

echo "⏱️  Estimated Runtime: 30-60 minutes (with GPU, 50 images)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^robin "; then
    echo "✅ 'robin' environment already exists!"
    echo "   Run: conda activate robin"
else
    echo "🔧 Ready to create 'robin' environment"
    echo "   Run: conda env create -f environment.yml"
fi

echo ""
echo "📚 For detailed instructions, see:"
echo "   - README.md (complete documentation)"
echo "   - QUICK_START.md (5-minute guide)"
echo "   - config.txt (configuration options)"
