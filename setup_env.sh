#!/bin/bash

echo "Creating conda environment for landmark detection..."

# Create conda environment from yml file
conda env create -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate landmark-detection"
echo ""
echo "To verify the installation, run:"
echo "  conda activate landmark-detection"
echo "  python -c 'import cv2, mediapipe, numpy, pandas, matplotlib; print(\"All packages imported successfully!\")'"