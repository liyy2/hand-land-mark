#!/bin/bash

# Launcher script for PD Movement Analysis Gradio App

echo "🏥 Parkinson's Disease Movement Analysis System"
echo "=============================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /home/yl2428/miniconda/etc/profile.d/conda.sh
conda activate landmark

# Check if Gradio is installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "Installing Gradio..."
    pip install gradio --quiet
fi

# Check if scipy is installed
if ! python -c "import scipy" 2>/dev/null; then
    echo "Installing scipy..."
    pip install scipy --quiet
fi

# Kill any existing Gradio processes on ports 7860-7880
echo "Checking for existing processes..."
for port in {7860..7880}; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

# Start the application
echo ""
echo "Starting application..."
echo "=============================================="
python /home/yl2428/hand-land-mark/gradio_app.py