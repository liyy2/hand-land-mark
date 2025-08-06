# 🏥 PD Movement Analysis Gradio Application

A comprehensive web application for analyzing hand movements to detect Parkinson's Disease biomarkers.

## Features

### 📹 Video Processing
- Upload and process videos with hand landmark detection
- Real-time progress tracking
- Adjustable detection confidence
- Optional face detection
- Contrast enhancement for poor lighting conditions
- Export landmark data as CSV

### 🔬 PD Movement Analysis
- **Tremor Analysis**: FFT-based frequency spectrum analysis (4-6 Hz for PD resting tremor)
- **Finger Tapping**: Detect bradykinesia through tapping patterns
- **Amplitude Decrement**: Track progressive reduction in movement
- **Movement Asymmetry**: Compare left vs right hand movements
- **Bradykinesia Score**: Comprehensive severity assessment

### 📊 Visualizations
- Tremor frequency spectrum plots
- Finger tapping temporal patterns
- Amplitude progression charts
- Comprehensive clinical dashboard
- Time series analysis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install gradio scipy
```

## Usage

### Option 1: Using the launcher script
```bash
./run_app.sh
```

### Option 2: Direct Python execution
```bash
# Activate conda environment
source /home/yl2428/miniconda/etc/profile.d/conda.sh
conda activate landmark

# Run the application
python gradio_app.py
```

### Option 3: Custom port
```bash
GRADIO_SERVER_PORT=8080 python gradio_app.py
```

## Access the Application

Once started, the application will display:
```
Running on local URL: http://0.0.0.0:7860
```

Open your browser and navigate to:
- Local: `http://localhost:7860`
- Network: `http://<your-ip>:7860`

## Recording Guidelines

### Finger Tapping Task
1. Sit comfortably with hands clearly visible
2. Tap index finger and thumb together repeatedly
3. Make movements as large and fast as possible
4. Continue for 10-15 seconds per hand
5. Test both hands separately or together

### Rest Tremor Assessment
1. Rest hands on lap or armrests
2. Relax completely for 10-20 seconds
3. Avoid voluntary movements

### Action Tremor Assessment
1. Extend arms forward
2. Hold position for 10-15 seconds
3. Or perform finger-to-nose task

## Understanding Results

### Tremor Frequencies
- **4-6 Hz**: Typical PD resting tremor
- **6-12 Hz**: PD action tremor or essential tremor
- **>12 Hz**: Likely physiological or enhanced tremor

### Bradykinesia Indicators
- **Slow tapping**: < 3 Hz frequency
- **Amplitude decrement**: > 30% reduction
- **Irregular rhythm**: CV > 0.3
- **Progressive slowing**: > 20% speed decrease
- **Hesitations**: Pauses during movement

### Severity Levels
- ✅ **None**: No bradykinesia detected
- ⚠️ **Mild**: 1-2 indicators present
- 🟠 **Moderate**: 3 indicators present
- 🔴 **Severe**: 4+ indicators present

## Output Files

The application generates several output files in a temporary session directory:

- `tracked_video.mp4`: Video with hand landmarks overlaid
- `landmarks.csv`: Raw landmark coordinates and metadata
- `pd_analysis_dashboard.png`: Comprehensive visualization
- `pd_analysis_report.json`: Detailed analysis results

## Technical Details

### Architecture
- **Frontend**: Gradio web interface
- **Detection**: MediaPipe Holistic model for robust hand tracking
- **Analysis**: Custom PD-specific algorithms
- **Visualization**: Matplotlib-based clinical dashboards

### Performance
- Processing speed: ~5-10 FPS on CPU
- Recommended video: 10-60 seconds at 30+ FPS
- Memory usage: ~500MB-1GB depending on video size

## Troubleshooting

### Port Already in Use
The app automatically finds an available port between 7860-7880. If all ports are busy:
```bash
GRADIO_SERVER_PORT=8888 python gradio_app.py
```

### Detection Issues
- Ensure good lighting
- Keep hands in frame
- Try lowering detection confidence (0.1-0.3)
- Enable contrast enhancement for dark videos

### Analysis Issues
- Ensure at least 10 seconds of recording
- Check that hands are detected in most frames
- Verify tapping movements are clear and distinct

## Disclaimer

⚠️ **Important**: This system is for research and educational purposes only. It should not be used for clinical diagnosis. Always consult healthcare professionals for medical advice.

## Citation

If you use this tool in research, please cite:
```
PD Movement Analysis System
Hand Landmark Detection Pipeline with Parkinson's Disease Analysis
2024
```

## License

This project is for research purposes. See LICENSE file for details.