# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

EDITS SHOULD BE SIMPLE BE FOOLISH. DON"T OVER ENGINEER. ALWAYS STARTS WITH SIMPLE EDITS.

## Project Overview

This is a unified landmark detection pipeline that processes videos to detect and track both hand and facial movements using MediaPipe. The system outputs annotated videos and exports time-series data of landmarks, with a focus on future Parkinson's disease diagnostic applications.

## Common Commands

### Setup and Installation
```bash
pip install -r requirements.txt
```

### Activate conda env
source /home/yl2428/miniconda/etc/profile.d/conda.sh && conda activate landmark

### TEST FILE LOCATED



### Running the Pipeline
```bash
# Basic usage (hands only)
python main.py input_video.mp4

# With face detection
python main.py input_video.mp4 --extract-face

# Full analysis with visualizations
python main.py input_video.mp4 --extract-face --visualize --output-dir results

# Custom parameters for both detectors
python main.py input_video.mp4 --extract-face --max-hands 2 --max-faces 1 --hand-detection-confidence 0.7 --face-detection-confidence 0.6
```

## Architecture

### Core Components

1. **UnifiedLandmarkDetector** (`unified_landmark_detector.py`): Combined detection class
   - Integrates MediaPipe's Hands (21 landmarks) and Face Mesh (468 landmarks)
   - Processes both detectors on same frame for efficiency
   - Outputs unified data structure with landmark_type field
   - Handles video I/O and frame annotation for both types

2. **HandLandmarkDetector** (`hand_landmark_detector.py`): Legacy hand-only detector
   - Kept for reference/backward compatibility
   - Original implementation for hand landmarks only

3. **Visualization Module** (`visualize_landmarks.py`): Data analysis and plotting
   - `plot_landmark_timeseries()`: Plots x,y,z coordinates over time for hands or face
   - `plot_movement_heatmap()`: Creates spatial heatmaps for movement patterns
   - `create_landmark_summary()`: Generates unified detection statistics
   - Supports both hand and face landmark visualization

4. **CLI Interface** (`main.py`): Command-line entry point
   - Separate arguments for hand and face detection parameters
   - Uses UnifiedLandmarkDetector for processing
   - Generates separate visualizations for each landmark type

### Data Flow

1. Video input → Frame extraction
2. Frame → Parallel processing:
   - MediaPipe hand detection (21 landmarks per hand)
   - MediaPipe face mesh detection (468 landmarks per face)
3. Unified landmarks → CSV export with landmark_type field
4. Landmarks → Annotated video output (both types drawn)
5. Optional: CSV data → Separate visualization plots for hands/face

### Key Technical Details

- **Landmark Systems**: 
  - Hands: 21 points (0=wrist, 1-4=thumb, 5-8=index, etc.)
  - Face: 468 3D points (full facial geometry with refine_landmarks)
- **Coordinate Systems**: 
  - Normalized (0-1) for model output
  - Pixel coordinates for visualization
  - Z-coordinate represents depth (face more detailed than hands)
- **Performance**: Runs on CPU by default
  - Hands only: ~30-60 FPS
  - Hands + Face: ~15-30 FPS (depends on hardware)
- **Error Handling**: Validates video properties, handles missing frames gracefully

### Future PD Diagnostic Considerations

The pipeline is designed with Parkinson's disease detection in mind:
- High-frequency tremor analysis requires good temporal resolution
- Face landmarks enable hypomimia (facial masking) detection
- Hand landmarks capture bradykinesia and tremor patterns
- Unified CSV format facilitates multi-modal analysis

### Critical Validations

The pipeline includes safeguards for:
- Invalid video files or dimensions
- Zero/negative FPS values
- Frame processing failures
- Directory creation for outputs
- Pixel coordinate bounds checking

## Output Structure

For input `video.mp4` with `--extract-face --visualize`, the pipeline generates:
```
output/
├── video_landmarks.mp4           # Annotated video with hands and face
├── video_landmarks.csv           # Unified time-series data
├── video_summary.txt             # Detection statistics for both types
├── video_hand_timeseries.png     # Hand coordinate plots
├── video_face_timeseries.png     # Face coordinate plots
├── video_hand_heatmap.png        # Hand movement heatmap
└── video_face_heatmap.png        # Face movement heatmap
```