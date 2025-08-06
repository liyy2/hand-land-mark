# Hand and Face Landmark Detection Pipeline

This pipeline detects hand and face landmarks in videos using MediaPipe, visualizes them, and exports time-series data.

## Features

- Detects up to 2 hands per frame (21 landmarks each)
- Detects face landmarks (468 3D points with refine_landmarks)
- Draws landmarks with connections on video
- Exports unified CSV time-series data for both hands and face
- Generates separate visualization plots for hands and face
- Creates movement heatmaps for spatial analysis
- Provides comprehensive detection statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage (hands only):
```bash
python main.py input_video.mp4
```

With face detection:
```bash
python main.py input_video.mp4 --extract-face --visualize
```

Full options:
```bash
python main.py input_video.mp4 --output-dir results --extract-face --max-hands 2 --max-faces 1 --visualize
```

### Arguments

**General:**
- `input_video`: Path to input video file
- `--output-dir`: Directory for output files (default: `output`)
- `--visualize`: Generate visualization plots

**Hand Detection:**
- `--max-hands`: Maximum hands to detect (default: 2)
- `--hand-detection-confidence`: Minimum hand detection confidence (default: 0.5)
- `--hand-tracking-confidence`: Minimum hand tracking confidence (default: 0.5)

**Face Detection:**
- `--extract-face`: Enable face landmark extraction
- `--max-faces`: Maximum faces to detect (default: 1)
- `--face-detection-confidence`: Minimum face detection confidence (default: 0.5)
- `--face-tracking-confidence`: Minimum face tracking confidence (default: 0.5)

## Output Files

- `*_landmarks.mp4`: Video with hand and/or face landmarks drawn
- `*_landmarks.csv`: Unified time-series data of all landmarks
- `*_summary.txt`: Detection statistics for both hands and face
- `*_hand_timeseries.png`: Hand landmark coordinate plots (if --visualize)
- `*_face_timeseries.png`: Face landmark coordinate plots (if --visualize and --extract-face)
- `*_hand_heatmap.png`: Hand movement heatmap (if --visualize)
- `*_face_heatmap.png`: Face movement heatmap (if --visualize and --extract-face)

## Landmark IDs

**MediaPipe Hand Landmarks (21 points):**
- 0: Wrist
- 1-4: Thumb
- 5-8: Index finger
- 9-12: Middle finger
- 13-16: Ring finger
- 17-20: Pinky finger

**MediaPipe Face Landmarks (468 points):**
- Face mesh includes detailed 3D facial geometry
- Key regions: face oval, eyes, eyebrows, nose, mouth, lips
- With refine_landmarks enabled: includes iris tracking

## CSV Format

The exported unified CSV contains:
- `frame`: Frame number
- `timestamp`: Time in seconds
- `landmark_type`: 'hand' or 'face'
- `body_part_id`: Identifier for specific hand (0-1) or face (0+)
- `label`: 'Left', 'Right' for hands, or 'Face_0', 'Face_1' for faces
- `confidence`: Detection confidence
- `landmark_id`: Landmark ID (0-20 for hands, 0-467 for face)
- `x`, `y`, `z`: Normalized coordinates (0-1)
- `x_pixel`, `y_pixel`: Pixel coordinates