# Visualization Outputs

This directory contains all generated visualizations from the PD analysis pipeline.

## Files

- `differential_movement_timeline.png` - Timeline showing hand movement patterns and classifications
- `movement_statistics.png` - Statistical analysis of movement patterns
- `movement_visualization.png` - Detailed frequency and signal analysis
- `movement_first_10s.png` - Zoomed view of first 10 seconds
- `pd_analysis_dashboard.png` - PD analysis dashboard
- `pd_time_series.png` - Time series plots of landmarks
- Other diagnostic images

## To regenerate visualizations:

```bash
# Basic PD analysis with visualizations
python pd_analysis.py

# Differential movement analysis
python visualize_differential_patterns.py

# Movement pattern analysis
python visualize_movements.py
```