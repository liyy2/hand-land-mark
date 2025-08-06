"""
Visualize differential movement patterns with timeline
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
from pd_analysis import PDMovementAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Initialize analyzer
print("Loading landmark data...")
analyzer = PDMovementAnalyzer('/home/yl2428/hand-land-mark/output_holistic/Y00078.1_trimmed_1080p_landmarks.csv')

# Get movement classifications
movement_types = analyzer.detect_movement_type_per_hand(window_size=60, overlap=0.75)  # More overlap for finer detail

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 1, 1, 1.5], hspace=0.3)

# 1. Left hand distance signal
ax1 = fig.add_subplot(gs[0])
left_data = analyzer.left_hand
thumb_left = left_data[left_data['landmark_id'] == analyzer.THUMB_TIP].sort_values('frame')
index_left = left_data[left_data['landmark_id'] == analyzer.INDEX_TIP].sort_values('frame')
merged_left = pd.merge(thumb_left, index_left, on='frame', suffixes=('_thumb', '_index'))

distances_left = np.sqrt(
    (merged_left['x_thumb'] - merged_left['x_index'])**2 +
    (merged_left['y_thumb'] - merged_left['y_index'])**2 +
    (merged_left['z_thumb'] - merged_left['z_index'])**2
).values
time_left = merged_left['frame'].values / analyzer.fps

ax1.plot(time_left, distances_left, 'b-', alpha=0.6, linewidth=0.8)
ax1.set_ylabel('Left Hand\nThumb-Index\nDistance', fontsize=10)
ax1.set_title('Hand Movement Patterns Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Mark tapping peaks
peaks_left, _ = signal.find_peaks(distances_left, prominence=np.std(distances_left)*0.3)
if len(peaks_left) > 0:
    ax1.scatter(time_left[peaks_left], distances_left[peaks_left], 
                c='red', s=20, alpha=0.6, zorder=5)

# 2. Right hand distance signal
ax2 = fig.add_subplot(gs[1])
right_data = analyzer.right_hand
thumb_right = right_data[right_data['landmark_id'] == analyzer.THUMB_TIP].sort_values('frame')
index_right = right_data[right_data['landmark_id'] == analyzer.INDEX_TIP].sort_values('frame')
merged_right = pd.merge(thumb_right, index_right, on='frame', suffixes=('_thumb', '_index'))

distances_right = np.sqrt(
    (merged_right['x_thumb'] - merged_right['x_index'])**2 +
    (merged_right['y_thumb'] - merged_right['y_index'])**2 +
    (merged_right['z_thumb'] - merged_right['z_index'])**2
).values
time_right = merged_right['frame'].values / analyzer.fps

ax2.plot(time_right, distances_right, 'g-', alpha=0.6, linewidth=0.8)
ax2.set_ylabel('Right Hand\nThumb-Index\nDistance', fontsize=10)
ax2.grid(True, alpha=0.3)

# Mark tapping peaks
peaks_right, _ = signal.find_peaks(distances_right, prominence=np.std(distances_right)*0.3)
if len(peaks_right) > 0:
    ax2.scatter(time_right[peaks_right], distances_right[peaks_right], 
                c='red', s=20, alpha=0.6, zorder=5)

# 3. Movement classification timeline - Left hand
ax3 = fig.add_subplot(gs[2])
colors = {'tapping': '#2ecc71', 'tremor': '#e74c3c', 'static': '#95a5a6'}

for window in movement_types.get('left', []):
    start_time = window['start_frame'] / analyzer.fps
    end_time = window['end_frame'] / analyzer.fps
    color = colors[window['movement_type']]
    alpha = min(1.0, 0.3 + window['confidence'] * 0.7)  # Vary alpha by confidence
    
    ax3.barh(0, end_time - start_time, left=start_time, height=0.8,
             color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

ax3.set_ylim([-0.5, 0.5])
ax3.set_ylabel('Left\nClass', fontsize=10)
ax3.set_yticks([])
ax3.grid(True, alpha=0.3, axis='x')

# 4. Movement classification timeline - Right hand
ax4 = fig.add_subplot(gs[3])

for window in movement_types.get('right', []):
    start_time = window['start_frame'] / analyzer.fps
    end_time = window['end_frame'] / analyzer.fps
    color = colors[window['movement_type']]
    alpha = min(1.0, 0.3 + window['confidence'] * 0.7)
    
    ax4.barh(0, end_time - start_time, left=start_time, height=0.8,
             color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

ax4.set_ylim([-0.5, 0.5])
ax4.set_ylabel('Right\nClass', fontsize=10)
ax4.set_yticks([])
ax4.grid(True, alpha=0.3, axis='x')

# 5. Differential movement timeline
ax5 = fig.add_subplot(gs[4])

# Find differential periods
differential_periods = []
for left_window in movement_types.get('left', []):
    for right_window in movement_types.get('right', []):
        overlap_start = max(left_window['start_frame'], right_window['start_frame'])
        overlap_end = min(left_window['end_frame'], right_window['end_frame'])
        
        if overlap_start < overlap_end:
            start_time = overlap_start / analyzer.fps
            end_time = overlap_end / analyzer.fps
            left_type = left_window['movement_type']
            right_type = right_window['movement_type']
            
            # Color based on combination
            if left_type == right_type:
                color = '#3498db'  # Blue for synchronized
                y_pos = 0
            elif 'tapping' in [left_type, right_type] and 'tremor' in [left_type, right_type]:
                color = '#e67e22'  # Orange for tapping vs tremor
                y_pos = 1
            elif 'tapping' in [left_type, right_type] and 'static' in [left_type, right_type]:
                color = '#9b59b6'  # Purple for tapping vs static
                y_pos = 0.5
            else:
                color = '#34495e'  # Dark gray for other
                y_pos = 0
            
            ax5.barh(y_pos, end_time - start_time, left=start_time, height=0.3,
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

ax5.set_ylim([-0.5, 1.5])
ax5.set_ylabel('Movement\nCombination', fontsize=10)
ax5.set_xlabel('Time (seconds)', fontsize=11)
ax5.set_yticks([0, 0.5, 1])
ax5.set_yticklabels(['Sync', 'Tap-Static', 'Tap-Tremor'], fontsize=9)
ax5.grid(True, alpha=0.3, axis='x')

# Add legend
legend_elements = [
    mpatches.Patch(color='#2ecc71', label='Tapping'),
    mpatches.Patch(color='#e74c3c', label='Tremor'),
    mpatches.Patch(color='#95a5a6', label='Static'),
    mpatches.Patch(color='#3498db', label='Synchronized'),
    mpatches.Patch(color='#9b59b6', label='Tap vs Static'),
    mpatches.Patch(color='#e67e22', label='Tap vs Tremor')
]
ax5.legend(handles=legend_elements, loc='upper right', ncol=6, fontsize=9)

# Set consistent x-axis
max_time = max(time_left[-1] if len(time_left) > 0 else 0,
               time_right[-1] if len(time_right) > 0 else 0)
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xlim([0, min(max_time, 40)])  # Show first 40 seconds

plt.suptitle('Differential Hand Movement Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('differential_movement_timeline.png', dpi=150, bbox_inches='tight')
print("\n✓ Timeline visualization saved to: differential_movement_timeline.png")

# Create summary statistics plot
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle('Movement Pattern Statistics', fontsize=14, fontweight='bold')

# Count movement types for each hand
left_counts = {'tapping': 0, 'tremor': 0, 'static': 0}
right_counts = {'tapping': 0, 'tremor': 0, 'static': 0}

for window in movement_types.get('left', []):
    left_counts[window['movement_type']] += 1
for window in movement_types.get('right', []):
    right_counts[window['movement_type']] += 1

# 1. Movement type distribution
ax = axes[0, 0]
x = np.arange(3)
width = 0.35
labels = ['Tapping', 'Tremor', 'Static']
left_vals = [left_counts['tapping'], left_counts['tremor'], left_counts['static']]
right_vals = [right_counts['tapping'], right_counts['tremor'], right_counts['static']]

ax.bar(x - width/2, left_vals, width, label='Left Hand', color='#3498db')
ax.bar(x + width/2, right_vals, width, label='Right Hand', color='#2ecc71')
ax.set_xlabel('Movement Type')
ax.set_ylabel('Number of Windows')
ax.set_title('Movement Type Distribution')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Movement combinations pie chart
ax = axes[0, 1]
combinations = {}
for left_window in movement_types.get('left', []):
    for right_window in movement_types.get('right', []):
        overlap_start = max(left_window['start_frame'], right_window['start_frame'])
        overlap_end = min(left_window['end_frame'], right_window['end_frame'])
        
        if overlap_start < overlap_end:
            duration = (overlap_end - overlap_start) / analyzer.fps
            combo = f"{left_window['movement_type']}-{right_window['movement_type']}"
            if combo not in combinations:
                combinations[combo] = 0
            combinations[combo] += duration

# Simplify combinations
simplified = {
    'Both Tapping': combinations.get('tapping-tapping', 0),
    'Both Static': combinations.get('static-static', 0),
    'Both Tremor': combinations.get('tremor-tremor', 0),
    'Tap vs Static': combinations.get('tapping-static', 0) + combinations.get('static-tapping', 0),
    'Tap vs Tremor': combinations.get('tapping-tremor', 0) + combinations.get('tremor-tapping', 0),
    'Other': sum(v for k, v in combinations.items() if k not in 
                ['tapping-tapping', 'static-static', 'tremor-tremor', 
                 'tapping-static', 'static-tapping', 'tapping-tremor', 'tremor-tapping'])
}

# Remove zero values
simplified = {k: v for k, v in simplified.items() if v > 0}

if simplified:
    ax.pie(simplified.values(), labels=simplified.keys(), autopct='%1.1f%%', startangle=90)
    ax.set_title('Movement Combination Distribution')

# 3. Tapping frequency histogram
ax = axes[1, 0]
all_tap_frequencies = []

# Calculate tapping frequencies from peaks
if len(peaks_left) > 1:
    tap_intervals_left = np.diff(peaks_left) / analyzer.fps
    tap_freq_left = 1.0 / tap_intervals_left
    all_tap_frequencies.extend(tap_freq_left)

if len(peaks_right) > 1:
    tap_intervals_right = np.diff(peaks_right) / analyzer.fps
    tap_freq_right = 1.0 / tap_intervals_right
    all_tap_frequencies.extend(tap_freq_right)

if all_tap_frequencies:
    ax.hist(all_tap_frequencies, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_tap_frequencies), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_tap_frequencies):.2f} Hz')
    ax.set_xlabel('Tapping Frequency (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Tapping Frequency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 4. Movement synchrony over time
ax = axes[1, 1]
time_bins = np.arange(0, max_time, 5)  # 5-second bins
sync_ratio = []

for t_start in time_bins[:-1]:
    t_end = t_start + 5
    sync_time = 0
    diff_time = 0
    
    for left_window in movement_types.get('left', []):
        for right_window in movement_types.get('right', []):
            # Check if windows overlap with this time bin
            overlap_start = max(left_window['start_frame'] / analyzer.fps, t_start)
            overlap_end = min(left_window['end_frame'] / analyzer.fps, t_end)
            
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                if left_window['movement_type'] == right_window['movement_type']:
                    sync_time += duration
                else:
                    diff_time += duration
    
    total = sync_time + diff_time
    if total > 0:
        sync_ratio.append(sync_time / total * 100)
    else:
        sync_ratio.append(50)  # Default to 50% if no data

ax.plot(time_bins[:-1] + 2.5, sync_ratio, 'o-', linewidth=2, markersize=6)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Synchrony (%)')
ax.set_title('Movement Synchrony Over Time')
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('movement_statistics.png', dpi=150)
print("✓ Statistics visualization saved to: movement_statistics.png")

# Print summary
print("\n=== VISUALIZATION SUMMARY ===")
print(f"Total windows analyzed: Left={len(movement_types.get('left', []))}, Right={len(movement_types.get('right', []))}")
print(f"Tapping peaks detected: Left={len(peaks_left)}, Right={len(peaks_right)}")
if all_tap_frequencies:
    print(f"Average tapping frequency: {np.mean(all_tap_frequencies):.2f} Hz (±{np.std(all_tap_frequencies):.2f})")
print(f"Movement combinations found: {len(combinations)}")
print("\nVisualization complete! Check the generated PNG files.")