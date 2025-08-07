"""
Visualize enhanced tremor detection with multi-landmark analysis
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

# Get multi-landmark tremor analysis
print("Analyzing tremor across all landmarks...")
multi_tremor = analyzer.calculate_multi_landmark_tremor('both')

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Color map for landmarks
landmark_colors = {
    'wrist': '#e74c3c',
    'thumb_tip': '#3498db',
    'index_tip': '#2ecc71',
    'middle_tip': '#f39c12',
    'ring_tip': '#9b59b6',
    'pinky_tip': '#1abc9c',
    'index_mcp': '#34495e',
    'middle_mcp': '#95a5a6',
    'center_of_mass': '#000000'
}

# Landmark IDs
landmark_ids = {
    'wrist': 0,
    'thumb_tip': 4,
    'index_tip': 8,
    'middle_tip': 12,
    'ring_tip': 16,
    'pinky_tip': 20,
    'index_mcp': 5,
    'middle_mcp': 9
}

# Process each hand
for hand_idx, (hand_name, hand_data) in enumerate([('left', analyzer.left_hand), 
                                                    ('right', analyzer.right_hand)]):
    
    # 1. Time series of all landmarks (top row)
    ax1 = fig.add_subplot(gs[0, hand_idx])
    
    # Plot each landmark's movement
    for lm_name, lm_id in landmark_ids.items():
        lm_data = hand_data[hand_data['landmark_id'] == lm_id].sort_values('frame')
        if len(lm_data) > 0:
            time = lm_data['frame'].values / analyzer.fps
            # Use y-coordinate for visualization (vertical movement)
            y_vals = lm_data['y'].values
            y_detrended = signal.detrend(y_vals)
            ax1.plot(time[:500], y_detrended[:500], 
                    label=lm_name, color=landmark_colors[lm_name], 
                    alpha=0.6, linewidth=0.8)
    
    # Add center of mass
    frames = sorted(hand_data['frame'].unique())[:500]
    com_y = []
    com_time = []
    for frame in frames:
        frame_data = hand_data[hand_data['frame'] == frame]
        if len(frame_data) >= 21:
            com_y.append(frame_data['y'].mean())
            com_time.append(frame / analyzer.fps)
    
    if com_y:
        com_y_detrended = signal.detrend(com_y)
        ax1.plot(com_time, com_y_detrended, 'k-', 
                label='COM', linewidth=2, alpha=0.8)
    
    ax1.set_title(f'{hand_name.upper()} Hand - All Landmarks Movement', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Detrended Y Position')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 4])  # Show first 4 seconds
    
    # 2. Power spectrum for each landmark (second row)
    ax2 = fig.add_subplot(gs[1, hand_idx])
    
    for lm_name, lm_id in landmark_ids.items():
        lm_data = hand_data[hand_data['landmark_id'] == lm_id].sort_values('frame')
        if len(lm_data) > 100:
            # Calculate PSD
            y_vals = lm_data['y'].values
            y_detrended = signal.detrend(y_vals)
            freqs, psd = signal.welch(y_detrended, analyzer.fps, 
                                     nperseg=min(256, len(y_detrended)))
            
            # Plot only in relevant frequency range
            freq_mask = freqs <= 15
            ax2.semilogy(freqs[freq_mask], psd[freq_mask], 
                        label=lm_name, color=landmark_colors[lm_name], 
                        alpha=0.7, linewidth=1)
    
    ax2.set_title(f'Power Spectrum - All Landmarks')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 15])
    
    # Mark tremor bands
    ax2.axvspan(0.5, 3, alpha=0.1, color='green', label='Tapping')
    ax2.axvspan(4, 6, alpha=0.1, color='red', label='PD Tremor')
    ax2.axvspan(6, 12, alpha=0.1, color='orange', label='Action Tremor')
    
    # 3. Amplitude comparison (third row)
    ax3 = fig.add_subplot(gs[2, hand_idx])
    
    if hand_name in multi_tremor and 'landmark_results' in multi_tremor[hand_name]:
        landmarks = []
        amplitudes = []
        frequencies = []
        
        for lm_name, lm_data in multi_tremor[hand_name]['landmark_results'].items():
            if isinstance(lm_data, dict) and 'amplitude' in lm_data:
                landmarks.append(lm_name.replace('_', '\n'))
                amplitudes.append(lm_data['amplitude'])
                frequencies.append(lm_data.get('frequency', 0))
        
        if landmarks:
            x_pos = np.arange(len(landmarks))
            bars = ax3.bar(x_pos, amplitudes, color=[landmark_colors.get(lm.replace('\n', '_'), 'gray') 
                                                      for lm in landmarks])
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(landmarks, fontsize=8)
            ax3.set_ylabel('Tremor Amplitude')
            ax3.set_title(f'Amplitude by Landmark')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add frequency labels on bars
            for i, (bar, freq) in enumerate(zip(bars, frequencies)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{freq:.1f}Hz', ha='center', va='bottom', fontsize=7)
    
    # 4. Frequency consistency (fourth row, left)
    if hand_idx == 0:
        ax4 = fig.add_subplot(gs[3, 0])
        
        left_freqs = []
        right_freqs = []
        landmark_names = []
        
        for lm_name in landmark_ids.keys():
            if 'left' in multi_tremor and 'landmark_results' in multi_tremor['left']:
                if lm_name in multi_tremor['left']['landmark_results']:
                    left_freqs.append(multi_tremor['left']['landmark_results'][lm_name].get('frequency', 0))
                else:
                    left_freqs.append(0)
            else:
                left_freqs.append(0)
                
            if 'right' in multi_tremor and 'landmark_results' in multi_tremor['right']:
                if lm_name in multi_tremor['right']['landmark_results']:
                    right_freqs.append(multi_tremor['right']['landmark_results'][lm_name].get('frequency', 0))
                else:
                    right_freqs.append(0)
            else:
                right_freqs.append(0)
            
            landmark_names.append(lm_name.replace('_', ' '))
        
        x = np.arange(len(landmark_names))
        width = 0.35
        
        ax4.bar(x - width/2, left_freqs, width, label='Left Hand', color='#3498db', alpha=0.7)
        ax4.bar(x + width/2, right_freqs, width, label='Right Hand', color='#2ecc71', alpha=0.7)
        
        ax4.set_xlabel('Landmark')
        ax4.set_ylabel('Peak Frequency (Hz)')
        ax4.set_title('Frequency Consistency Across Landmarks')
        ax4.set_xticks(x)
        ax4.set_xticklabels(landmark_names, rotation=45, ha='right', fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add consensus frequency lines
        if 'left' in multi_tremor and 'consensus_frequency' in multi_tremor['left']:
            ax4.axhline(multi_tremor['left']['consensus_frequency'], 
                       color='#3498db', linestyle='--', linewidth=2, alpha=0.5,
                       label=f"Left consensus: {multi_tremor['left']['consensus_frequency']:.2f} Hz")
        if 'right' in multi_tremor and 'consensus_frequency' in multi_tremor['right']:
            ax4.axhline(multi_tremor['right']['consensus_frequency'], 
                       color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5,
                       label=f"Right consensus: {multi_tremor['right']['consensus_frequency']:.2f} Hz")

# 5. Summary statistics (fourth row, middle)
ax5 = fig.add_subplot(gs[3, 1:])
summary_text = "MULTI-LANDMARK TREMOR ANALYSIS SUMMARY\n" + "="*45 + "\n\n"

for hand in ['left', 'right']:
    if hand in multi_tremor and 'consensus_frequency' in multi_tremor[hand]:
        m = multi_tremor[hand]
        summary_text += f"{hand.upper()} HAND:\n"
        summary_text += f"  • Consensus Frequency: {m['consensus_frequency']:.2f} Hz\n"
        summary_text += f"  • Frequency Consistency: {m['frequency_consistency']:.1%}\n"
        summary_text += f"  • Median Amplitude: {m['median_amplitude']:.4f}\n"
        summary_text += f"  • Max Amplitude: {m['max_amplitude']:.4f}\n"
        summary_text += f"  • Tremor Detected: {m['has_tremor']}\n"
        if m['has_tremor']:
            summary_text += f"  • Tremor Type: {m['tremor_type'].replace('_', ' ').title()}\n"
        summary_text += "\n"

# Add interpretation
summary_text += "INTERPRETATION:\n"
if 'left' in multi_tremor and 'right' in multi_tremor:
    left_freq = multi_tremor['left'].get('consensus_frequency', 0)
    right_freq = multi_tremor['right'].get('consensus_frequency', 0)
    
    if abs(left_freq - right_freq) < 0.5:
        summary_text += "  ✓ Bilateral symmetric movement pattern\n"
    else:
        summary_text += "  ⚠ Asymmetric movement frequencies\n"
    
    if left_freq < 4 and right_freq < 4:
        summary_text += "  ✓ Consistent with slow rhythmic tapping\n"
    elif 4 <= left_freq <= 6 or 4 <= right_freq <= 6:
        summary_text += "  ⚠ Frequency in PD tremor range (4-6 Hz)\n"
    
    left_consistency = multi_tremor['left'].get('frequency_consistency', 0)
    right_consistency = multi_tremor['right'].get('frequency_consistency', 0)
    if left_consistency > 0.8 and right_consistency > 0.8:
        summary_text += "  ✓ High frequency consistency across landmarks\n"
        summary_text += "    → Suggests coordinated movement\n"

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax5.axis('off')

plt.suptitle('Enhanced Multi-Landmark Tremor Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/enhanced_tremor_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: visualizations/enhanced_tremor_analysis.png")

# Create second figure for 3D hand visualization
fig2 = plt.figure(figsize=(14, 7))

for hand_idx, (hand_name, hand_data) in enumerate([('left', analyzer.left_hand), 
                                                    ('right', analyzer.right_hand)]):
    ax = fig2.add_subplot(1, 2, hand_idx + 1, projection='3d')
    
    # Get a single frame for hand structure
    frame_num = hand_data['frame'].iloc[1000] if len(hand_data) > 1000 else hand_data['frame'].iloc[0]
    frame_data = hand_data[hand_data['frame'] == frame_num]
    
    if len(frame_data) >= 21:
        # Plot hand structure
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for conn in connections:
            p1 = frame_data[frame_data['landmark_id'] == conn[0]].iloc[0]
            p2 = frame_data[frame_data['landmark_id'] == conn[1]].iloc[0]
            ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']], [p1['z'], p2['z']], 
                   'b-', alpha=0.6, linewidth=1)
        
        # Highlight analyzed landmarks with amplitude-based sizing
        if hand_name in multi_tremor and 'landmark_results' in multi_tremor[hand_name]:
            for lm_name, lm_id in landmark_ids.items():
                lm_point = frame_data[frame_data['landmark_id'] == lm_id]
                if len(lm_point) > 0 and lm_name in multi_tremor[hand_name]['landmark_results']:
                    amp = multi_tremor[hand_name]['landmark_results'][lm_name].get('amplitude', 0.01)
                    # Scale marker size by amplitude
                    marker_size = 100 + amp * 5000
                    lm_point = lm_point.iloc[0]
                    ax.scatter(lm_point['x'], lm_point['y'], lm_point['z'],
                             c=landmark_colors[lm_name], s=marker_size, 
                             alpha=0.8, edgecolors='black', linewidth=1,
                             label=f"{lm_name}: {amp:.3f}")
        
        # Add COM
        com_x = frame_data['x'].mean()
        com_y = frame_data['y'].mean()
        com_z = frame_data['z'].mean()
        ax.scatter(com_x, com_y, com_z, c='black', s=200, marker='*',
                  alpha=0.9, edgecolors='gold', linewidth=2,
                  label='Center of Mass')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{hand_name.upper()} Hand - Landmark Amplitudes\n(Marker size ∝ tremor amplitude)')
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        
        # Set equal aspect ratio
        max_range = 0.3
        ax.set_xlim([com_x - max_range, com_x + max_range])
        ax.set_ylim([com_y - max_range, com_y + max_range])
        ax.set_zlim([com_z - max_range, com_z + max_range])

plt.suptitle('3D Hand Visualization with Tremor Amplitudes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/hand_3d_tremor.png', dpi=150, bbox_inches='tight')
print("✓ 3D visualization saved to: visualizations/hand_3d_tremor.png")

print("\nVisualization complete!")