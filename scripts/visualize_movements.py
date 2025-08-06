"""
Visualize hand movements to debug tapping detection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pd_analysis import PDMovementAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Initialize analyzer
print("Loading landmark data...")
analyzer = PDMovementAnalyzer('/home/yl2428/hand-land-mark/output_holistic/Y00078.1_trimmed_1080p_landmarks.csv')

# Create comprehensive visualization
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('Hand Movement Analysis - Tapping vs Tremor Detection', fontsize=16)

# Process each hand
for hand_idx, (hand_name, hand_data) in enumerate([('left', analyzer.left_hand), 
                                                    ('right', analyzer.right_hand)]):
    if len(hand_data) == 0:
        continue
    
    # Get thumb and index data
    thumb = hand_data[hand_data['landmark_id'] == analyzer.THUMB_TIP].sort_values('frame')
    index = hand_data[hand_data['landmark_id'] == analyzer.INDEX_TIP].sort_values('frame')
    merged = pd.merge(thumb, index, on='frame', suffixes=('_thumb', '_index'))
    
    if len(merged) == 0:
        continue
    
    # Calculate distances
    distances = np.sqrt(
        (merged['x_thumb'] - merged['x_index'])**2 +
        (merged['y_thumb'] - merged['y_index'])**2 +
        (merged['z_thumb'] - merged['z_index'])**2
    ).values
    
    time = merged['frame'].values / analyzer.fps
    
    # 1. Raw distance signal
    ax1 = axes[0, hand_idx]
    ax1.plot(time, distances, 'b-', alpha=0.6, linewidth=0.8)
    ax1.set_title(f'{hand_name.upper()} Hand - Thumb-Index Distance')
    ax1.set_ylabel('Distance')
    ax1.grid(True, alpha=0.3)
    
    # Detect peaks with different thresholds
    # More sensitive peak detection for tapping
    peaks_sensitive, props_sensitive = signal.find_peaks(
        distances, 
        prominence=np.std(distances)*0.2,  # Very low threshold
        distance=int(analyzer.fps * 0.15)  # Min 0.15s between taps
    )
    
    peaks_normal, props_normal = signal.find_peaks(
        distances,
        prominence=np.std(distances)*0.5,
        distance=int(analyzer.fps * 0.2)
    )
    
    # Mark peaks
    if len(peaks_sensitive) > 0:
        ax1.plot(time[peaks_sensitive], distances[peaks_sensitive], 'ro', 
                markersize=4, label=f'Sensitive peaks (n={len(peaks_sensitive)})', alpha=0.5)
    if len(peaks_normal) > 0:
        ax1.plot(time[peaks_normal], distances[peaks_normal], 'go', 
                markersize=6, label=f'Normal peaks (n={len(peaks_normal)})')
    ax1.legend(fontsize=8)
    
    # 2. Detrended signal
    ax2 = axes[1, hand_idx]
    distances_detrended = signal.detrend(distances)
    ax2.plot(time, distances_detrended, 'g-', alpha=0.6, linewidth=0.8)
    ax2.set_title(f'Detrended Signal')
    ax2.set_ylabel('Detrended Distance')
    ax2.grid(True, alpha=0.3)
    
    # 3. Power spectrum
    ax3 = axes[2, hand_idx]
    freqs, psd = signal.welch(distances_detrended, analyzer.fps, nperseg=min(256, len(distances)))
    ax3.semilogy(freqs, psd, 'b-')
    ax3.set_title(f'Power Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_xlim([0, 20])
    ax3.grid(True, alpha=0.3)
    
    # Mark frequency bands
    ax3.axvspan(0.5, 3, alpha=0.2, color='green', label='Tapping (0.5-3 Hz)')
    ax3.axvspan(4, 6, alpha=0.2, color='red', label='PD Tremor (4-6 Hz)')
    ax3.axvspan(6, 12, alpha=0.2, color='orange', label='Action Tremor (6-12 Hz)')
    
    # Find dominant frequency
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    ax3.axvline(peak_freq, color='black', linestyle='--', 
               label=f'Peak: {peak_freq:.2f} Hz')
    ax3.legend(fontsize=8)
    
    # 4. Spectrogram (time-frequency analysis)
    ax4 = axes[3, hand_idx]
    f, t_spec, Sxx = signal.spectrogram(distances_detrended, analyzer.fps, 
                                        nperseg=min(128, len(distances)//4))
    
    # Plot spectrogram
    pcm = ax4.pcolormesh(t_spec + time[0], f, 10 * np.log10(Sxx + 1e-10), 
                        shading='gouraud', cmap='viridis')
    ax4.set_title(f'Spectrogram')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylim([0, 15])
    plt.colorbar(pcm, ax=ax4, label='Power (dB)')
    
    # Print statistics
    print(f"\n{hand_name.upper()} HAND STATISTICS:")
    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  Std deviation: {np.std(distances):.4f}")
    print(f"  Signal variance: {np.var(distances_detrended):.6f}")
    print(f"  Peaks detected (sensitive): {len(peaks_sensitive)}")
    print(f"  Peaks detected (normal): {len(peaks_normal)}")
    
    if len(peaks_sensitive) > 1:
        tap_intervals = np.diff(peaks_sensitive) / analyzer.fps
        tap_freq = 1.0 / np.mean(tap_intervals)
        print(f"  Tapping frequency (sensitive): {tap_freq:.2f} Hz")
        print(f"  Tap interval CV: {np.std(tap_intervals)/np.mean(tap_intervals):.2f}")
    
    # Analyze frequency content
    tapping_band = (freqs >= 0.5) & (freqs <= 3)
    tremor_band = (freqs >= 4) & (freqs <= 6)
    
    tapping_power = np.sum(psd[tapping_band])
    tremor_power = np.sum(psd[tremor_band])
    total_power = np.sum(psd)
    
    print(f"  Power in tapping band (0.5-3 Hz): {tapping_power/total_power*100:.1f}%")
    print(f"  Power in tremor band (4-6 Hz): {tremor_power/total_power*100:.1f}%")

plt.tight_layout()
plt.savefig('movement_visualization.png', dpi=150)
print(f"\n✓ Visualization saved to: movement_visualization.png")

# Also create a zoomed-in view of first 10 seconds
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
fig2.suptitle('First 10 Seconds - Detailed View', fontsize=14)

for hand_idx, (hand_name, hand_data) in enumerate([('left', analyzer.left_hand), 
                                                    ('right', analyzer.right_hand)]):
    if len(hand_data) == 0:
        continue
    
    # Get thumb and index data
    thumb = hand_data[hand_data['landmark_id'] == analyzer.THUMB_TIP].sort_values('frame')
    index = hand_data[hand_data['landmark_id'] == analyzer.INDEX_TIP].sort_values('frame')
    merged = pd.merge(thumb, index, on='frame', suffixes=('_thumb', '_index'))
    
    # Filter to first 10 seconds
    time_limit = 10
    mask = (merged['frame'] / analyzer.fps) <= time_limit
    merged = merged[mask]
    
    if len(merged) == 0:
        continue
    
    # Calculate distances
    distances = np.sqrt(
        (merged['x_thumb'] - merged['x_index'])**2 +
        (merged['y_thumb'] - merged['y_index'])**2 +
        (merged['z_thumb'] - merged['z_index'])**2
    ).values
    
    time = merged['frame'].values / analyzer.fps
    
    ax = axes2[hand_idx]
    ax.plot(time, distances, 'b-', alpha=0.7, linewidth=1.0)
    ax.set_title(f'{hand_name.upper()} Hand - First 10 seconds')
    ax.set_ylabel('Thumb-Index Distance')
    if hand_idx == 1:
        ax.set_xlabel('Time (s)')
    
    # Detect valleys (tap closures) as well as peaks
    peaks, _ = signal.find_peaks(distances, prominence=np.std(distances)*0.3)
    valleys, _ = signal.find_peaks(-distances, prominence=np.std(distances)*0.3)
    
    if len(peaks) > 0:
        ax.plot(time[peaks], distances[peaks], 'ro', markersize=8, 
               label=f'Peaks/Opens (n={len(peaks)})')
    if len(valleys) > 0:
        ax.plot(time[valleys], distances[valleys], 'go', markersize=6, 
               label=f'Valleys/Closes (n={len(valleys)})')
    
    # Add horizontal line at mean
    ax.axhline(np.mean(distances), color='gray', linestyle='--', alpha=0.5, 
              label=f'Mean: {np.mean(distances):.3f}')
    
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('movement_first_10s.png', dpi=150)
print(f"✓ Zoomed view saved to: movement_first_10s.png")