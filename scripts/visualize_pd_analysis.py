#!/usr/bin/env python
"""
Advanced visualization for Parkinson's Disease movement analysis
Creates multiple plots to visualize tremor, tapping, and movement patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from pd_analysis import PDMovementAnalyzer
import json

def create_comprehensive_visualization(csv_path: str):
    """Create comprehensive PD analysis visualization"""
    
    # Initialize analyzer
    print("Loading and analyzing data...")
    analyzer = PDMovementAnalyzer(csv_path)
    
    # Get all analyses
    tremor = analyzer.calculate_tremor_frequency('both')
    tapping = analyzer.analyze_finger_tapping('both')
    asymmetry = analyzer.calculate_movement_asymmetry()
    bradykinesia = analyzer.detect_bradykinesia('both')
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1])
    
    # Color scheme
    left_color = '#2E86AB'  # Blue for left hand
    right_color = '#A23B72'  # Purple for right hand
    
    # ========== ROW 1: Tremor Frequency Analysis ==========
    # Left hand tremor spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    if 'left' in tremor:
        t = tremor['left']
        ax1.plot(t['frequencies'], t['psd'], color=left_color, linewidth=2, alpha=0.8)
        ax1.fill_between(t['frequencies'], 0, t['psd'], color=left_color, alpha=0.3)
        ax1.axvline(t['peak_frequency'], color='red', linestyle='--', linewidth=2,
                   label=f'Peak: {t["peak_frequency"]:.2f} Hz')
        
        # PD frequency bands
        ax1.axvspan(4, 6, alpha=0.15, color='red', label='PD Rest (4-6 Hz)')
        ax1.axvspan(6, 12, alpha=0.15, color='orange', label='PD Action (6-12 Hz)')
        
        ax1.set_title('Left Hand - Tremor Frequency Spectrum', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.set_xlim([0, 15])
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    # Right hand tremor spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    if 'right' in tremor:
        t = tremor['right']
        ax2.plot(t['frequencies'], t['psd'], color=right_color, linewidth=2, alpha=0.8)
        ax2.fill_between(t['frequencies'], 0, t['psd'], color=right_color, alpha=0.3)
        ax2.axvline(t['peak_frequency'], color='red', linestyle='--', linewidth=2,
                   label=f'Peak: {t["peak_frequency"]:.2f} Hz')
        
        ax2.axvspan(4, 6, alpha=0.15, color='red', label='PD Rest (4-6 Hz)')
        ax2.axvspan(6, 12, alpha=0.15, color='orange', label='PD Action (6-12 Hz)')
        
        ax2.set_title('Right Hand - Tremor Frequency Spectrum', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_xlim([0, 15])
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # Tremor comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if 'left' in tremor and 'right' in tremor:
        hands = ['Left', 'Right']
        frequencies = [tremor['left']['peak_frequency'], tremor['right']['peak_frequency']]
        amplitudes = [tremor['left']['tremor_amplitude'], tremor['right']['tremor_amplitude']]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, frequencies, width, label='Frequency (Hz)', 
                       color=[left_color, right_color], alpha=0.7)
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, amplitudes, width, label='Amplitude', 
                            color=[left_color, right_color], alpha=0.5, hatch='///')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(hands)
        ax3.set_ylabel('Frequency (Hz)', color='black')
        ax3_twin.set_ylabel('Amplitude', color='gray')
        ax3.set_title('Tremor Comparison', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars1, frequencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, amplitudes):
            height = bar.get_height()
            ax3_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ========== ROW 2: Finger Tapping Pattern ==========
    # Left hand tapping
    ax4 = fig.add_subplot(gs[1, 0])
    if 'left' in tapping:
        t = tapping['left']
        if len(t['distances']) > 0:
            time = np.arange(len(t['distances'])) / analyzer.fps
            ax4.plot(time, t['distances'], color=left_color, linewidth=1.5, alpha=0.6)
            
            if len(t['peaks']) > 0:
                ax4.scatter(time[t['peaks']], t['distances'][t['peaks']], 
                          color='red', s=50, zorder=5, label='Tap peaks')
            if len(t['valleys']) > 0:
                ax4.scatter(time[t['valleys']], t['distances'][t['valleys']], 
                          color='green', s=30, zorder=5, label='Tap valleys')
            
            ax4.set_title(f'Left Hand - Tapping Pattern (Rate: {t["tap_frequency"]:.2f} Hz)', 
                         fontsize=12, fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Thumb-Index Distance')
            ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(True, alpha=0.3)
    
    # Right hand tapping
    ax5 = fig.add_subplot(gs[1, 1])
    if 'right' in tapping:
        t = tapping['right']
        if len(t['distances']) > 0:
            time = np.arange(len(t['distances'])) / analyzer.fps
            ax5.plot(time, t['distances'], color=right_color, linewidth=1.5, alpha=0.6)
            
            if len(t['peaks']) > 0:
                ax5.scatter(time[t['peaks']], t['distances'][t['peaks']], 
                          color='red', s=50, zorder=5, label='Tap peaks')
            if len(t['valleys']) > 0:
                ax5.scatter(time[t['valleys']], t['distances'][t['valleys']], 
                          color='green', s=30, zorder=5, label='Tap valleys')
            
            ax5.set_title(f'Right Hand - Tapping Pattern (Rate: {t["tap_frequency"]:.2f} Hz)', 
                         fontsize=12, fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Thumb-Index Distance')
            ax5.legend(loc='upper right', fontsize=9)
            ax5.grid(True, alpha=0.3)
    
    # Tapping metrics comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if 'left' in tapping and 'right' in tapping:
        metrics = ['Tap\nFreq (Hz)', 'Amplitude\nDecrement (%)', 'Rhythm\nCV', 'Hesitations']
        left_vals = [
            tapping['left']['tap_frequency'],
            tapping['left']['amplitude_decrement'],
            tapping['left']['rhythm_cv'],
            tapping['left']['hesitations']
        ]
        right_vals = [
            tapping['right']['tap_frequency'],
            tapping['right']['amplitude_decrement'],
            tapping['right']['rhythm_cv'],
            tapping['right']['hesitations']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax6.bar(x - width/2, left_vals, width, label='Left', color=left_color, alpha=0.7)
        ax6.bar(x + width/2, right_vals, width, label='Right', color=right_color, alpha=0.7)
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics, fontsize=9)
        ax6.set_ylabel('Value')
        ax6.set_title('Tapping Metrics Comparison', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    # ========== ROW 3: Amplitude Progression & Bradykinesia ==========
    # Left hand amplitude progression
    ax7 = fig.add_subplot(gs[2, 0])
    if 'left' in tapping and len(tapping['left']['tap_amplitudes']) > 0:
        t = tapping['left']
        tap_nums = np.arange(len(t['tap_amplitudes']))
        ax7.scatter(tap_nums, t['tap_amplitudes'], color=left_color, s=40, alpha=0.7)
        ax7.plot(tap_nums, t['tap_amplitudes'], color=left_color, linewidth=1, alpha=0.5)
        
        # Trend line
        if len(tap_nums) > 1:
            z = np.polyfit(tap_nums, t['tap_amplitudes'], 1)
            p = np.poly1d(z)
            ax7.plot(tap_nums, p(tap_nums), 'r--', linewidth=2,
                    label=f'Decrement: {t["amplitude_decrement"]:.1f}%')
        
        ax7.set_title('Left Hand - Amplitude Progression', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Tap Number')
        ax7.set_ylabel('Amplitude')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # Right hand amplitude progression
    ax8 = fig.add_subplot(gs[2, 1])
    if 'right' in tapping and len(tapping['right']['tap_amplitudes']) > 0:
        t = tapping['right']
        tap_nums = np.arange(len(t['tap_amplitudes']))
        ax8.scatter(tap_nums, t['tap_amplitudes'], color=right_color, s=40, alpha=0.7)
        ax8.plot(tap_nums, t['tap_amplitudes'], color=right_color, linewidth=1, alpha=0.5)
        
        # Trend line
        if len(tap_nums) > 1:
            z = np.polyfit(tap_nums, t['tap_amplitudes'], 1)
            p = np.poly1d(z)
            ax8.plot(tap_nums, p(tap_nums), 'r--', linewidth=2,
                    label=f'Decrement: {t["amplitude_decrement"]:.1f}%')
        
        ax8.set_title('Right Hand - Amplitude Progression', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Tap Number')
        ax8.set_ylabel('Amplitude')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Bradykinesia assessment
    ax9 = fig.add_subplot(gs[2, 2])
    if 'left' in bradykinesia and 'right' in bradykinesia:
        # Create radar chart for bradykinesia indicators
        categories = ['Slow\nTapping', 'Amplitude\nDecrement', 'Irregular\nRhythm', 
                     'Progressive\nSlowing', 'Hesitations']
        
        # Score each indicator (0 or 1)
        left_scores = []
        right_scores = []
        
        for indicator in ['slow_tapping', 'amplitude_decrement', 'irregular_rhythm', 
                         'progressive_slowing', 'hesitations']:
            left_scores.append(1 if indicator in bradykinesia['left']['indicators'] else 0)
            right_scores.append(1 if indicator in bradykinesia['right']['indicators'] else 0)
        
        # Bar chart for bradykinesia indicators
        x = np.arange(len(categories))
        width = 0.35
        
        ax9.bar(x - width/2, left_scores, width, label=f'Left ({bradykinesia["left"]["severity"]})', 
               color=left_color, alpha=0.7)
        ax9.bar(x + width/2, right_scores, width, label=f'Right ({bradykinesia["right"]["severity"]})', 
               color=right_color, alpha=0.7)
        
        ax9.set_xticks(x)
        ax9.set_xticklabels(categories, fontsize=9)
        ax9.set_ylabel('Present (1) / Absent (0)')
        ax9.set_title('Bradykinesia Indicators', fontsize=12, fontweight='bold')
        ax9.legend()
        ax9.set_ylim([0, 1.2])
        ax9.grid(True, alpha=0.3, axis='y')
    
    # ========== ROW 4: Clinical Summary ==========
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    
    # Create summary text
    summary_text = "CLINICAL SUMMARY\n" + "="*80 + "\n\n"
    
    # Tremor summary
    summary_text += "TREMOR ANALYSIS:\n"
    for hand in ['left', 'right']:
        if hand in tremor:
            t = tremor[hand]
            summary_text += f"  {hand.capitalize()} Hand: {t['peak_frequency']:.2f} Hz, "
            summary_text += f"Amplitude: {t['tremor_amplitude']:.4f}, "
            summary_text += f"Tremor: {'Yes' if t['has_tremor'] else 'No'}\n"
    
    # Tapping summary
    summary_text += "\nFINGER TAPPING:\n"
    for hand in ['left', 'right']:
        if hand in tapping:
            t = tapping[hand]
            summary_text += f"  {hand.capitalize()} Hand: {t['tap_count']} taps, "
            summary_text += f"{t['tap_frequency']:.2f} Hz, "
            summary_text += f"Decrement: {t['amplitude_decrement']:.1f}%\n"
    
    # Asymmetry summary
    if asymmetry.get('has_both_hands'):
        summary_text += f"\nMOVEMENT ASYMMETRY:\n"
        summary_text += f"  Tremor: {asymmetry.get('tremor_asymmetry', 0):.1%}, "
        summary_text += f"Tapping: {asymmetry.get('tapping_asymmetry', 0):.1%}, "
        summary_text += f"More Affected: {asymmetry.get('more_affected', 'unknown')}\n"
    
    # Bradykinesia summary
    summary_text += "\nBRADYKINESIA:\n"
    for hand in ['left', 'right']:
        if hand in bradykinesia:
            b = bradykinesia[hand]
            summary_text += f"  {hand.capitalize()} Hand: Score {b['bradykinesia_score']}/5 ({b['severity']})\n"
    
    # Add text to plot
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, 
             fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    # Main title
    fig.suptitle('Parkinson\'s Disease Movement Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig

def create_time_series_plot(csv_path: str):
    """Create time series plot of hand movements"""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Process each hand
    for i, (hand, color) in enumerate([('Left', '#2E86AB'), ('Right', '#A23B72')]):
        hand_data = df[df['label'] == hand]
        
        # Get index finger tip data (landmark 8)
        index_tip = hand_data[hand_data['landmark_id'] == 8].sort_values('frame')
        
        if len(index_tip) > 0:
            # Convert frame to time
            time = index_tip['frame'].values / 119.0  # 119 fps
            
            # Plot X coordinate
            axes[i, 0].plot(time, index_tip['x'].values, color=color, linewidth=0.5, alpha=0.8)
            axes[i, 0].set_title(f'{hand} Hand - Index Finger X Position', fontweight='bold')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Normalized X Position')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot Y coordinate
            axes[i, 1].plot(time, index_tip['y'].values, color=color, linewidth=0.5, alpha=0.8)
            axes[i, 1].set_title(f'{hand} Hand - Index Finger Y Position', fontweight='bold')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Normalized Y Position')
            axes[i, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Hand Movement Time Series', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    csv_path = '/home/yl2428/hand-land-mark/output_holistic/Y00078.1_trimmed_1080p_landmarks.csv'
    
    print("Creating comprehensive PD analysis visualization...")
    fig1 = create_comprehensive_visualization(csv_path)
    fig1.savefig('pd_analysis_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved: pd_analysis_dashboard.png")
    
    print("Creating time series visualization...")
    fig2 = create_time_series_plot(csv_path)
    fig2.savefig('pd_time_series.png', dpi=150, bbox_inches='tight')
    print("Saved: pd_time_series.png")
    
    plt.show()
    print("\nVisualization complete!")