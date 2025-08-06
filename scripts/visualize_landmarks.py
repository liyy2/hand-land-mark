import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os

def plot_landmark_timeseries(df: pd.DataFrame, 
                           landmark_ids: List[int] = None,
                           landmark_type: str = 'hand',
                           body_part_label: str = None,
                           save_path: str = None):
    """
    Plot time series of landmarks.
    
    Args:
        df: DataFrame with landmark data
        landmark_ids: List of landmark IDs to plot (default: all)
        landmark_type: Type of landmarks to plot ('hand' or 'face')
        body_part_label: Filter by label (e.g., 'Left', 'Right', 'Face_0')
        save_path: Path to save the plot
    """
    if len(df) == 0:
        print("No landmark data to plot")
        return
    
    # Filter by landmark type
    filtered_df = df[df['landmark_type'] == landmark_type].copy()
    
    if body_part_label:
        filtered_df = filtered_df[filtered_df['label'] == body_part_label]
    
    if landmark_ids is None:
        # Default landmark selection based on type
        if landmark_type == 'hand':
            landmark_ids = [0, 4, 8, 12, 16, 20]  # Key hand points
        else:  # face
            landmark_ids = [0, 13, 14, 61, 291, 84]  # Key face points
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    title = f'{landmark_type.capitalize()} Landmark Time Series'
    if body_part_label:
        title += f' - {body_part_label}'
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(landmark_ids)))
    
    for idx, landmark_id in enumerate(landmark_ids):
        landmark_data = filtered_df[filtered_df['landmark_id'] == landmark_id]
        
        if len(landmark_data) > 0:
            grouped = landmark_data.groupby(['timestamp', 'body_part_id']).first().reset_index()
            
            axes[0].plot(grouped['timestamp'], grouped['x'], 
                        label=f'Landmark {landmark_id}', color=colors[idx], alpha=0.7)
            axes[1].plot(grouped['timestamp'], grouped['y'], 
                        color=colors[idx], alpha=0.7)
            axes[2].plot(grouped['timestamp'], grouped['z'], 
                        color=colors[idx], alpha=0.7)
    
    axes[0].set_ylabel('X Coordinate')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Y Coordinate')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Z Coordinate (Depth)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_movement_heatmap(df: pd.DataFrame, 
                         video_width: int, 
                         video_height: int,
                         landmark_type: str = 'hand',
                         save_path: str = None):
    """
    Create a heatmap showing landmark movement patterns.
    
    Args:
        df: DataFrame with landmark data
        video_width: Width of the video
        video_height: Height of the video
        landmark_type: Type of landmarks ('hand' or 'face')
        save_path: Path to save the plot
    """
    if len(df) == 0:
        print("No landmark data to plot")
        return
    
    # Filter by landmark type
    filtered_df = df[df['landmark_type'] == landmark_type]
    
    if landmark_type == 'hand':
        labels = ['Left', 'Right']
        title = 'Hand Movement Heatmap'
    else:
        # Get unique face labels
        labels = sorted(filtered_df['label'].unique())
        title = 'Face Movement Heatmap'
    
    num_plots = len(labels)
    fig, axes = plt.subplots(1, num_plots, figsize=(7.5 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)
    
    for idx, label in enumerate(labels):
        part_data = filtered_df[filtered_df['label'] == label]
        
        if len(part_data) > 0:
            heatmap, xedges, yedges = np.histogram2d(
                part_data['x_pixel'], 
                part_data['y_pixel'],
                bins=[50, 50],
                range=[[0, video_width], [0, video_height]]
            )
            
            im = axes[idx].imshow(heatmap.T, origin='lower', 
                                 extent=[0, video_width, 0, video_height],
                                 aspect='auto', cmap='hot')
            axes[idx].set_title(f'{label}')
            axes[idx].set_xlabel('X Position (pixels)')
            axes[idx].set_ylabel('Y Position (pixels)')
            axes[idx].invert_yaxis()
            
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Frequency')
        else:
            axes[idx].text(0.5, 0.5, f'No {label} detected', 
                         ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{label}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()

def create_landmark_summary(df: pd.DataFrame, save_path: str = None):
    """
    Create a summary of landmark detection statistics.
    
    Args:
        df: DataFrame with landmark data
        save_path: Path to save the summary
    """
    if len(df) == 0:
        print("No landmark data to summarize")
        return {}
    
    max_frame = df['frame'].max()
    detection_rate = (df['frame'].nunique() / (max_frame + 1) * 100) if max_frame >= 0 else 0
    
    summary = {
        'Total Frames': df['frame'].nunique(),
        'Detection Rate': f"{detection_rate:.1f}%",
        'Average Confidence': f"{df['confidence'].mean():.3f}",
        'Min Confidence': f"{df['confidence'].min():.3f}",
        'Max Confidence': f"{df['confidence'].max():.3f}"
    }
    
    # Hand-specific statistics
    hand_df = df[df['landmark_type'] == 'hand']
    if len(hand_df) > 0:
        summary['Frames with Hands'] = hand_df['frame'].nunique()
        summary['Left Hand Detections'] = len(hand_df[hand_df['label'] == 'Left']['frame'].unique())
        summary['Right Hand Detections'] = len(hand_df[hand_df['label'] == 'Right']['frame'].unique())
    
    # Face-specific statistics
    face_df = df[df['landmark_type'] == 'face']
    if len(face_df) > 0:
        summary['Frames with Faces'] = face_df['frame'].nunique()
        summary['Number of Faces Detected'] = face_df['body_part_id'].nunique()
    
    print("\n=== Landmark Detection Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    if save_path:
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        with open(save_path, 'w') as f:
            f.write("Landmark Detection Summary\n")
            f.write("=" * 40 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        print(f"\nSummary saved to: {save_path}")
    
    return summary