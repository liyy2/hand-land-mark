#!/usr/bin/env python
"""
Gradio Web Application for Parkinson's Disease Movement Analysis
This module provides a web interface for:
- Video upload and hand landmark detection
- Real-time processing with progress tracking
- PD-specific movement analysis
- Clinical visualization dashboard
"""

import gradio as gr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import json
from pathlib import Path
import shutil
from datetime import datetime
from scipy import stats
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Import existing modules
from unified_landmark_detector import UnifiedLandmarkDetector
from pd_analysis import PDMovementAnalyzer
from scripts.visualize_pd_analysis import create_comprehensive_visualization, create_time_series_plot
from video_annotation import VideoAnnotationManager, PDTaskType, AnnotationIntegrator
from custom_video_player import VideoPlayerWithTime

class PDAnalysisApp:
    """Main application class for PD movement analysis"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="pd_analysis_")
        self.current_session = None
        self.comparison_sessions = {}
        self.annotation_manager = None
        self.current_video_path = None
        self.current_video_duration = 0
        
    def process_video(self, video_path, detection_confidence, enable_face, enhance_contrast, progress=gr.Progress()):
        """
        Process uploaded video for hand tracking
        
        Args:
            video_path: Path to uploaded video
            detection_confidence: Confidence threshold for detection
            enable_face: Whether to detect face landmarks
            enhance_contrast: Whether to apply contrast enhancement
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (output_video_path, csv_path, summary_text)
        """
        if video_path is None:
            return None, None, "Please upload a video first."
        
        try:
            # Create session directory
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = Path(self.temp_dir) / session_id
            session_dir.mkdir(exist_ok=True)
            self.current_session = session_dir
            
            # Initialize progress
            progress(0, desc="Initializing detector...")
            
            # Initialize detector with holistic model
            detector = UnifiedLandmarkDetector(
                extract_hands=True,
                extract_face=enable_face,
                max_num_hands=2,
                hand_detection_confidence=detection_confidence,
                hand_tracking_confidence=detection_confidence
            )
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Define output paths
            output_video = str(session_dir / "tracked_video.mp4")
            output_csv = str(session_dir / "landmarks.csv")
            
            # Process video with progress updates
            progress(0.1, desc=f"Processing {total_frames} frames...")
            
            # Custom processing with progress callback
            landmarks_data = []
            cap = cv2.VideoCapture(video_path)
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                if frame_idx % 10 == 0:
                    progress_val = 0.1 + (frame_idx / total_frames) * 0.8
                    progress(progress_val, desc=f"Processing frame {frame_idx}/{total_frames}")
                
                # Process frame
                annotated_frame, landmarks_dict = detector.process_frame(
                    frame, frame_idx, enhance_contrast=enhance_contrast
                )
                
                # Write annotated frame
                out.write(annotated_frame)
                
                # Collect landmark data
                if landmarks_dict:
                    # Process hands
                    for hand in landmarks_dict.get('hands', []):
                        for landmark in hand['landmarks']:
                            landmarks_data.append({
                                'frame': frame_idx,
                                'timestamp': frame_idx / fps,
                                'landmark_type': 'hand',
                                'body_part_id': hand.get('hand_id', 0),
                                'label': hand['hand_label'],
                                'confidence': hand['confidence'],
                                'landmark_id': landmark['landmark_id'],
                                'x': landmark['x'],
                                'y': landmark['y'],
                                'z': landmark['z'],
                                'x_pixel': int(landmark['x'] * width),
                                'y_pixel': int(landmark['y'] * height)
                            })
                    
                    # Process face if enabled
                    if enable_face:
                        for face in landmarks_dict.get('faces', []):
                            for landmark in face['landmarks']:
                                landmarks_data.append({
                                    'frame': frame_idx,
                                    'timestamp': frame_idx / fps,
                                    'landmark_type': 'face',
                                    'body_part_id': face.get('face_id', 0),
                                    'label': 'Face',
                                    'confidence': face.get('confidence', 1.0),
                                    'landmark_id': landmark['landmark_id'],
                                    'x': landmark['x'],
                                    'y': landmark['y'],
                                    'z': landmark['z'],
                                    'x_pixel': int(landmark['x'] * width),
                                    'y_pixel': int(landmark['y'] * height)
                                })
                
                frame_idx += 1
            
            cap.release()
            out.release()
            
            # Save landmarks to CSV
            progress(0.9, desc="Saving landmark data...")
            df = pd.DataFrame(landmarks_data)
            df.to_csv(output_csv, index=False)
            
            # Generate summary
            summary = self.generate_tracking_summary(df, fps, total_frames)
            
            progress(1.0, desc="Processing complete!")
            
            return output_video, output_csv, summary
            
        except Exception as e:
            return None, None, f"Error processing video: {str(e)}"
    
    def generate_tracking_summary(self, df, fps, total_frames):
        """Generate summary statistics for tracking results"""
        if df.empty:
            return "❌ No landmarks detected in the video."
        
        hand_data = df[df['landmark_type'] == 'hand']
        face_data = df[df['landmark_type'] == 'face'] if 'face' in df['landmark_type'].values else pd.DataFrame()
        
        left_frames = len(hand_data[hand_data['label'] == 'Left']['frame'].unique())
        right_frames = len(hand_data[hand_data['label'] == 'Right']['frame'].unique())
        both_hands_frames = min(left_frames, right_frames)
        
        # Create styled summary
        summary = f"""
        <div style="font-family: 'Inter', system-ui, sans-serif; line-height: 1.6;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h3 style="margin: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📊</span> Tracking Summary
                </h3>
            </div>
            
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: #4a5568; margin-top: 0;">📹 Video Information</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>• Total frames: <strong>{total_frames}</strong></div>
                    <div>• Frame rate: <strong>{fps:.2f} FPS</strong></div>
                    <div>• Duration: <strong>{total_frames/fps:.2f} seconds</strong></div>
                    <div>• Resolution: <strong>Processed</strong></div>
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: #4a5568; margin-top: 0;">✋ Hand Detection Results</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>• Left hand: <strong>{left_frames}/{total_frames}</strong> frames ({100*left_frames/total_frames:.1f}%)</div>
                    <div>• Right hand: <strong>{right_frames}/{total_frames}</strong> frames ({100*right_frames/total_frames:.1f}%)</div>
                    <div>• Both hands: <strong>{both_hands_frames}/{total_frames}</strong> frames ({100*both_hands_frames/total_frames:.1f}%)</div>
                    <div>• Total landmarks: <strong>{len(hand_data):,}</strong></div>
                </div>
            </div>
        """
        
        if not face_data.empty:
            face_frames = face_data['frame'].nunique()
            summary += f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: #4a5568; margin-top: 0;">😊 Face Detection Results</h4>
                <div>• Frames with face: <strong>{face_frames}/{total_frames}</strong> ({100*face_frames/total_frames:.1f}%)</div>
                <div>• Total landmarks: <strong>{len(face_data):,}</strong></div>
            </div>
            """
        
        # Quality assessment
        avg_confidence = df['confidence'].mean()
        detection_rate = 100 * hand_data['frame'].nunique() / total_frames
        
        quality_color = "#48bb78" if detection_rate > 80 else "#ed8936" if detection_rate > 50 else "#f56565"
        quality_label = "Excellent" if detection_rate > 80 else "Good" if detection_rate > 50 else "Poor"
        
        summary += f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                <h4 style="color: #4a5568; margin-top: 0;">📈 Data Quality</h4>
                <div>• Average confidence: <strong>{avg_confidence:.3f}</strong></div>
                <div>• Detection rate: <strong style="color: {quality_color};">{detection_rate:.1f}% ({quality_label})</strong></div>
            </div>
        </div>
        """
        
        return summary
    
    def analyze_pd_movements(self, csv_path, progress=gr.Progress()):
        """
        Perform PD movement analysis on tracked data
        
        Args:
            csv_path: Path to landmarks CSV
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (report_text, tremor_plot, tapping_plot, dashboard_plot)
        """
        if csv_path is None:
            return "No tracking data available. Please process a video first.", None, None, None
        
        try:
            progress(0, desc="Loading landmark data...")
            
            # Initialize analyzer
            analyzer = PDMovementAnalyzer(csv_path)
            
            # Generate analysis report
            progress(0.3, desc="Analyzing tremor frequency...")
            tremor = analyzer.calculate_tremor_frequency('both')
            
            progress(0.5, desc="Analyzing finger tapping...")
            tapping = analyzer.analyze_finger_tapping('both')
            
            progress(0.7, desc="Detecting bradykinesia...")
            bradykinesia = analyzer.detect_bradykinesia('both')
            asymmetry = analyzer.calculate_movement_asymmetry()
            
            # Generate text report
            report_text = self.format_pd_report(tremor, tapping, bradykinesia, asymmetry)
            
            # Generate visualizations
            progress(0.8, desc="Creating visualizations...")
            
            # Tremor frequency plot
            tremor_fig = self.create_tremor_plot(tremor)
            
            # Tapping pattern plot
            tapping_fig = self.create_tapping_plot(tapping, analyzer.fps)
            
            # Comprehensive dashboard
            progress(0.9, desc="Creating dashboard...")
            dashboard_fig = create_comprehensive_visualization(csv_path)
            
            progress(1.0, desc="Analysis complete!")
            
            return report_text, tremor_fig, tapping_fig, dashboard_fig
            
        except Exception as e:
            return f"Error in PD analysis: {str(e)}", None, None, None
    
    def format_pd_report(self, tremor, tapping, bradykinesia, asymmetry):
        """Format PD analysis results as beautiful HTML report"""
        
        # Custom CSS for beautiful styling
        style = """
        <style>
            .report-container {
                font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                color: #2d3748;
                max-width: 1200px;
                margin: 0 auto;
            }
            .report-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .report-title {
                font-size: 2rem;
                font-weight: 700;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            .report-subtitle {
                font-size: 1rem;
                opacity: 0.95;
                margin-top: 0.5rem;
            }
            .section-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.07);
                border: 1px solid #e2e8f0;
            }
            .section-title {
                font-size: 1.3rem;
                font-weight: 600;
                color: #4a5568;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e2e8f0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .hand-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-top: 1rem;
            }
            .hand-card {
                background: #f7fafc;
                border-radius: 8px;
                padding: 1rem;
                border-left: 4px solid;
            }
            .hand-card.left {
                border-left-color: #4299e1;
            }
            .hand-card.right {
                border-left-color: #9f7aea;
            }
            .hand-label {
                font-weight: 600;
                font-size: 1.1rem;
                margin-bottom: 0.75rem;
                color: #2d3748;
            }
            .metric-row {
                display: flex;
                justify-content: space-between;
                padding: 0.4rem 0;
                border-bottom: 1px solid #e2e8f0;
            }
            .metric-row:last-child {
                border-bottom: none;
            }
            .metric-label {
                color: #718096;
                font-size: 0.95rem;
            }
            .metric-value {
                font-weight: 600;
                color: #2d3748;
                font-size: 0.95rem;
            }
            .severity-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            .severity-none {
                background: #c6f6d5;
                color: #22543d;
            }
            .severity-mild {
                background: #fefcbf;
                color: #744210;
            }
            .severity-moderate {
                background: #fed7aa;
                color: #7c2d12;
            }
            .severity-severe {
                background: #feb2b2;
                color: #742a2a;
            }
            .indicator-chip {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                background: #edf2f7;
                color: #4a5568;
                border-radius: 4px;
                font-size: 0.85rem;
                margin: 0.2rem;
            }
            .asymmetry-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .asymmetry-item {
                background: #f7fafc;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            }
            .asymmetry-value {
                font-size: 1.5rem;
                font-weight: 700;
                color: #4a5568;
            }
            .asymmetry-label {
                font-size: 0.9rem;
                color: #718096;
                margin-top: 0.25rem;
            }
            .findings-list {
                background: #f0fff4;
                border-left: 4px solid #48bb78;
                padding: 1rem;
                border-radius: 4px;
                margin-top: 1rem;
            }
            .finding-item {
                color: #22543d;
                margin: 0.5rem 0;
                padding-left: 1.5rem;
                position: relative;
            }
            .finding-item:before {
                content: "▸";
                position: absolute;
                left: 0;
                color: #48bb78;
            }
            .disclaimer {
                background: #fff5f5;
                border: 1px solid #fc8181;
                color: #742a2a;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 2rem;
                font-size: 0.9rem;
                text-align: center;
            }
            .emoji-icon {
                font-size: 1.5rem;
            }
        </style>
        """
        
        # Build HTML report
        html_report = style + '<div class="report-container">'
        
        # Header
        html_report += """
        <div class="report-header">
            <h1 class="report-title">
                <span class="emoji-icon">🏥</span>
                Parkinson's Disease Movement Analysis Report
            </h1>
            <p class="report-subtitle">Comprehensive analysis of tremor, bradykinesia, and movement patterns</p>
        </div>
        """
        
        # Tremor Analysis Section
        html_report += """
        <div class="section-card">
            <h2 class="section-title">
                <span class="emoji-icon">📊</span>
                Tremor Analysis
            </h2>
            <div class="hand-section">
        """
        
        for hand in ['left', 'right']:
            if hand in tremor:
                t = tremor[hand]
                tremor_status = "Detected" if t['has_tremor'] else "Not Detected"
                tremor_color = "#48bb78" if not t['has_tremor'] else "#f56565"
                
                html_report += f"""
                <div class="hand-card {hand}">
                    <div class="hand-label">{hand.capitalize()} Hand</div>
                    <div class="metric-row">
                        <span class="metric-label">Peak Frequency:</span>
                        <span class="metric-value">{t['peak_frequency']:.2f} Hz</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Tremor Status:</span>
                        <span class="metric-value" style="color: {tremor_color};">{tremor_status}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Tremor Type:</span>
                        <span class="metric-value">{t.get('tremor_type', 'N/A').replace('_', ' ').title()}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Amplitude:</span>
                        <span class="metric-value">{t['tremor_amplitude']:.4f}</span>
                    </div>
                </div>
                """
        
        html_report += """
            </div>
        </div>
        """
        
        # Finger Tapping Section
        html_report += """
        <div class="section-card">
            <h2 class="section-title">
                <span class="emoji-icon">👆</span>
                Finger Tapping Analysis
            </h2>
            <div class="hand-section">
        """
        
        for hand in ['left', 'right']:
            if hand in tapping:
                t = tapping[hand]
                html_report += f"""
                <div class="hand-card {hand}">
                    <div class="hand-label">{hand.capitalize()} Hand</div>
                    <div class="metric-row">
                        <span class="metric-label">Total Taps:</span>
                        <span class="metric-value">{t['tap_count']}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Frequency:</span>
                        <span class="metric-value">{t['tap_frequency']:.2f} Hz</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Amplitude Decrement:</span>
                        <span class="metric-value">{t['amplitude_decrement']:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Rhythm Variability:</span>
                        <span class="metric-value">{t['rhythm_cv']:.2f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Hesitations:</span>
                        <span class="metric-value">{t['hesitations']}</span>
                    </div>
                </div>
                """
        
        html_report += """
            </div>
        </div>
        """
        
        # Bradykinesia Section
        html_report += """
        <div class="section-card">
            <h2 class="section-title">
                <span class="emoji-icon">🔬</span>
                Bradykinesia Assessment
            </h2>
            <div class="hand-section">
        """
        
        for hand in ['left', 'right']:
            if hand in bradykinesia:
                b = bradykinesia[hand]
                severity_class = f"severity-{b['severity']}"
                
                html_report += f"""
                <div class="hand-card {hand}">
                    <div class="hand-label">{hand.capitalize()} Hand</div>
                    <div class="metric-row">
                        <span class="metric-label">Score:</span>
                        <span class="metric-value">{b['bradykinesia_score']}/5</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Severity:</span>
                        <span class="severity-badge {severity_class}">{b['severity']}</span>
                    </div>
                    <div style="margin-top: 0.75rem;">
                        <div class="metric-label">Indicators:</div>
                        <div style="margin-top: 0.5rem;">
                """
                
                if b['indicators']:
                    for indicator in b['indicators']:
                        html_report += f'<span class="indicator-chip">{indicator.replace("_", " ").title()}</span>'
                else:
                    html_report += '<span class="indicator-chip">None</span>'
                
                html_report += """
                        </div>
                    </div>
                </div>
                """
        
        html_report += """
            </div>
        </div>
        """
        
        # Asymmetry Section
        if asymmetry.get('has_both_hands'):
            html_report += f"""
            <div class="section-card">
                <h2 class="section-title">
                    <span class="emoji-icon">⚖️</span>
                    Movement Asymmetry
                </h2>
                <div class="asymmetry-grid">
                    <div class="asymmetry-item">
                        <div class="asymmetry-value">{asymmetry.get('tremor_asymmetry', 0):.1%}</div>
                        <div class="asymmetry-label">Tremor Asymmetry</div>
                    </div>
                    <div class="asymmetry-item">
                        <div class="asymmetry-value">{asymmetry.get('tapping_asymmetry', 0):.1%}</div>
                        <div class="asymmetry-label">Tapping Asymmetry</div>
                    </div>
                    <div class="asymmetry-item">
                        <div class="asymmetry-value">{asymmetry.get('more_affected', 'Unknown').upper()}</div>
                        <div class="asymmetry-label">More Affected Side</div>
                    </div>
                </div>
            </div>
            """
        
        # Clinical Interpretation
        html_report += """
        <div class="section-card">
            <h2 class="section-title">
                <span class="emoji-icon">📋</span>
                Clinical Interpretation
            </h2>
        """
        
        # Key findings
        findings = []
        for hand in ['left', 'right']:
            if hand in tremor and tremor[hand]['has_tremor']:
                if 4 <= tremor[hand]['peak_frequency'] <= 6:
                    findings.append(f"{hand.capitalize()} hand shows resting tremor in PD range (4-6 Hz)")
            
            if hand in bradykinesia and bradykinesia[hand]['severity'] in ['moderate', 'severe']:
                findings.append(f"{hand.capitalize()} hand shows {bradykinesia[hand]['severity']} bradykinesia")
        
        if asymmetry.get('tremor_asymmetry', 0) > 0.3:
            findings.append(f"Significant tremor asymmetry detected ({asymmetry['tremor_asymmetry']:.1%})")
        
        if findings:
            html_report += '<div class="findings-list">'
            for finding in findings:
                html_report += f'<div class="finding-item">{finding}</div>'
            html_report += '</div>'
        else:
            html_report += '<div class="findings-list"><div class="finding-item">No significant PD indicators detected in this recording</div></div>'
        
        html_report += """
        </div>
        
        <div class="disclaimer">
            ⚠️ <strong>Important:</strong> This analysis is for research purposes only and should not be used for clinical diagnosis without professional medical consultation.
        </div>
        
        </div>
        """
        
        return html_report
    
    def create_tremor_plot(self, tremor):
        """Create tremor frequency spectrum plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, hand in enumerate(['left', 'right']):
            if hand in tremor:
                t = tremor[hand]
                ax = axes[i]
                
                # Plot PSD
                ax.plot(t['frequencies'], t['psd'], 'b-', linewidth=2, alpha=0.8)
                ax.fill_between(t['frequencies'], 0, t['psd'], alpha=0.3)
                
                # Mark peak frequency
                ax.axvline(t['peak_frequency'], color='red', linestyle='--', 
                          label=f'Peak: {t["peak_frequency"]:.2f} Hz')
                
                # PD frequency ranges
                ax.axvspan(4, 6, alpha=0.2, color='red', label='PD Rest (4-6 Hz)')
                ax.axvspan(6, 12, alpha=0.2, color='orange', label='PD Action (6-12 Hz)')
                
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density')
                ax.set_title(f'{hand.capitalize()} Hand - Tremor Spectrum')
                ax.set_xlim([0, 15])
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Tremor Frequency Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_tapping_plot(self, tapping, fps):
        """Create finger tapping pattern plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, hand in enumerate(['left', 'right']):
            if hand in tapping:
                t = tapping[hand]
                
                # Tapping pattern
                ax1 = axes[i, 0]
                if len(t['distances']) > 0:
                    time = np.arange(len(t['distances'])) / fps
                    ax1.plot(time, t['distances'], 'b-', alpha=0.6)
                    
                    if len(t['peaks']) > 0:
                        ax1.scatter(time[t['peaks']], t['distances'][t['peaks']], 
                                  color='red', s=50, label='Peaks')
                    if len(t['valleys']) > 0:
                        ax1.scatter(time[t['valleys']], t['distances'][t['valleys']], 
                                  color='green', s=30, label='Valleys')
                    
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Thumb-Index Distance')
                    ax1.set_title(f'{hand.capitalize()} Hand - Tapping Pattern')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                
                # Amplitude progression
                ax2 = axes[i, 1]
                if len(t['tap_amplitudes']) > 0:
                    tap_nums = np.arange(len(t['tap_amplitudes']))
                    ax2.scatter(tap_nums, t['tap_amplitudes'], s=40, alpha=0.7)
                    
                    # Trend line
                    if len(tap_nums) > 1:
                        z = np.polyfit(tap_nums, t['tap_amplitudes'], 1)
                        p = np.poly1d(z)
                        ax2.plot(tap_nums, p(tap_nums), 'r--', 
                                label=f'Decrement: {t["amplitude_decrement"]:.1f}%')
                    
                    ax2.set_xlabel('Tap Number')
                    ax2.set_ylabel('Amplitude')
                    ax2.set_title(f'{hand.capitalize()} Hand - Amplitude Progression')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Finger Tapping Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def process_for_comparison(self, video_path, session_name, detection_confidence=0.3, enable_face=False, enhance_contrast=False, progress=gr.Progress()):
        """
        Process video for comparative analysis, storing results in session
        """
        if video_path is None:
            return None, None, f"Please upload a video for {session_name}"
        
        try:
            # Process video using existing method
            output_video, csv_path, summary = self.process_video(
                video_path, detection_confidence, enable_face, enhance_contrast, progress
            )
            
            if csv_path:
                # Store session data for comparison
                self.comparison_sessions[session_name] = {
                    'video': output_video,
                    'csv': csv_path,
                    'summary': summary,
                    'pd_analysis': None  # Will be populated during analysis
                }
            
            return output_video, csv_path, summary
            
        except Exception as e:
            return None, None, f"Error processing {session_name}: {str(e)}"
    
    def compare_videos(self, progress=gr.Progress()):
        """
        Perform comparative analysis between two processed videos
        """
        if 'video1' not in self.comparison_sessions or 'video2' not in self.comparison_sessions:
            return "Please process both videos first", None, None, None, None
        
        try:
            progress(0, desc="Loading analysis data...")
            
            # Analyze both videos
            results = {}
            for name in ['video1', 'video2']:
                session = self.comparison_sessions[name]
                if session['csv']:
                    analyzer = PDMovementAnalyzer(session['csv'])
                    
                    # Perform all analyses
                    tremor = analyzer.calculate_tremor_frequency('both')
                    tapping = analyzer.analyze_finger_tapping('both')
                    bradykinesia = analyzer.detect_bradykinesia('both')
                    asymmetry = analyzer.calculate_movement_asymmetry()
                    
                    results[name] = {
                        'tremor': tremor,
                        'tapping': tapping,
                        'bradykinesia': bradykinesia,
                        'asymmetry': asymmetry,
                        'analyzer': analyzer
                    }
                    
                    session['pd_analysis'] = results[name]
            
            progress(0.3, desc="Generating comparison report...")
            comparison_report = self.generate_comparison_report(results)
            
            progress(0.5, desc="Creating comparison visualizations...")
            tremor_comparison = self.create_comparative_tremor_plot(results)
            
            progress(0.6, desc="Creating metric comparison...")
            metrics_comparison = self.create_comparative_metrics_plot(results)
            
            progress(0.8, desc="Creating overlay visualizations...")
            overlay_plot = self.create_overlay_timeseries_plot(results)
            
            progress(0.9, desc="Creating similarity heatmap...")
            similarity_plot = self.create_similarity_heatmap(results)
            
            progress(1.0, desc="Comparison complete!")
            
            return comparison_report, tremor_comparison, metrics_comparison, overlay_plot, similarity_plot
            
        except Exception as e:
            return f"Error in comparative analysis: {str(e)}", None, None, None, None
    
    def generate_comparison_report(self, results):
        """
        Generate detailed comparison report between two videos
        """
        v1 = results['video1']
        v2 = results['video2']
        
        # Calculate similarity scores
        tremor_similarity = self.calculate_tremor_similarity(v1['tremor'], v2['tremor'])
        brady_similarity = self.calculate_bradykinesia_similarity(v1['bradykinesia'], v2['bradykinesia'])
        
        html = """
        <style>
            .comparison-container { font-family: 'Inter', sans-serif; }
            .comparison-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;
            }
            .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
            .video-card {
                background: white; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .video-title { font-size: 1.2rem; font-weight: 600; color: #2d3748; margin-bottom: 1rem; }
            .metric-item { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #f0f0f0; }
            .metric-label { color: #718096; }
            .metric-value { font-weight: 600; }
            .similarity-section {
                background: #f7fafc; border-radius: 8px; padding: 1.5rem; margin-top: 1.5rem;
                border: 1px solid #cbd5e0;
            }
            .similarity-score {
                display: inline-block; padding: 0.5rem 1rem; border-radius: 20px;
                font-weight: 600; margin: 0.5rem;
            }
            .high-similarity { background: #c6f6d5; color: #22543d; }
            .medium-similarity { background: #fefcbf; color: #744210; }
            .low-similarity { background: #feb2b2; color: #742a2a; }
            .difference-highlight { background: #fffaf0; padding: 1rem; border-left: 4px solid #ed8936; margin: 1rem 0; }
        </style>
        
        <div class="comparison-container">
            <div class="comparison-header">
                <h2 style="margin: 0;">🔄 Comparative Analysis Report</h2>
                <p style="margin-top: 0.5rem; opacity: 0.9;">Side-by-side comparison of movement patterns</p>
            </div>
            
            <div class="comparison-grid">
                <div class="video-card">
                    <div class="video-title">📹 Video 1</div>
        """
        
        # Add metrics for video 1
        for hand in ['left', 'right']:
            if hand in v1['tremor']:
                html += f"""
                    <div class="metric-item">
                        <span class="metric-label">{hand.capitalize()} Tremor:</span>
                        <span class="metric-value">{v1['tremor'][hand]['peak_frequency']:.2f} Hz</span>
                    </div>
                """
            if hand in v1['bradykinesia']:
                html += f"""
                    <div class="metric-item">
                        <span class="metric-label">{hand.capitalize()} Bradykinesia:</span>
                        <span class="metric-value">{v1['bradykinesia'][hand]['severity']}</span>
                    </div>
                """
        
        html += """
                </div>
                <div class="video-card">
                    <div class="video-title">📹 Video 2</div>
        """
        
        # Add metrics for video 2
        for hand in ['left', 'right']:
            if hand in v2['tremor']:
                html += f"""
                    <div class="metric-item">
                        <span class="metric-label">{hand.capitalize()} Tremor:</span>
                        <span class="metric-value">{v2['tremor'][hand]['peak_frequency']:.2f} Hz</span>
                    </div>
                """
            if hand in v2['bradykinesia']:
                html += f"""
                    <div class="metric-item">
                        <span class="metric-label">{hand.capitalize()} Bradykinesia:</span>
                        <span class="metric-value">{v2['bradykinesia'][hand]['severity']}</span>
                    </div>
                """
        
        html += f"""
                </div>
            </div>
            
            <div class="similarity-section">
                <h3 style="margin-top: 0;">📊 Similarity Analysis</h3>
                <div>
                    <span class="similarity-score {self.get_similarity_class(tremor_similarity)}">
                        Tremor Similarity: {tremor_similarity:.1%}
                    </span>
                    <span class="similarity-score {self.get_similarity_class(brady_similarity)}">
                        Bradykinesia Similarity: {brady_similarity:.1%}
                    </span>
                </div>
            </div>
        """
        
        # Add key differences
        differences = self.identify_key_differences(v1, v2)
        if differences:
            html += '<div class="difference-highlight"><h4 style="margin-top: 0;">⚠️ Key Differences</h4><ul>'
            for diff in differences:
                html += f'<li>{diff}</li>'
            html += '</ul></div>'
        
        html += '</div>'
        return html
    
    def calculate_tremor_similarity(self, tremor1, tremor2):
        """Calculate similarity between tremor patterns"""
        similarities = []
        for hand in ['left', 'right']:
            if hand in tremor1 and hand in tremor2:
                freq_diff = abs(tremor1[hand]['peak_frequency'] - tremor2[hand]['peak_frequency'])
                # Convert frequency difference to similarity (0-1)
                similarity = max(0, 1 - freq_diff / 10)  # 10 Hz max difference
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0
    
    def calculate_bradykinesia_similarity(self, brady1, brady2):
        """Calculate similarity between bradykinesia scores"""
        severity_map = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        similarities = []
        
        for hand in ['left', 'right']:
            if hand in brady1 and hand in brady2:
                score1 = severity_map.get(brady1[hand]['severity'], 0)
                score2 = severity_map.get(brady2[hand]['severity'], 0)
                diff = abs(score1 - score2)
                similarity = 1 - (diff / 3)  # Max difference is 3
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0
    
    def get_similarity_class(self, score):
        """Get CSS class based on similarity score"""
        if score >= 0.8:
            return 'high-similarity'
        elif score >= 0.5:
            return 'medium-similarity'
        else:
            return 'low-similarity'
    
    def identify_key_differences(self, v1, v2):
        """Identify and list key differences between analyses"""
        differences = []
        
        for hand in ['left', 'right']:
            # Check tremor differences
            if hand in v1['tremor'] and hand in v2['tremor']:
                freq_diff = abs(v1['tremor'][hand]['peak_frequency'] - v2['tremor'][hand]['peak_frequency'])
                if freq_diff > 2:  # Significant if > 2 Hz
                    differences.append(f"{hand.capitalize()} hand tremor frequency differs by {freq_diff:.1f} Hz")
            
            # Check bradykinesia differences
            if hand in v1['bradykinesia'] and hand in v2['bradykinesia']:
                if v1['bradykinesia'][hand]['severity'] != v2['bradykinesia'][hand]['severity']:
                    differences.append(
                        f"{hand.capitalize()} hand bradykinesia: "
                        f"Video 1 = {v1['bradykinesia'][hand]['severity']}, "
                        f"Video 2 = {v2['bradykinesia'][hand]['severity']}"
                    )
        
        return differences
    
    def create_comparative_tremor_plot(self, results):
        """Create side-by-side tremor frequency comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        videos = ['video1', 'video2']
        colors = ['blue', 'red']
        
        for row, hand in enumerate(['left', 'right']):
            for col, video_name in enumerate(videos):
                ax = axes[row, col]
                
                if hand in results[video_name]['tremor']:
                    tremor = results[video_name]['tremor'][hand]
                    
                    # Plot PSD
                    ax.plot(tremor['frequencies'], tremor['psd'], 
                           color=colors[col], linewidth=2, alpha=0.8)
                    ax.fill_between(tremor['frequencies'], 0, tremor['psd'], 
                                   color=colors[col], alpha=0.2)
                    
                    # Mark peak
                    ax.axvline(tremor['peak_frequency'], color=colors[col], 
                             linestyle='--', linewidth=2,
                             label=f'Peak: {tremor["peak_frequency"]:.2f} Hz')
                    
                    # PD ranges
                    ax.axvspan(4, 6, alpha=0.1, color='red')
                    ax.axvspan(6, 12, alpha=0.1, color='orange')
                    
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power Spectral Density')
                    ax.set_title(f'{video_name.capitalize()} - {hand.capitalize()} Hand')
                    ax.set_xlim([0, 15])
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comparative Tremor Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_comparative_metrics_plot(self, results):
        """Create bar chart comparing key metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Prepare data
        metrics = {
            'Tremor Frequency (Hz)': {},
            'Bradykinesia Score': {},
            'Tapping Frequency (Hz)': {}
        }
        
        for video_name in ['video1', 'video2']:
            v = results[video_name]
            
            # Collect tremor frequencies
            for hand in ['left', 'right']:
                if hand in v['tremor']:
                    key = f"{video_name}_{hand}"
                    metrics['Tremor Frequency (Hz)'][key] = v['tremor'][hand]['peak_frequency']
                
                if hand in v['bradykinesia']:
                    key = f"{video_name}_{hand}"
                    severity_map = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
                    metrics['Bradykinesia Score'][key] = severity_map.get(
                        v['bradykinesia'][hand]['severity'], 0
                    )
                
                if hand in v['tapping']:
                    key = f"{video_name}_{hand}"
                    metrics['Tapping Frequency (Hz)'][key] = v['tapping'][hand]['tap_frequency']
        
        # Create bar plots
        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = axes[idx]
            
            if metric_data:
                labels = list(metric_data.keys())
                values = list(metric_data.values())
                
                # Color code by video
                colors = ['#667eea' if 'video1' in l else '#764ba2' for l in labels]
                
                bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels([l.replace('_', '\n').title() for l in labels], rotation=0)
                ax.set_ylabel(metric_name)
                ax.set_title(metric_name)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}' if isinstance(val, float) else str(val),
                           ha='center', va='bottom')
        
        plt.suptitle('Comparative Metrics Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_overlay_timeseries_plot(self, results):
        """Create overlaid time series comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {'video1': 'blue', 'video2': 'red'}
        
        for col, hand in enumerate(['left', 'right']):
            for row, landmark_id in enumerate([8, 4]):  # Index tip and thumb tip
                ax = axes[row, col]
                
                for video_name, color in colors.items():
                    analyzer = results[video_name]['analyzer']
                    df = analyzer.df
                    
                    # Filter for specific hand and landmark
                    hand_data = df[(df['label'] == hand.capitalize()) & 
                                  (df['landmark_id'] == landmark_id)]
                    
                    if not hand_data.empty:
                        # Plot movement
                        ax.plot(hand_data['timestamp'], hand_data['y'], 
                               label=video_name.capitalize(), color=color, 
                               alpha=0.7, linewidth=1.5)
                
                landmark_name = 'Index Tip' if landmark_id == 8 else 'Thumb Tip'
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Y Position (normalized)')
                ax.set_title(f'{hand.capitalize()} Hand - {landmark_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Movement Pattern Overlay Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_similarity_heatmap(self, results):
        """Create heatmap showing similarity between various metrics"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate similarity matrix
        metrics = ['L Tremor', 'R Tremor', 'L Brady', 'R Brady', 'L Tapping', 'R Tapping']
        similarity_matrix = np.zeros((len(metrics), len(metrics)))
        
        v1 = results['video1']
        v2 = results['video2']
        
        # Helper function to calculate metric similarity
        def calc_similarity(val1, val2, max_diff):
            if val1 is None or val2 is None:
                return 0
            diff = abs(val1 - val2)
            return max(0, 1 - diff / max_diff)
        
        # Fill similarity matrix
        similarities = []
        
        for hand in ['left', 'right']:
            prefix = 'L' if hand == 'left' else 'R'
            
            # Tremor similarity
            if hand in v1['tremor'] and hand in v2['tremor']:
                sim = calc_similarity(
                    v1['tremor'][hand]['peak_frequency'],
                    v2['tremor'][hand]['peak_frequency'],
                    10
                )
                similarities.append((f"{prefix} Tremor", sim))
            
            # Bradykinesia similarity
            if hand in v1['bradykinesia'] and hand in v2['bradykinesia']:
                severity_map = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
                sim = calc_similarity(
                    severity_map.get(v1['bradykinesia'][hand]['severity'], 0),
                    severity_map.get(v2['bradykinesia'][hand]['severity'], 0),
                    3
                )
                similarities.append((f"{prefix} Brady", sim))
            
            # Tapping similarity
            if hand in v1['tapping'] and hand in v2['tapping']:
                sim = calc_similarity(
                    v1['tapping'][hand]['tap_frequency'],
                    v2['tapping'][hand]['tap_frequency'],
                    5
                )
                similarities.append((f"{prefix} Tapping", sim))
        
        # Create bar plot instead of heatmap for clearer visualization
        if similarities:
            labels, values = zip(*similarities)
            colors_list = ['#48bb78' if v >= 0.8 else '#ed8936' if v >= 0.5 else '#f56565' 
                          for v in values]
            
            bars = ax.barh(range(len(labels)), values, color=colors_list, alpha=0.8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Similarity Score')
            ax.set_xlim([0, 1])
            ax.set_title('Metric Similarity Comparison', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{val:.1%}', ha='left', va='center', fontweight='bold')
            
            # Add reference lines
            ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='High (>80%)')
            ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (>50%)')
            ax.legend(loc='lower right')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return fig
    
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# Create Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    app = PDAnalysisApp()
    
    # Custom CSS for the entire interface
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b4199 100%) !important;
    }
    """
    
    with gr.Blocks(
        title="PD Movement Analysis",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            font=["Inter", "system-ui", "sans-serif"]
        ),
        css=custom_css
    ) as interface:
        gr.Markdown("""
        # 🏥 Parkinson's Disease Movement Analysis System
        
        Upload a video to analyze hand movements for Parkinson's Disease biomarkers including tremor, 
        bradykinesia, and finger tapping patterns.
        """)
        
        with gr.Tabs():
            # Tab 1: Video Processing
            with gr.TabItem("📹 Video Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video")
                        
                        with gr.Accordion("⚙️ Processing Settings", open=True):
                            confidence_slider = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.1,
                                label="Detection Confidence",
                                info="Lower values detect more landmarks but may be less accurate"
                            )
                            
                            face_checkbox = gr.Checkbox(
                                label="Include Face Detection",
                                value=False,
                                info="Enable to also track facial landmarks (slower processing)"
                            )
                            
                            contrast_checkbox = gr.Checkbox(
                                label="Enhance Contrast",
                                value=False,
                                info="Apply contrast enhancement for better detection in poor lighting"
                            )
                        
                        process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        output_video = gr.Video(label="Tracked Video")
                        tracking_summary = gr.HTML(label="Tracking Summary")
                        csv_output = gr.File(label="Download Landmarks CSV", visible=False)
            
            # Tab 2: PD Analysis
            with gr.TabItem("🔬 PD Analysis"):
                analyze_btn = gr.Button("🧪 Analyze Movement Patterns", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        analysis_report = gr.HTML(label="Analysis Report")
                    
                with gr.Row():
                    tremor_plot = gr.Plot(label="Tremor Frequency Analysis")
                    tapping_plot = gr.Plot(label="Finger Tapping Patterns")
                
                dashboard_plot = gr.Plot(label="Comprehensive Dashboard")
            
            # Tab 3: Comparative Analysis
            with gr.TabItem("🔄 Comparative Analysis"):
                gr.Markdown("""
                ### Compare Two Videos Side-by-Side
                Upload and process two videos to compare movement patterns, tremor characteristics, and bradykinesia scores.
                """)
                
                with gr.Row():
                    # Video 1 column
                    with gr.Column():
                        gr.Markdown("#### 📹 Video 1")
                        video1_input = gr.Video(label="Upload First Video")
                        
                        with gr.Accordion("Video 1 Settings", open=False):
                            video1_confidence = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.1,
                                label="Detection Confidence"
                            )
                            video1_face = gr.Checkbox(label="Include Face", value=False)
                            video1_contrast = gr.Checkbox(label="Enhance Contrast", value=False)
                        
                        process_video1_btn = gr.Button("Process Video 1", variant="secondary")
                        video1_output = gr.Video(label="Tracked Video 1", visible=False)
                        video1_summary = gr.HTML(label="Video 1 Summary")
                    
                    # Video 2 column
                    with gr.Column():
                        gr.Markdown("#### 📹 Video 2")
                        video2_input = gr.Video(label="Upload Second Video")
                        
                        with gr.Accordion("Video 2 Settings", open=False):
                            video2_confidence = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.1,
                                label="Detection Confidence"
                            )
                            video2_face = gr.Checkbox(label="Include Face", value=False)
                            video2_contrast = gr.Checkbox(label="Enhance Contrast", value=False)
                        
                        process_video2_btn = gr.Button("Process Video 2", variant="secondary")
                        video2_output = gr.Video(label="Tracked Video 2", visible=False)
                        video2_summary = gr.HTML(label="Video 2 Summary")
                
                # Compare button
                with gr.Row():
                    compare_btn = gr.Button(
                        "🔬 Compare Videos", 
                        variant="primary", 
                        size="lg",
                        interactive=False  # Initially disabled
                    )
                
                # Comparison results
                with gr.Row():
                    comparison_report = gr.HTML(label="Comparison Report")
                
                with gr.Row():
                    tremor_comparison_plot = gr.Plot(label="Tremor Frequency Comparison")
                    metrics_comparison_plot = gr.Plot(label="Key Metrics Comparison")
                
                with gr.Row():
                    overlay_plot = gr.Plot(label="Movement Pattern Overlay")
                    similarity_plot = gr.Plot(label="Similarity Analysis")
                
                # Hidden state components to track processing
                video1_processed = gr.State(False)
                video2_processed = gr.State(False)
            
            # Tab 4: Video Annotation
            with gr.TabItem("🎥 Video Annotation"):
                gr.Markdown("""
                ### Clinical Video Segmentation Tool
                Annotate video segments with standardized PD assessment tasks for precise analysis.
                
                **Two Options:**
                1. Use the manual time entry below (with time helper)
                2. Click "Launch Advanced Annotator" for automatic time capture
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Standard Gradio video player
                        annotation_video = gr.Video(
                            label="Video to Annotate - Note the time shown in player (e.g., 0:13)",
                            interactive=False
                        )
                        
                        # Button to launch advanced annotator
                        launch_annotator_btn = gr.Button(
                            "🚀 Launch Advanced Annotator (Opens in New Window)",
                            variant="primary",
                            size="lg"
                        )
                        
                        def launch_annotation_server():
                            if app.current_video_path:
                                import subprocess
                                import threading
                                import os
                                
                                # Check which server file exists - prioritize video editor style
                                video_editor_server = "video_editor_annotation_server.py"
                                enhanced_server = "enhanced_annotation_server.py"
                                basic_server = "video_annotation_server.py"
                                
                                if os.path.exists(video_editor_server):
                                    server_file = video_editor_server
                                elif os.path.exists(enhanced_server):
                                    server_file = enhanced_server
                                else:
                                    server_file = basic_server
                                
                                def run_server():
                                    subprocess.run([
                                        "python", 
                                        server_file,
                                        app.current_video_path
                                    ])
                                
                                thread = threading.Thread(target=run_server)
                                thread.daemon = True
                                thread.start()
                                
                                if server_file == video_editor_server:
                                    features_list = """
                                    <ul>
                                        <li>🎬 Professional video editor-style interface</li>
                                        <li>📐 Multi-track timeline (4 tracks)</li>
                                        <li>✋ Drag & drop timeline segments</li>
                                        <li>↔️ Resize segments by dragging edges</li>
                                        <li>✂️ Split, cut, copy, paste operations</li>
                                        <li>🎨 Color-coded by task type</li>
                                        <li>🔍 Zoom in/out timeline</li>
                                        <li>📝 Properties inspector panel</li>
                                        <li>⌨️ Pro keyboard shortcuts</li>
                                        <li>🎯 Playhead with timecode display</li>
                                    </ul>
                                    """
                                    server_name = "Video Editor"
                                elif server_file == enhanced_server:
                                    features_list = """
                                    <ul>
                                        <li>🔥 Movement heatmap visualization</li>
                                        <li>🏷️ Color-coded task labels</li>
                                        <li>✏️ Edit existing annotations</li>
                                        <li>📊 Real-time statistics</li>
                                        <li>🎯 Timeline visualization</li>
                                        <li>⌨️ Enhanced keyboard shortcuts</li>
                                        <li>🔍 Search and filter annotations</li>
                                        <li>📥 Import/Export (JSON & CSV)</li>
                                    </ul>
                                    """
                                    server_name = "Enhanced"
                                else:
                                    features_list = """
                                    <ul>
                                        <li>Real-time video time capture</li>
                                        <li>Keyboard shortcuts (C, S, E, Q)</li>
                                        <li>Automatic time display</li>
                                        <li>Export to JSON</li>
                                    </ul>
                                    """
                                    server_name = "Basic"
                                
                                return f"""
                                <div style='padding: 20px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
                                    <h3>✅ {server_name} Annotation Server Started!</h3>
                                    <p>Open <a href='http://localhost:5555' target='_blank' style='color: white; text-decoration: underline; font-weight: bold;'>
                                    http://localhost:5555</a> in a new browser tab.</p>
                                    <p>The {server_name} annotator provides:</p>
                                    {features_list}
                                </div>
                                """
                            else:
                                return "<div style='color: red;'>Please load a video first!</div>"
                        
                        server_status = gr.HTML()
                        
                        launch_annotator_btn.click(
                            fn=launch_annotation_server,
                            outputs=[server_status]
                        )
                        
                        # Timeline visualization
                        timeline_plot = gr.Plot(
                            label="Annotation Timeline",
                            elem_id="timeline-plot"
                        )
                        
                    with gr.Column(scale=1):
                        # Annotation controls
                        gr.Markdown("#### Add New Segment")
                        
                        # Time conversion helper
                        gr.Markdown("### ⏱️ Time Helper")
                        with gr.Row():
                            time_minutes = gr.Number(label="Minutes", value=0, precision=0, scale=1)
                            time_seconds = gr.Number(label="Seconds", value=0, precision=0, scale=1)
                            time_total = gr.Number(label="= Total Seconds", value=0, precision=1, interactive=False, scale=2)
                        
                        def convert_time(mins, secs):
                            return mins * 60 + secs
                        
                        time_minutes.change(convert_time, [time_minutes, time_seconds], time_total)
                        time_seconds.change(convert_time, [time_minutes, time_seconds], time_total)
                        
                        gr.Markdown("---")
                        
                        with gr.Row():
                            segment_start = gr.Number(
                                label="Start Time (sec)",
                                value=0,
                                precision=1
                            )
                            segment_end = gr.Number(
                                label="End Time (sec)",
                                value=5,
                                precision=1
                            )
                        
                        with gr.Row():
                            use_total_as_start = gr.Button(
                                "⬅️ Use Helper → Start",
                                variant="secondary",
                                size="sm"
                            )
                            use_total_as_end = gr.Button(
                                "➡️ Use Helper → End", 
                                variant="secondary",
                                size="sm"
                            )
                        
                        def set_start_from_total(total):
                            return total
                        
                        def set_end_from_total(total):
                            return total
                        
                        use_total_as_start.click(set_start_from_total, [time_total], [segment_start])
                        use_total_as_end.click(set_end_from_total, [time_total], [segment_end])
                        
                        # Task selection with categories
                        task_category = gr.Dropdown(
                            label="Task Category",
                            choices=[
                                "Rest Assessment",
                                "Tremor Tests",
                                "Bradykinesia Tests",
                                "Gait & Balance",
                                "Facial Assessment",
                                "Complex Tasks",
                                "Other"
                            ],
                            value="Bradykinesia Tests"
                        )
                        
                        task_type = gr.Dropdown(
                            label="Specific Task",
                            choices=[t.value for t in PDTaskType],
                            value=PDTaskType.FINGER_TAPPING.value
                        )
                        
                        task_side = gr.Radio(
                            label="Side",
                            choices=["left", "right", "bilateral", "n/a"],
                            value="bilateral"
                        )
                        
                        severity_score = gr.Slider(
                            label="Severity (UPDRS 0-4)",
                            minimum=0,
                            maximum=4,
                            step=1,
                            value=0,
                            info="0=Normal, 4=Severe"
                        )
                        
                        segment_notes = gr.Textbox(
                            label="Notes",
                            placeholder="Optional notes about this segment",
                            lines=2
                        )
                        
                        annotator_name = gr.Textbox(
                            label="Annotator Name",
                            placeholder="Your name",
                            value=""
                        )
                        
                        with gr.Row():
                            add_segment_btn = gr.Button(
                                "➕ Add Segment",
                                variant="primary"
                            )
                            clear_segment_btn = gr.Button(
                                "🗑️ Clear Form",
                                variant="secondary"
                            )
                
                # Segment list and management
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Current Annotations")
                        segments_table = gr.Dataframe(
                            headers=["ID", "Start", "End", "Task", "Side", "Severity", "Notes"],
                            label="Segments",
                            interactive=False
                        )
                        
                        with gr.Row():
                            selected_segment_id = gr.Number(
                                label="Segment ID to Edit/Delete",
                                value=0,
                                precision=0
                            )
                            delete_segment_btn = gr.Button(
                                "🗑️ Delete Segment",
                                variant="stop"
                            )
                    
                    with gr.Column():
                        # Quick stats
                        annotation_stats = gr.HTML(
                            label="Annotation Statistics",
                            value="<p>No annotations yet</p>"
                        )
                        
                        # File operations
                        gr.Markdown("#### Save/Load Annotations")
                        
                        with gr.Row():
                            save_annotations_btn = gr.Button(
                                "💾 Save Annotations",
                                variant="primary"
                            )
                            load_annotations_btn = gr.UploadButton(
                                "📂 Load Annotations",
                                file_types=[".json"],
                                variant="secondary"
                            )
                        
                        annotation_file = gr.File(
                            label="Download Annotation File",
                            visible=False
                        )
                        
                        # Export for analysis
                        export_for_analysis_btn = gr.Button(
                            "📊 Export for Analysis",
                            variant="primary"
                        )
                        
                        analysis_csv = gr.File(
                            label="Analysis-Ready CSV",
                            visible=False
                        )
                
                # Task filter for focused analysis
                with gr.Accordion("🎯 Task-Specific Analysis", open=False):
                    gr.Markdown("""
                    Process only specific annotated segments for targeted analysis.
                    """)
                    
                    task_filter = gr.CheckboxGroup(
                        label="Select Tasks to Analyze",
                        choices=[
                            PDTaskType.FINGER_TAPPING.value,
                            PDTaskType.REST_TREMOR.value,
                            PDTaskType.POSTURAL_TREMOR.value,
                            PDTaskType.HAND_OPENING_CLOSING.value,
                            PDTaskType.PRONATION_SUPINATION.value
                        ],
                        value=[PDTaskType.FINGER_TAPPING.value]
                    )
                    
                    filtered_analysis_btn = gr.Button(
                        "🔬 Run Filtered Analysis",
                        variant="primary"
                    )
                    
                    filtered_results = gr.HTML(
                        label="Filtered Analysis Results"
                    )
            
            # Tab 5: Instructions
            with gr.TabItem("📖 Instructions"):
                gr.Markdown("""
                ## How to Use This System
                
                ### 1. Video Requirements
                - **Format:** MP4, AVI, or MOV
                - **Quality:** At least 720p recommended
                - **Lighting:** Good lighting improves detection accuracy
                - **Position:** Hands should be clearly visible in frame
                - **Duration:** 10-60 seconds recommended for analysis
                
                ### 2. Recording Guidelines for PD Assessment
                
                #### Finger Tapping Task:
                1. Sit comfortably with hands visible
                2. Tap index finger and thumb together repeatedly
                3. Make the movement as large and fast as possible
                4. Continue for 10-15 seconds per hand
                5. Test both hands separately or together
                
                #### Rest Tremor Assessment:
                1. Rest hands on lap or armrests
                2. Relax completely for 10-20 seconds
                3. Avoid voluntary movements
                
                #### Action Tremor Assessment:
                1. Extend arms forward
                2. Hold position for 10-15 seconds
                3. Or perform finger-to-nose task
                
                ### 3. Understanding Results
                
                #### Tremor Analysis:
                - **4-6 Hz:** Typical PD resting tremor range
                - **6-12 Hz:** PD action tremor or essential tremor
                - **Amplitude:** Higher values indicate more pronounced tremor
                
                #### Bradykinesia Indicators:
                - **Slow tapping:** Frequency < 3 Hz
                - **Amplitude decrement:** Progressive reduction in movement size
                - **Irregular rhythm:** Inconsistent tapping intervals
                - **Hesitations:** Pauses or freezing during movement
                
                #### Severity Levels:
                - ✅ **None:** No bradykinesia detected
                - ⚠️ **Mild:** 1-2 indicators present
                - 🟠 **Moderate:** 3 indicators present
                - 🔴 **Severe:** 4+ indicators present
                
                ### ⚠️ Important Notes
                - This system is for research and educational purposes only
                - Results should not be used for clinical diagnosis
                - Always consult healthcare professionals for medical advice
                - Multiple assessments recommended for consistency
                """)
        
        # Event handlers
        process_btn.click(
            fn=app.process_video,
            inputs=[video_input, confidence_slider, face_checkbox, contrast_checkbox],
            outputs=[output_video, csv_output, tracking_summary]
        ).then(
            fn=lambda x: gr.update(visible=True, value=x),
            inputs=[csv_output],
            outputs=[csv_output]
        )
        
        analyze_btn.click(
            fn=app.analyze_pd_movements,
            inputs=[csv_output],
            outputs=[analysis_report, tremor_plot, tapping_plot, dashboard_plot]
        )
        
        # Comparative analysis event handlers
        def process_video1_wrapper(video, conf, face, contrast):
            output_video, csv, summary = app.process_for_comparison(
                video, 'video1', conf, face, contrast
            )
            return (
                output_video,
                gr.update(visible=True, value=output_video) if output_video else gr.update(),
                summary,
                True if csv else False  # Update processed state
            )
        
        def process_video2_wrapper(video, conf, face, contrast):
            output_video, csv, summary = app.process_for_comparison(
                video, 'video2', conf, face, contrast
            )
            return (
                output_video,
                gr.update(visible=True, value=output_video) if output_video else gr.update(),
                summary,
                True if csv else False  # Update processed state
            )
        
        def update_compare_button(v1_processed, v2_processed):
            """Enable compare button when both videos are processed"""
            if v1_processed and v2_processed:
                return gr.update(interactive=True, variant="primary")
            else:
                return gr.update(interactive=False, variant="secondary")
        
        # Process video 1
        process_video1_btn.click(
            fn=process_video1_wrapper,
            inputs=[video1_input, video1_confidence, video1_face, video1_contrast],
            outputs=[video1_output, video1_output, video1_summary, video1_processed]
        ).then(
            fn=update_compare_button,
            inputs=[video1_processed, video2_processed],
            outputs=[compare_btn]
        )
        
        # Process video 2
        process_video2_btn.click(
            fn=process_video2_wrapper,
            inputs=[video2_input, video2_confidence, video2_face, video2_contrast],
            outputs=[video2_output, video2_output, video2_summary, video2_processed]
        ).then(
            fn=update_compare_button,
            inputs=[video1_processed, video2_processed],
            outputs=[compare_btn]
        )
        
        # Compare videos
        compare_btn.click(
            fn=app.compare_videos,
            inputs=[],
            outputs=[
                comparison_report,
                tremor_comparison_plot,
                metrics_comparison_plot,
                overlay_plot,
                similarity_plot
            ]
        )
        
        # Annotation event handlers
        def load_video_for_annotation(video_file):
            """Load video and initialize annotation manager"""
            if video_file is None:
                return None, None, "<p>Please upload a video first</p>", None
            
            app.current_video_path = video_file
            app.annotation_manager = VideoAnnotationManager(video_file)
            
            # Get video duration
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            app.current_video_duration = duration
            app.annotation_manager.metadata['total_duration'] = duration
            
            return (
                video_file,  # Return the video path for Gradio's video component
                create_timeline_plot(app.annotation_manager),
                f"<p>Video loaded. Duration: {duration:.1f} seconds. Ready for annotation.</p>",
                update_segments_table(app.annotation_manager)
            )
        
        # Store current video state
        app.video_current_time = 0
        
        def capture_video_time(evt: gr.EventData):
            """Capture current time using JavaScript execution"""
            # This uses Gradio's ability to execute JavaScript and return value
            js_code = """
            () => {
                const videos = document.querySelectorAll('video');
                for (let video of videos) {
                    if (!video.paused || video.currentTime > 0) {
                        return video.currentTime;
                    }
                }
                return 0;
            }
            """
            return gr.update()
        
        def get_current_time_js():
            """JavaScript to get current video time"""
            return """
            () => {
                const videos = document.querySelectorAll('video');
                for (let video of videos) {
                    if (video && video.src) {
                        return video.currentTime;
                    }
                }
                return 0;
            }
            """
        
        def use_current_as_start(current_time_value):
            """Use current video time as start time"""
            if isinstance(current_time_value, (int, float)):
                return float(current_time_value)
            return 0
        
        def use_current_as_end(current_time_value):
            """Use current video time as end time"""
            if isinstance(current_time_value, (int, float)):
                return float(current_time_value)
            return 5
        
        def create_timeline_plot(manager):
            """Create interactive timeline visualization"""
            if not manager:
                return None
            
            timeline_data = manager.get_timeline_data()
            
            fig = go.Figure()
            
            # Add segments as horizontal bars
            for segment in timeline_data['segments']:
                fig.add_trace(go.Scatter(
                    x=[segment['start'], segment['end'], segment['end'], segment['start'], segment['start']],
                    y=[0, 0, 1, 1, 0],
                    fill='toself',
                    fillcolor=segment['color'],
                    line=dict(color=segment['color']),
                    name=segment['label'],
                    text=segment['task'],
                    hovertemplate=(
                        f"<b>{segment['task']}</b><br>"
                        f"Time: {segment['start']:.1f}s - {segment['end']:.1f}s<br>"
                        f"Side: {segment['side']}<br>"
                        f"Notes: {segment['notes']}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title="Annotation Timeline",
                xaxis_title="Time (seconds)",
                yaxis=dict(visible=False, range=[-0.1, 1.1]),
                xaxis=dict(range=[0, timeline_data.get('total_duration', 60)]),
                height=200,
                hovermode='x unified',
                plot_bgcolor='#f8f9fa'
            )
            
            return fig
        
        def add_segment(start, end, task, side, severity, notes, annotator):
            """Add a new annotation segment"""
            if not app.annotation_manager:
                return None, None, "<p>Please load a video first</p>"
            
            try:
                segment = app.annotation_manager.add_segment(
                    start=start,
                    end=end,
                    task=task,
                    side=side,
                    severity=severity if severity > 0 else None,
                    notes=notes,
                    annotator=annotator
                )
                
                return (
                    create_timeline_plot(app.annotation_manager),
                    update_segments_table(app.annotation_manager),
                    generate_stats_html(app.annotation_manager)
                )
            except Exception as e:
                return None, None, f"<p style='color: red;'>Error: {str(e)}</p>"
        
        def delete_segment(segment_id):
            """Delete a segment by ID"""
            if not app.annotation_manager:
                return None, None, "<p>No annotations loaded</p>"
            
            try:
                app.annotation_manager.remove_segment(int(segment_id))
                return (
                    create_timeline_plot(app.annotation_manager),
                    update_segments_table(app.annotation_manager),
                    generate_stats_html(app.annotation_manager)
                )
            except Exception as e:
                return None, None, f"<p style='color: red;'>Error: {str(e)}</p>"
        
        def update_segments_table(manager):
            """Update the segments display table"""
            if not manager or not manager.segments:
                return pd.DataFrame(columns=["ID", "Start", "End", "Task", "Side", "Severity", "Notes"])
            
            data = []
            for i, seg in enumerate(manager.segments):
                data.append([
                    i,
                    f"{seg.start_time:.1f}",
                    f"{seg.end_time:.1f}",
                    seg.task_type[:30],  # Truncate long task names
                    seg.side,
                    seg.severity if seg.severity is not None else "N/A",
                    seg.notes[:50] if seg.notes else ""
                ])
            
            return pd.DataFrame(data, columns=["ID", "Start", "End", "Task", "Side", "Severity", "Notes"])
        
        def generate_stats_html(manager):
            """Generate annotation statistics HTML"""
            if not manager or not manager.segments:
                return "<p>No annotations yet</p>"
            
            report = manager.generate_analysis_report()
            
            html = f"""
            <div style="font-family: 'Inter', sans-serif; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <h4 style="margin-top: 0; color: #2d3748;">📊 Statistics</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>🎥 Total Segments: <strong>{report['total_segments']}</strong></div>
                    <div>⏱️ Total Time: <strong>{report['total_annotated_time']:.1f}s</strong></div>
                    <div>📝 Unique Tasks: <strong>{report['unique_tasks']}</strong></div>
                    <div>⌚ Avg Duration: <strong>{report['average_segment_duration']:.1f}s</strong></div>
                </div>
                
                <h5 style="margin-top: 1rem; color: #4a5568;">Task Distribution:</h5>
                <ul style="margin: 0; padding-left: 1.5rem;">
            """
            
            for task, count in sorted(report['task_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
                duration = report['task_durations'][task]
                html += f"<li>{task}: {count} segments ({duration:.1f}s)</li>"
            
            html += """</ul></div>"""
            
            return html
        
        def save_annotations():
            """Save annotations to file"""
            if not app.annotation_manager:
                return None, "<p>No annotations to save</p>"
            
            try:
                filepath = os.path.join(app.temp_dir, "annotations.json")
                app.annotation_manager.save_to_json(filepath)
                return gr.update(visible=True, value=filepath), "<p style='color: green;'>Annotations saved!</p>"
            except Exception as e:
                return None, f"<p style='color: red;'>Error saving: {str(e)}</p>"
        
        def load_annotations(file):
            """Load annotations from file"""
            if not file:
                return None, None, "<p>No file selected</p>"
            
            try:
                if not app.annotation_manager:
                    app.annotation_manager = VideoAnnotationManager()
                
                app.annotation_manager.load_from_json(file.name)
                
                return (
                    create_timeline_plot(app.annotation_manager),
                    update_segments_table(app.annotation_manager),
                    generate_stats_html(app.annotation_manager)
                )
            except Exception as e:
                return None, None, f"<p style='color: red;'>Error loading: {str(e)}</p>"
        
        def export_for_analysis():
            """Export annotations for analysis"""
            if not app.annotation_manager:
                return None
            
            try:
                filepath = os.path.join(app.temp_dir, "annotations_analysis.csv")
                app.annotation_manager.export_to_csv(filepath)
                return gr.update(visible=True, value=filepath)
            except Exception as e:
                return None
        
        def update_task_choices(category):
            """Update task choices based on category"""
            category_tasks = {
                "Rest Assessment": [
                    PDTaskType.REST_TREMOR.value,
                    PDTaskType.REST_HANDS_LAP.value,
                    PDTaskType.REST_HANDS_HANGING.value,
                    PDTaskType.REST_DISTRACTION.value
                ],
                "Tremor Tests": [
                    PDTaskType.POSTURAL_TREMOR.value,
                    PDTaskType.KINETIC_TREMOR.value,
                    PDTaskType.WING_BEATING.value
                ],
                "Bradykinesia Tests": [
                    PDTaskType.FINGER_TAPPING.value,
                    PDTaskType.HAND_OPENING_CLOSING.value,
                    PDTaskType.PRONATION_SUPINATION.value,
                    PDTaskType.TOE_TAPPING.value,
                    PDTaskType.LEG_AGILITY.value
                ],
                "Gait & Balance": [
                    PDTaskType.GAIT_WALKING.value,
                    PDTaskType.GAIT_TURNING.value,
                    PDTaskType.FREEZING_OF_GAIT.value,
                    PDTaskType.POSTURAL_STABILITY.value
                ],
                "Facial Assessment": [
                    PDTaskType.FACIAL_EXPRESSION.value,
                    PDTaskType.SPEECH_ASSESSMENT.value,
                    PDTaskType.GLABELLAR_REFLEX.value
                ],
                "Complex Tasks": [
                    PDTaskType.FINGER_TO_NOSE.value,
                    PDTaskType.RAPID_ALTERNATING.value,
                    PDTaskType.ARISE_FROM_CHAIR.value,
                    PDTaskType.WRITING_SAMPLE.value,
                    PDTaskType.SPIRAL_DRAWING.value
                ],
                "Other": [
                    PDTaskType.PREPARATION.value,
                    PDTaskType.INSTRUCTION.value,
                    PDTaskType.TRANSITION.value,
                    PDTaskType.BREAK.value,
                    PDTaskType.INVALID.value,
                    PDTaskType.OTHER.value
                ]
            }
            
            return gr.update(choices=category_tasks.get(category, []))
        
        def run_filtered_analysis(task_filter):
            """Run analysis on filtered segments"""
            if not app.annotation_manager or not app.current_session:
                return "<p>Please process a video and create annotations first</p>"
            
            try:
                # Load landmarks data
                csv_path = os.path.join(app.current_session, "landmarks.csv")
                if not os.path.exists(csv_path):
                    return "<p>No landmark data found. Please process the video first.</p>"
                
                landmarks_df = pd.read_csv(csv_path)
                
                # Filter by annotated segments
                filtered_df = AnnotationIntegrator.filter_landmarks_by_segments(
                    landmarks_df, app.annotation_manager, task_filter
                )
                
                if filtered_df.empty:
                    return "<p>No data found for selected tasks</p>"
                
                # Generate analysis
                results = AnnotationIntegrator.analyze_by_task(
                    landmarks_df, app.annotation_manager
                )
                
                # Format results
                html = "<div style='font-family: Inter, sans-serif;'>"
                html += "<h3>Filtered Analysis Results</h3>"
                
                for task in task_filter:
                    if task in results:
                        r = results[task]
                        html += f"""
                        <div style='background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 8px;'>
                            <h4>{task}</h4>
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;'>
                                <div>Segments: <strong>{r['num_segments']}</strong></div>
                                <div>Duration: <strong>{r['total_duration']:.1f}s</strong></div>
                                <div>Frames: <strong>{r['total_frames']}</strong></div>
                                <div>Detection: <strong>{r['detection_rate']:.1%}</strong></div>
                            </div>
                        </div>
                        """
                
                html += "</div>"
                return html
                
            except Exception as e:
                return f"<p style='color: red;'>Error in analysis: {str(e)}</p>"
        
        # Wire up annotation event handlers
        video_input.change(
            fn=load_video_for_annotation,
            inputs=[video_input],
            outputs=[annotation_video, timeline_plot, annotation_stats, segments_table]
        )
        
        
        task_category.change(
            fn=update_task_choices,
            inputs=[task_category],
            outputs=[task_type]
        )
        
        add_segment_btn.click(
            fn=add_segment,
            inputs=[segment_start, segment_end, task_type, task_side, 
                   severity_score, segment_notes, annotator_name],
            outputs=[timeline_plot, segments_table, annotation_stats]
        )
        
        delete_segment_btn.click(
            fn=delete_segment,
            inputs=[selected_segment_id],
            outputs=[timeline_plot, segments_table, annotation_stats]
        )
        
        clear_segment_btn.click(
            fn=lambda: [0, 5, PDTaskType.FINGER_TAPPING.value, "bilateral", 0, "", ""],
            outputs=[segment_start, segment_end, task_type, task_side, 
                    severity_score, segment_notes, annotator_name]
        )
        
        save_annotations_btn.click(
            fn=save_annotations,
            outputs=[annotation_file, annotation_stats]
        )
        
        load_annotations_btn.upload(
            fn=load_annotations,
            inputs=[load_annotations_btn],
            outputs=[timeline_plot, segments_table, annotation_stats]
        )
        
        export_for_analysis_btn.click(
            fn=export_for_analysis,
            outputs=[analysis_csv]
        )
        
        filtered_analysis_btn.click(
            fn=run_filtered_analysis,
            inputs=[task_filter],
            outputs=[filtered_results]
        )
    
    return interface


# Main execution
if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7860, max_port=7880):
        """Find an available port"""
        for port in range(start_port, max_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        return None
    
    print("Starting PD Movement Analysis Server...")
    interface = create_interface()
    
    # Find available port
    port = find_free_port()
    if port is None:
        print("Error: No available ports found in range 7860-7880")
        print("Please specify a different port using: GRADIO_SERVER_PORT=<port> python gradio_app.py")
    else:
        print(f"Launching on port {port}...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            inbrowser=False
        )