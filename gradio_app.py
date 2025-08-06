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

# Import existing modules
from unified_landmark_detector import UnifiedLandmarkDetector
from pd_analysis import PDMovementAnalyzer
from visualize_pd_analysis import create_comprehensive_visualization, create_time_series_plot

class PDAnalysisApp:
    """Main application class for PD movement analysis"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="pd_analysis_")
        self.current_session = None
        
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
            
            # Tab 3: Instructions
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