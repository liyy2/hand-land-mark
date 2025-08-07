#!/usr/bin/env python
"""
Simple Video Annotator with Manual Time Entry
Since Gradio doesn't expose video currentTime API, we use manual time entry
"""

import gradio as gr
import cv2
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class SimpleVideoAnnotator:
    """Simple annotation interface with manual time entry"""
    
    def __init__(self):
        self.annotations = []
        self.video_info = {}
    
    def get_video_info(self, video_path):
        """Extract video information"""
        if not video_path:
            return "No video loaded", 0
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.video_info = {
            'path': video_path,
            'fps': fps,
            'frames': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
        
        info_text = f"""
        **Video Information:**
        - Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)
        - FPS: {fps:.2f}
        - Resolution: {width}x{height}
        - Total Frames: {int(frame_count)}
        
        **Time Helper:**
        - For time 0:13 → Enter 13
        - For time 1:30 → Enter 90
        - For time 2:45 → Enter 165
        """
        
        return info_text, duration
    
    def add_annotation(self, start_time, end_time, task_type, notes):
        """Add a new annotation"""
        if start_time >= end_time:
            return "Error: Start time must be before end time", self.get_annotations_df()
        
        annotation = {
            'id': len(self.annotations),
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'task': task_type,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        self.annotations.append(annotation)
        
        return f"Added: {task_type} from {start_time:.1f}s to {end_time:.1f}s", self.get_annotations_df()
    
    def get_annotations_df(self):
        """Get annotations as DataFrame"""
        if not self.annotations:
            return pd.DataFrame(columns=['ID', 'Start', 'End', 'Duration', 'Task', 'Notes'])
        
        df = pd.DataFrame(self.annotations)
        return df[['id', 'start', 'end', 'duration', 'task', 'notes']].rename(columns={
            'id': 'ID', 'start': 'Start', 'end': 'End', 
            'duration': 'Duration', 'task': 'Task', 'notes': 'Notes'
        })
    
    def save_annotations(self, output_path="annotations.json"):
        """Save annotations to file"""
        data = {
            'video_info': self.video_info,
            'annotations': self.annotations,
            'created': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Saved {len(self.annotations)} annotations to {output_path}"
    
    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations = []
        return "Annotations cleared", self.get_annotations_df()


def create_annotation_interface():
    """Create Gradio interface for video annotation"""
    
    annotator = SimpleVideoAnnotator()
    
    with gr.Blocks(title="Video Time Annotator") as interface:
        gr.Markdown("""
        # 🎥 Simple Video Time Annotator
        
        **Instructions:**
        1. Upload a video below
        2. Play the video and note timestamps
        3. Manually enter start/end times in seconds
        4. Add task type and notes
        5. Click "Add Annotation"
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                video = gr.Video(label="Video Player")
                video_info = gr.Markdown("Upload a video to begin")
                
            with gr.Column(scale=1):
                gr.Markdown("### Annotation Controls")
                
                gr.Markdown("**Quick Time Calculator:**")
                with gr.Row():
                    minutes_input = gr.Number(label="Minutes", value=0, precision=0)
                    seconds_input = gr.Number(label="Seconds", value=0, precision=0)
                    calculated_time = gr.Number(label="Total Seconds", value=0, precision=1, interactive=False)
                
                def calculate_seconds(mins, secs):
                    return mins * 60 + secs
                
                gr.Markdown("---")
                
                start_time = gr.Number(label="Start Time (seconds)", value=0, precision=1)
                end_time = gr.Number(label="End Time (seconds)", value=5, precision=1)
                
                task_type = gr.Dropdown(
                    label="Task Type",
                    choices=[
                        "Finger Tapping",
                        "Hand Opening/Closing", 
                        "Rest Tremor",
                        "Postural Tremor",
                        "Gait",
                        "Other"
                    ],
                    value="Finger Tapping"
                )
                
                notes = gr.Textbox(label="Notes", placeholder="Optional notes")
                
                with gr.Row():
                    add_btn = gr.Button("➕ Add Annotation", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear All", variant="stop")
                
                status = gr.Textbox(label="Status", interactive=False)
        
        # Annotations table
        annotations_table = gr.Dataframe(
            label="Annotations",
            headers=["ID", "Start", "End", "Duration", "Task", "Notes"],
            interactive=False
        )
        
        with gr.Row():
            save_btn = gr.Button("💾 Save Annotations", variant="primary")
            save_status = gr.Textbox(label="Save Status", interactive=False)
        
        # Event handlers
        video.upload(
            fn=annotator.get_video_info,
            inputs=[video],
            outputs=[video_info, end_time]
        )
        
        # Time calculator
        minutes_input.change(
            fn=calculate_seconds,
            inputs=[minutes_input, seconds_input],
            outputs=[calculated_time]
        )
        
        seconds_input.change(
            fn=calculate_seconds,
            inputs=[minutes_input, seconds_input],
            outputs=[calculated_time]
        )
        
        add_btn.click(
            fn=annotator.add_annotation,
            inputs=[start_time, end_time, task_type, notes],
            outputs=[status, annotations_table]
        )
        
        clear_btn.click(
            fn=annotator.clear_annotations,
            outputs=[status, annotations_table]
        )
        
        save_btn.click(
            fn=annotator.save_annotations,
            outputs=[save_status]
        )
    
    return interface


if __name__ == "__main__":
    interface = create_annotation_interface()
    interface.launch(share=False)