import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os

class HandLandmarkDetector:
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand landmark detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmark_data = []
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process a single frame to detect hand landmarks.
        
        Args:
            frame: Input frame (BGR format)
            frame_idx: Frame index/number
            
        Returns:
            Tuple of (annotated_frame, landmarks_dict)
        """
        if frame is None or frame.size == 0:
            return frame, None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        results = self.hands.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        landmarks_dict = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            landmarks_dict = {
                'frame': frame_idx,
                'hands': []
            }
            
            num_hands = min(len(results.multi_hand_landmarks), len(results.multi_handedness))
            
            for hand_idx in range(num_hands):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                handedness = results.multi_handedness[hand_idx]
                
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                
                landmarks_list = []
                for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks_list.append({
                        'landmark_id': landmark_idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                landmarks_dict['hands'].append({
                    'hand_id': hand_idx,
                    'hand_label': hand_label,
                    'confidence': hand_score,
                    'landmarks': landmarks_list
                })
        
        return annotated_frame, landmarks_dict
    
    def process_video(self, 
                     input_video_path: str, 
                     output_video_path: str,
                     csv_output_path: str = None) -> pd.DataFrame:
        """
        Process entire video to detect hand landmarks.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video with landmarks
            csv_output_path: Optional path to save landmarks as CSV
            
        Returns:
            DataFrame containing all landmark data
        """
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if fps <= 0:
            fps = 30  # Default fallback
            print(f"Warning: Invalid FPS detected, using default: {fps}")
        
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot create output video file: {output_video_path}")
        
        print(f"Processing video: {input_video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        
        frame_idx = 0
        all_landmarks_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, landmarks_dict = self.process_frame(frame, frame_idx)
            
            if landmarks_dict:
                for hand in landmarks_dict['hands']:
                    for landmark in hand['landmarks']:
                        all_landmarks_data.append({
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps,
                            'hand_id': hand['hand_id'],
                            'hand_label': hand['hand_label'],
                            'confidence': hand['confidence'],
                            'landmark_id': landmark['landmark_id'],
                            'x': landmark['x'],
                            'y': landmark['y'],
                            'z': landmark['z'],
                            'x_pixel': max(0, min(width-1, int(landmark['x'] * width))),
                            'y_pixel': max(0, min(height-1, int(landmark['y'] * height)))
                        })
            
            out.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0 and total_frames > 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        df = pd.DataFrame(all_landmarks_data)
        
        if csv_output_path and len(df) > 0:
            # Ensure directory exists
            csv_dir = os.path.dirname(csv_output_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
            df.to_csv(csv_output_path, index=False)
            print(f"Saved landmark data to: {csv_output_path}")
        
        print(f"Video processing complete. Output saved to: {output_video_path}")
        
        return df
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'hands'):
            self.hands.close()