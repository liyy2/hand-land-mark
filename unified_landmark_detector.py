import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import os

class UnifiedLandmarkDetector:
    def __init__(self, 
                 extract_hands: bool = True,
                 extract_face: bool = False,
                 max_num_hands: int = 2,
                 max_num_faces: int = 1,
                 hand_detection_confidence: float = 0.5,
                 hand_tracking_confidence: float = 0.5,
                 face_detection_confidence: float = 0.5,
                 face_tracking_confidence: float = 0.5):
        """
        Initialize the unified landmark detector for both hands and face.
        
        Args:
            extract_hands: Whether to extract hand landmarks
            extract_face: Whether to extract face landmarks
            max_num_hands: Maximum number of hands to detect
            max_num_faces: Maximum number of faces to detect
            hand_detection_confidence: Minimum confidence for hand detection
            hand_tracking_confidence: Minimum confidence for hand tracking
            face_detection_confidence: Minimum confidence for face detection
            face_tracking_confidence: Minimum confidence for face tracking
        """
        self.extract_hands = extract_hands
        self.extract_face = extract_face
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic detector for better hand detection
        if extract_hands:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,  # Use tracking for better temporal consistency
                min_detection_confidence=hand_detection_confidence,
                min_tracking_confidence=hand_tracking_confidence,
                model_complexity=2  # Use most accurate model
            )
            self.mp_hands = mp.solutions.hands  # Keep for drawing utilities
        
        # Initialize face detector
        if extract_face:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,  # Enables iris landmarks
                min_detection_confidence=face_detection_confidence,
                min_tracking_confidence=face_tracking_confidence
            )
        
        print(f"Unified detector initialized - Hands: {extract_hands}, Face: {extract_face}")
        
    def process_frame(self, frame: np.ndarray, frame_idx: int, enhance_contrast: bool = False, multi_pass: bool = False) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process a single frame to detect hand and/or face landmarks.
        
        Args:
            frame: Input frame (BGR format)
            frame_idx: Frame index/number
            enhance_contrast: Whether to apply contrast enhancement
            
        Returns:
            Tuple of (annotated_frame, landmarks_dict)
        """
        if frame is None or frame.size == 0:
            return frame, None
            
        # Apply contrast enhancement if requested
        if enhance_contrast:
            # Simple and effective preprocessing
            # Step 1: Gamma correction to brighten dark areas
            gamma = 1.3
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            brightened = cv2.LUT(frame, table)
            
            # Step 2: CLAHE enhancement
            lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            l = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            enhanced_frame = cv2.merge([l, a, b])
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
            
            frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        landmarks_dict = {
            'frame': frame_idx,
            'hands': [],
            'faces': []
        }
        
        annotated_frame = frame.copy()
        
        # Process hands using holistic model for better detection
        if self.extract_hands:
            holistic_results = self.holistic.process(frame_rgb)
            
            hand_idx = 0
            # Process left hand
            if holistic_results.left_hand_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    holistic_results.left_hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks_list = []
                for landmark_idx, landmark in enumerate(holistic_results.left_hand_landmarks.landmark):
                    landmarks_list.append({
                        'landmark_id': landmark_idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                landmarks_dict['hands'].append({
                    'hand_id': hand_idx,
                    'hand_label': 'Left',
                    'confidence': 0.99,  # Holistic doesn't provide per-hand confidence
                    'landmarks': landmarks_list
                })
                hand_idx += 1
            
            # Process right hand
            if holistic_results.right_hand_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    holistic_results.right_hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks_list = []
                for landmark_idx, landmark in enumerate(holistic_results.right_hand_landmarks.landmark):
                    landmarks_list.append({
                        'landmark_id': landmark_idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                landmarks_dict['hands'].append({
                    'hand_id': hand_idx,
                    'hand_label': 'Right',
                    'confidence': 0.99,  # Holistic doesn't provide per-hand confidence
                    'landmarks': landmarks_list
                })
        
        # Process face
        if self.extract_face:
            face_results = self.face_mesh.process(frame_rgb)
            
            if face_results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                    # Draw face mesh
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_TESSELATION,
                        None,
                        self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw contours
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        None,
                        self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    landmarks_list = []
                    for landmark_idx, landmark in enumerate(face_landmarks.landmark):
                        landmarks_list.append({
                            'landmark_id': landmark_idx,
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    landmarks_dict['faces'].append({
                        'face_id': face_idx,
                        'landmarks': landmarks_list
                    })
        
        return annotated_frame, landmarks_dict
    
    def process_video(self, 
                     input_video_path: str, 
                     output_video_path: str,
                     csv_output_path: str = None,
                     enhance_contrast: bool = False,
                     multi_pass: bool = False) -> pd.DataFrame:
        """
        Process entire video to detect landmarks.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video with landmarks
            csv_output_path: Optional path to save landmarks as CSV
            enhance_contrast: Whether to apply contrast enhancement to improve detection
            
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
        print(f"Extracting: {'Hands' if self.extract_hands else ''} {'Face' if self.extract_face else ''}")
        
        frame_idx = 0
        all_landmarks_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            annotated_frame, landmarks_dict = self.process_frame(frame, frame_idx, enhance_contrast, multi_pass)
            
            # Process hand landmarks
            if landmarks_dict and self.extract_hands:
                for hand in landmarks_dict['hands']:
                    for landmark in hand['landmarks']:
                        all_landmarks_data.append({
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps,
                            'landmark_type': 'hand',
                            'body_part_id': hand['hand_id'],
                            'label': hand['hand_label'],
                            'confidence': hand['confidence'],
                            'landmark_id': landmark['landmark_id'],
                            'x': landmark['x'],
                            'y': landmark['y'],
                            'z': landmark['z'],
                            'x_pixel': max(0, min(width-1, int(landmark['x'] * width))),
                            'y_pixel': max(0, min(height-1, int(landmark['y'] * height)))
                        })
            
            # Process face landmarks
            if landmarks_dict and self.extract_face:
                for face in landmarks_dict['faces']:
                    for landmark in face['landmarks']:
                        all_landmarks_data.append({
                            'frame': frame_idx,
                            'timestamp': frame_idx / fps,
                            'landmark_type': 'face',
                            'body_part_id': face['face_id'],
                            'label': f"Face_{face['face_id']}",
                            'confidence': 1.0,  # Face mesh doesn't provide confidence
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
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()