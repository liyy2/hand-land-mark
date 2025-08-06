#!/usr/bin/env python3
import argparse
import os
from unified_landmark_detector import UnifiedLandmarkDetector
from visualize_landmarks import (
    plot_landmark_timeseries, 
    plot_movement_heatmap,
    create_landmark_summary
)

def main():
    parser = argparse.ArgumentParser(description='Landmark Detection Pipeline for Hands and Face')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Directory to save output files (default: output)')
    
    # Hand detection arguments
    parser.add_argument('--max-hands', type=int, default=2, 
                       help='Maximum number of hands to detect (default: 2)')
    parser.add_argument('--hand-detection-confidence', type=float, default=0.5,
                       help='Minimum hand detection confidence (default: 0.5)')
    parser.add_argument('--hand-tracking-confidence', type=float, default=0.5,
                       help='Minimum hand tracking confidence (default: 0.5)')
    
    # Face detection arguments
    parser.add_argument('--extract-face', action='store_true',
                       help='Enable face landmark extraction')
    parser.add_argument('--max-faces', type=int, default=1,
                       help='Maximum number of faces to detect (default: 1)')
    parser.add_argument('--face-detection-confidence', type=float, default=0.5,
                       help='Minimum face detection confidence (default: 0.5)')
    parser.add_argument('--face-tracking-confidence', type=float, default=0.5,
                       help='Minimum face tracking confidence (default: 0.5)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--enhance-contrast', action='store_true',
                       help='Apply contrast enhancement to improve detection')
    parser.add_argument('--multi-pass', action='store_true',
                       help='Use multiple preprocessing attempts to detect both hands')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' not found")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_basename = os.path.splitext(os.path.basename(args.input_video))[0]
    output_video_path = os.path.join(args.output_dir, f"{input_basename}_landmarks.mp4")
    csv_output_path = os.path.join(args.output_dir, f"{input_basename}_landmarks.csv")
    
    print("Initializing Unified Landmark Detector...")
    detector = UnifiedLandmarkDetector(
        extract_hands=True,
        extract_face=args.extract_face,
        max_num_hands=args.max_hands,
        max_num_faces=args.max_faces,
        hand_detection_confidence=args.hand_detection_confidence,
        hand_tracking_confidence=args.hand_tracking_confidence,
        face_detection_confidence=args.face_detection_confidence,
        face_tracking_confidence=args.face_tracking_confidence
    )
    
    print("\nProcessing video...")
    df = detector.process_video(
        input_video_path=args.input_video,
        output_video_path=output_video_path,
        csv_output_path=csv_output_path,
        enhance_contrast=args.enhance_contrast,
        multi_pass=args.multi_pass
    )
    
    summary_path = os.path.join(args.output_dir, f"{input_basename}_summary.txt")
    create_landmark_summary(df, save_path=summary_path)
    
    if args.visualize and len(df) > 0:
        print("\nGenerating visualizations...")
        
        # Plot hand landmarks if available
        hand_df = df[df['landmark_type'] == 'hand']
        if len(hand_df) > 0:
            hand_timeseries_path = os.path.join(args.output_dir, f"{input_basename}_hand_timeseries.png")
            plot_landmark_timeseries(
                df, 
                landmark_ids=[0, 4, 8, 12, 16, 20],
                landmark_type='hand',
                save_path=hand_timeseries_path
            )
        
        # Plot face landmarks if available
        face_df = df[df['landmark_type'] == 'face']
        if len(face_df) > 0 and args.extract_face:
            face_timeseries_path = os.path.join(args.output_dir, f"{input_basename}_face_timeseries.png")
            plot_landmark_timeseries(
                df,
                landmark_ids=None,  # Use default key face points
                landmark_type='face',
                save_path=face_timeseries_path
            )
        
        import cv2
        cap = cv2.VideoCapture(args.input_video)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Generate heatmaps
        if len(hand_df) > 0:
            hand_heatmap_path = os.path.join(args.output_dir, f"{input_basename}_hand_heatmap.png")
            plot_movement_heatmap(
                df,
                video_width=video_width,
                video_height=video_height,
                landmark_type='hand',
                save_path=hand_heatmap_path
            )
        
        if len(face_df) > 0 and args.extract_face:
            face_heatmap_path = os.path.join(args.output_dir, f"{input_basename}_face_heatmap.png")
            plot_movement_heatmap(
                df,
                video_width=video_width,
                video_height=video_height,
                landmark_type='face',
                save_path=face_heatmap_path
            )
    
    print(f"\n✓ Processing complete!")
    print(f"Output video: {output_video_path}")
    print(f"Landmark data: {csv_output_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()