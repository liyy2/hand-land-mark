#!/usr/bin/env python
"""
Video Annotation Module for Clinical PD Assessment
Provides tools for segmenting and labeling video segments with standardized PD tasks
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class PDTaskType(Enum):
    """Standardized PD assessment task types based on MDS-UPDRS"""
    
    # Rest Assessment
    REST_TREMOR = "Rest Tremor"
    REST_HANDS_LAP = "Rest - Hands on Lap"
    REST_HANDS_HANGING = "Rest - Hands Hanging"
    REST_DISTRACTION = "Rest with Distraction Task"
    
    # Action/Postural Tremor
    POSTURAL_TREMOR = "Postural Tremor - Arms Extended"
    KINETIC_TREMOR = "Kinetic/Action Tremor"
    WING_BEATING = "Wing-beating Position"
    
    # Bradykinesia Tests
    FINGER_TAPPING = "Finger Tapping"
    HAND_OPENING_CLOSING = "Hand Opening/Closing"
    PRONATION_SUPINATION = "Pronation-Supination"
    TOE_TAPPING = "Toe Tapping"
    LEG_AGILITY = "Leg Agility"
    
    # Rigidity Assessment
    RIGIDITY_WRIST = "Rigidity - Wrist"
    RIGIDITY_ELBOW = "Rigidity - Elbow"
    RIGIDITY_SHOULDER = "Rigidity - Shoulder"
    
    # Gait and Balance
    GAIT_WALKING = "Gait - Normal Walking"
    GAIT_TURNING = "Gait - Turning"
    GAIT_HEEL_WALKING = "Gait - Heel Walking"
    GAIT_TANDEM = "Gait - Tandem Walking"
    FREEZING_OF_GAIT = "Freezing of Gait"
    POSTURAL_STABILITY = "Postural Stability/Pull Test"
    
    # Facial Assessment
    FACIAL_EXPRESSION = "Facial Expression/Hypomimia"
    SPEECH_ASSESSMENT = "Speech Assessment"
    GLABELLAR_REFLEX = "Glabellar Reflex"
    
    # Complex Tasks
    FINGER_TO_NOSE = "Finger-to-Nose Test"
    HEEL_TO_SHIN = "Heel-to-Shin Test"
    RAPID_ALTERNATING = "Rapid Alternating Movements"
    ARISE_FROM_CHAIR = "Arise from Chair"
    
    # Writing/Drawing
    WRITING_SAMPLE = "Writing Sample"
    SPIRAL_DRAWING = "Spiral Drawing"
    
    # Other
    PREPARATION = "Preparation/Setup"
    INSTRUCTION = "Instruction/Explanation"
    TRANSITION = "Transition Between Tasks"
    BREAK = "Break/Rest Period"
    INVALID = "Invalid/Unusable Segment"
    OTHER = "Other/Custom Task"


@dataclass
class VideoSegment:
    """Represents a labeled segment of video"""
    start_time: float  # in seconds
    end_time: float    # in seconds
    task_type: str     # PDTaskType value
    side: str          # 'left', 'right', 'bilateral', 'n/a'
    severity: Optional[int] = None  # 0-4 UPDRS scale
    notes: str = ""
    confidence: float = 1.0  # Annotator confidence (0-1)
    annotator: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class VideoAnnotationManager:
    """Manages video segmentation and annotation"""
    
    def __init__(self, video_path: str = None):
        self.video_path = video_path
        self.segments: List[VideoSegment] = []
        self.metadata = {
            'video_path': video_path,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'version': '1.0',
            'total_duration': 0
        }
    
    def add_segment(self, start: float, end: float, task: str, 
                   side: str = 'bilateral', severity: int = None,
                   notes: str = "", confidence: float = 1.0,
                   annotator: str = ""):
        """Add a new segment annotation"""
        
        # Validate times
        if start >= end:
            raise ValueError("Start time must be before end time")
        
        # Check for overlaps (optional - can allow overlaps for multi-task periods)
        overlapping = self.get_overlapping_segments(start, end)
        if overlapping and not self._allow_overlaps():
            raise ValueError(f"Segment overlaps with {len(overlapping)} existing segments")
        
        segment = VideoSegment(
            start_time=start,
            end_time=end,
            task_type=task,
            side=side,
            severity=severity,
            notes=notes,
            confidence=confidence,
            annotator=annotator
        )
        
        self.segments.append(segment)
        self._sort_segments()
        self.metadata['modified'] = datetime.now().isoformat()
        return segment
    
    def remove_segment(self, index: int):
        """Remove a segment by index"""
        if 0 <= index < len(self.segments):
            removed = self.segments.pop(index)
            self.metadata['modified'] = datetime.now().isoformat()
            return removed
        return None
    
    def update_segment(self, index: int, **kwargs):
        """Update a segment's properties"""
        if 0 <= index < len(self.segments):
            segment = self.segments[index]
            for key, value in kwargs.items():
                if hasattr(segment, key):
                    setattr(segment, key, value)
            self._sort_segments()
            self.metadata['modified'] = datetime.now().isoformat()
            return segment
        return None
    
    def get_overlapping_segments(self, start: float, end: float) -> List[VideoSegment]:
        """Find segments that overlap with given time range"""
        overlapping = []
        for segment in self.segments:
            if not (segment.end_time <= start or segment.start_time >= end):
                overlapping.append(segment)
        return overlapping
    
    def get_segments_at_time(self, time: float) -> List[VideoSegment]:
        """Get all segments that contain the given time point"""
        return [s for s in self.segments 
                if s.start_time <= time < s.end_time]
    
    def get_segments_by_task(self, task_type: str) -> List[VideoSegment]:
        """Get all segments of a specific task type"""
        return [s for s in self.segments if s.task_type == task_type]
    
    def get_segments_by_side(self, side: str) -> List[VideoSegment]:
        """Get all segments for a specific side"""
        return [s for s in self.segments if s.side == side]
    
    def merge_adjacent_segments(self, threshold: float = 0.5):
        """Merge segments of same type that are close together"""
        if len(self.segments) < 2:
            return
        
        merged = []
        i = 0
        while i < len(self.segments):
            current = self.segments[i]
            
            # Look for adjacent segment to merge
            j = i + 1
            while j < len(self.segments):
                next_seg = self.segments[j]
                
                # Check if mergeable (same task, same side, close in time)
                if (current.task_type == next_seg.task_type and
                    current.side == next_seg.side and
                    next_seg.start_time - current.end_time <= threshold):
                    
                    # Merge by extending current segment
                    current.end_time = next_seg.end_time
                    if next_seg.notes:
                        current.notes = f"{current.notes}; {next_seg.notes}".strip("; ")
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j
        
        self.segments = merged
        self.metadata['modified'] = datetime.now().isoformat()
    
    def split_segment(self, index: int, split_time: float) -> Tuple[VideoSegment, VideoSegment]:
        """Split a segment at given time"""
        if 0 <= index < len(self.segments):
            segment = self.segments[index]
            
            if segment.start_time < split_time < segment.end_time:
                # Create two new segments
                seg1 = VideoSegment(
                    start_time=segment.start_time,
                    end_time=split_time,
                    task_type=segment.task_type,
                    side=segment.side,
                    severity=segment.severity,
                    notes=f"{segment.notes} (part 1)" if segment.notes else "Part 1",
                    confidence=segment.confidence,
                    annotator=segment.annotator
                )
                
                seg2 = VideoSegment(
                    start_time=split_time,
                    end_time=segment.end_time,
                    task_type=segment.task_type,
                    side=segment.side,
                    severity=segment.severity,
                    notes=f"{segment.notes} (part 2)" if segment.notes else "Part 2",
                    confidence=segment.confidence,
                    annotator=segment.annotator
                )
                
                # Replace original with two segments
                self.segments[index] = seg1
                self.segments.insert(index + 1, seg2)
                self.metadata['modified'] = datetime.now().isoformat()
                
                return seg1, seg2
        
        return None, None
    
    def save_to_json(self, filepath: str):
        """Save annotations to JSON file"""
        data = {
            'metadata': self.metadata,
            'segments': [s.to_dict() for s in self.segments]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filepath: str):
        """Load annotations from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.segments = [VideoSegment.from_dict(s) for s in data.get('segments', [])]
        self._sort_segments()
    
    def export_to_csv(self, filepath: str):
        """Export annotations to CSV for analysis"""
        if not self.segments:
            pd.DataFrame().to_csv(filepath, index=False)
            return
        
        df = pd.DataFrame([s.to_dict() for s in self.segments])
        df.to_csv(filepath, index=False)
    
    def import_from_csv(self, filepath: str):
        """Import annotations from CSV"""
        df = pd.read_csv(filepath)
        self.segments = []
        
        for _, row in df.iterrows():
            segment = VideoSegment(
                start_time=row['start_time'],
                end_time=row['end_time'],
                task_type=row['task_type'],
                side=row.get('side', 'bilateral'),
                severity=row.get('severity', None),
                notes=row.get('notes', ''),
                confidence=row.get('confidence', 1.0),
                annotator=row.get('annotator', ''),
                timestamp=row.get('timestamp', '')
            )
            self.segments.append(segment)
        
        self._sort_segments()
    
    def get_timeline_data(self) -> Dict:
        """Get data formatted for timeline visualization"""
        timeline = []
        
        # Group by task type for coloring
        task_colors = self._generate_task_colors()
        
        for i, segment in enumerate(self.segments):
            timeline.append({
                'id': i,
                'start': segment.start_time,
                'end': segment.end_time,
                'task': segment.task_type,
                'side': segment.side,
                'severity': segment.severity,
                'notes': segment.notes,
                'color': task_colors.get(segment.task_type, '#gray'),
                'label': f"{segment.task_type} ({segment.side})"
            })
        
        return {
            'segments': timeline,
            'colors': task_colors,
            'total_duration': self.metadata.get('total_duration', 0)
        }
    
    def generate_analysis_report(self) -> Dict:
        """Generate summary statistics of annotations"""
        if not self.segments:
            return {'error': 'No segments to analyze'}
        
        # Task distribution
        task_counts = {}
        task_durations = {}
        
        for segment in self.segments:
            task = segment.task_type
            duration = segment.duration()
            
            task_counts[task] = task_counts.get(task, 0) + 1
            task_durations[task] = task_durations.get(task, 0) + duration
        
        # Side distribution
        side_counts = {'left': 0, 'right': 0, 'bilateral': 0, 'n/a': 0}
        for segment in self.segments:
            side_counts[segment.side] = side_counts.get(segment.side, 0) + 1
        
        # Severity distribution (for segments with severity scores)
        severity_dist = {}
        for segment in self.segments:
            if segment.severity is not None:
                severity_dist[segment.severity] = severity_dist.get(segment.severity, 0) + 1
        
        # Timeline coverage
        total_annotated = sum(s.duration() for s in self.segments)
        
        return {
            'total_segments': len(self.segments),
            'total_annotated_time': total_annotated,
            'task_counts': task_counts,
            'task_durations': task_durations,
            'side_distribution': side_counts,
            'severity_distribution': severity_dist,
            'average_segment_duration': total_annotated / len(self.segments) if self.segments else 0,
            'unique_tasks': len(task_counts)
        }
    
    def get_segments_for_analysis(self, task_filter: List[str] = None,
                                 side_filter: str = None,
                                 min_duration: float = None) -> pd.DataFrame:
        """Get segments filtered for analysis, returns DataFrame"""
        segments = self.segments
        
        # Apply filters
        if task_filter:
            segments = [s for s in segments if s.task_type in task_filter]
        
        if side_filter:
            segments = [s for s in segments if s.side == side_filter]
        
        if min_duration:
            segments = [s for s in segments if s.duration() >= min_duration]
        
        # Convert to DataFrame
        if segments:
            df = pd.DataFrame([s.to_dict() for s in segments])
            df['duration'] = df['end_time'] - df['start_time']
            return df
        else:
            return pd.DataFrame()
    
    def _sort_segments(self):
        """Sort segments by start time"""
        self.segments.sort(key=lambda s: s.start_time)
    
    def _allow_overlaps(self) -> bool:
        """Whether to allow overlapping segments"""
        return False  # Can be made configurable
    
    def _generate_task_colors(self) -> Dict[str, str]:
        """Generate consistent colors for each task type"""
        colors = {
            PDTaskType.REST_TREMOR.value: '#FF6B6B',
            PDTaskType.REST_HANDS_LAP.value: '#FF8787',
            PDTaskType.REST_HANDS_HANGING.value: '#FFA5A5',
            PDTaskType.POSTURAL_TREMOR.value: '#4ECDC4',
            PDTaskType.KINETIC_TREMOR.value: '#45B7AA',
            PDTaskType.FINGER_TAPPING.value: '#95E77E',
            PDTaskType.HAND_OPENING_CLOSING.value: '#7DD668',
            PDTaskType.PRONATION_SUPINATION.value: '#6BC752',
            PDTaskType.GAIT_WALKING.value: '#FFE66D',
            PDTaskType.GAIT_TURNING.value: '#FFD93D',
            PDTaskType.FACIAL_EXPRESSION.value: '#A8DADC',
            PDTaskType.WRITING_SAMPLE.value: '#B19CD9',
            PDTaskType.PREPARATION.value: '#E0E0E0',
            PDTaskType.INSTRUCTION.value: '#CCCCCC',
            PDTaskType.TRANSITION.value: '#B8B8B8',
            PDTaskType.INVALID.value: '#FF0000',
            PDTaskType.OTHER.value: '#808080'
        }
        
        # Add any missing task types with generated colors
        all_tasks = set(s.task_type for s in self.segments)
        for task in all_tasks:
            if task not in colors:
                # Generate a color based on hash
                import hashlib
                hash_val = int(hashlib.md5(task.encode()).hexdigest()[:6], 16)
                colors[task] = f'#{hash_val:06x}'
        
        return colors


class AnnotationIntegrator:
    """Integrates annotations with landmark analysis"""
    
    @staticmethod
    def filter_landmarks_by_segments(landmarks_df: pd.DataFrame,
                                    annotations: VideoAnnotationManager,
                                    task_filter: List[str] = None) -> pd.DataFrame:
        """Filter landmark data to only include annotated segments"""
        
        filtered_frames = []
        
        for segment in annotations.segments:
            # Apply task filter if specified
            if task_filter and segment.task_type not in task_filter:
                continue
            
            # Get frames within this segment
            mask = (landmarks_df['timestamp'] >= segment.start_time) & \
                   (landmarks_df['timestamp'] < segment.end_time)
            
            segment_data = landmarks_df[mask].copy()
            
            # Add segment metadata
            segment_data['segment_task'] = segment.task_type
            segment_data['segment_side'] = segment.side
            segment_data['segment_severity'] = segment.severity
            segment_data['segment_notes'] = segment.notes
            
            filtered_frames.append(segment_data)
        
        if filtered_frames:
            return pd.concat(filtered_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def analyze_by_task(landmarks_df: pd.DataFrame,
                       annotations: VideoAnnotationManager) -> Dict:
        """Analyze landmarks grouped by task type"""
        
        results = {}
        
        for task_type in set(s.task_type for s in annotations.segments):
            # Get segments for this task
            task_segments = annotations.get_segments_by_task(task_type)
            
            if not task_segments:
                continue
            
            # Combine all frames for this task
            task_frames = []
            for segment in task_segments:
                mask = (landmarks_df['timestamp'] >= segment.start_time) & \
                       (landmarks_df['timestamp'] < segment.end_time)
                task_frames.append(landmarks_df[mask])
            
            if task_frames:
                task_data = pd.concat(task_frames, ignore_index=True)
                
                # Calculate statistics for this task
                results[task_type] = {
                    'total_frames': len(task_data),
                    'total_duration': sum(s.duration() for s in task_segments),
                    'num_segments': len(task_segments),
                    'detection_rate': len(task_data[task_data['landmark_id'] == 0]) / len(task_segments) if task_segments else 0,
                    'segments': [s.to_dict() for s in task_segments]
                }
        
        return results