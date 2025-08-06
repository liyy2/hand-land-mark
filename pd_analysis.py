"""
Parkinson's Disease Movement Analysis Module
Analyzes hand landmark data for PD-specific biomarkers including:
- Tremor frequency and amplitude
- Finger tapping metrics
- Bradykinesia detection
- Movement asymmetry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PDMovementAnalyzer:
    """Analyzes hand movement data for Parkinson's Disease biomarkers"""
    
    # MediaPipe hand landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_MCP = 5
    MIDDLE_TIP = 12
    WRIST = 0
    
    def __init__(self, csv_path: str, fps: float = 119.0):
        """
        Initialize analyzer with landmark data
        
        Args:
            csv_path: Path to landmark CSV file
            fps: Video frame rate (default 119 for the test video)
        """
        self.df = pd.read_csv(csv_path)
        self.fps = fps
        self.dt = 1.0 / fps  # Time between frames
        
        # Separate left and right hand data
        self.left_hand = self.df[self.df['label'] == 'Left'].copy()
        self.right_hand = self.df[self.df['label'] == 'Right'].copy()
        
        print(f"Loaded {len(self.df)} landmark records")
        print(f"Left hand frames: {len(self.left_hand['frame'].unique())}")
        print(f"Right hand frames: {len(self.right_hand['frame'].unique())}")
        
    def calculate_tremor_frequency(self, hand: str = 'both', 
                                  landmark_id: int = None) -> Dict:
        """
        Calculate tremor frequency using FFT analysis
        
        PD resting tremor: typically 4-6 Hz
        PD action tremor: typically 4-12 Hz
        Essential tremor: typically 4-11 Hz
        
        Args:
            hand: 'left', 'right', or 'both'
            landmark_id: Specific landmark to analyze (default: index finger tip)
        
        Returns:
            Dictionary with frequency analysis results
        """
        if landmark_id is None:
            landmark_id = self.INDEX_TIP
            
        results = {}
        
        hands_to_analyze = []
        if hand in ['left', 'both'] and len(self.left_hand) > 0:
            hands_to_analyze.append(('left', self.left_hand))
        if hand in ['right', 'both'] and len(self.right_hand) > 0:
            hands_to_analyze.append(('right', self.right_hand))
            
        for hand_name, hand_data in hands_to_analyze:
            # Get position data for specific landmark
            landmark_data = hand_data[hand_data['landmark_id'] == landmark_id].sort_values('frame')
            
            if len(landmark_data) < 100:  # Need enough data for FFT
                print(f"Insufficient data for {hand_name} hand")
                continue
                
            # Extract position time series
            x = landmark_data['x'].values
            y = landmark_data['y'].values
            z = landmark_data['z'].values
            
            # Remove trend using detrending
            x_detrended = signal.detrend(x)
            y_detrended = signal.detrend(y)
            z_detrended = signal.detrend(z)
            
            # Apply Welch's method for power spectral density
            freq_x, psd_x = signal.welch(x_detrended, self.fps, nperseg=min(256, len(x_detrended)))
            freq_y, psd_y = signal.welch(y_detrended, self.fps, nperseg=min(256, len(y_detrended)))
            freq_z, psd_z = signal.welch(z_detrended, self.fps, nperseg=min(256, len(z_detrended)))
            
            # Find dominant frequency in PD range (3-12 Hz)
            pd_range = (freq_x >= 3) & (freq_x <= 12)
            
            # Calculate combined power (magnitude of 3D movement)
            combined_psd = psd_x + psd_y + psd_z
            
            # Find peak frequency
            peak_idx = np.argmax(combined_psd[pd_range])
            peak_freq = freq_x[pd_range][peak_idx]
            peak_power = combined_psd[pd_range][peak_idx]
            
            # Calculate tremor amplitude (RMS of detrended signal)
            tremor_amplitude = np.sqrt(np.mean(x_detrended**2 + y_detrended**2 + z_detrended**2))
            
            # Detect if tremor is present (LOWER power threshold for better sensitivity)
            power_threshold = np.mean(combined_psd) + 1 * np.std(combined_psd)  # Reduced from 2 to 1 std
            has_tremor = peak_power > power_threshold or (tremor_amplitude > 0.005 and peak_freq >= 3)
            
            # Classify tremor type based on frequency
            tremor_type = 'none'
            if has_tremor:
                if 4 <= peak_freq <= 6:
                    tremor_type = 'pd_resting'
                elif 6 < peak_freq <= 12:
                    tremor_type = 'pd_action_or_essential'
                else:
                    tremor_type = 'other'
                    
            results[hand_name] = {
                'peak_frequency': peak_freq,
                'peak_power': peak_power,
                'tremor_amplitude': tremor_amplitude,
                'has_tremor': has_tremor,
                'tremor_type': tremor_type,
                'frequencies': freq_x,
                'psd': combined_psd,
                'psd_x': psd_x,
                'psd_y': psd_y,
                'psd_z': psd_z
            }
            
        return results
    
    def analyze_finger_tapping(self, hand: str = 'both', 
                              window_size: int = 30) -> Dict:
        """
        Analyze finger tapping patterns for bradykinesia and sequence effect
        
        Key metrics:
        - Tapping frequency/speed
        - Amplitude (distance between thumb and index)
        - Amplitude decrement (progressive reduction)
        - Rhythm regularity
        - Hesitations/freezing
        
        Args:
            hand: 'left', 'right', or 'both'
            window_size: Frames for sliding window analysis
        
        Returns:
            Dictionary with tapping analysis results
        """
        results = {}
        
        hands_to_analyze = []
        if hand in ['left', 'both'] and len(self.left_hand) > 0:
            hands_to_analyze.append(('left', self.left_hand))
        if hand in ['right', 'both'] and len(self.right_hand) > 0:
            hands_to_analyze.append(('right', self.right_hand))
            
        for hand_name, hand_data in hands_to_analyze:
            # Get thumb and index finger tips
            thumb = hand_data[hand_data['landmark_id'] == self.THUMB_TIP].sort_values('frame')
            index = hand_data[hand_data['landmark_id'] == self.INDEX_TIP].sort_values('frame')
            
            # Merge on frame to get paired positions
            merged = pd.merge(thumb, index, on='frame', suffixes=('_thumb', '_index'))
            
            if len(merged) < 10:
                print(f"Insufficient tapping data for {hand_name} hand")
                continue
                
            # Calculate distance between thumb and index
            distances = []
            for _, row in merged.iterrows():
                dist = euclidean(
                    [row['x_thumb'], row['y_thumb'], row['z_thumb']],
                    [row['x_index'], row['y_index'], row['z_index']]
                )
                distances.append(dist)
                
            distances = np.array(distances)
            
            # Detect taps using distance threshold and peaks
            # Normalize distances
            dist_normalized = (distances - np.mean(distances)) / np.std(distances)
            
            # Find peaks (maximum separation during tap opening)
            peaks, properties = signal.find_peaks(dist_normalized, 
                                                 prominence=0.5,
                                                 distance=int(self.fps * 0.1))  # Min 0.1s between taps
            
            # Find valleys (minimum separation during tap closing)
            valleys, _ = signal.find_peaks(-dist_normalized,
                                          prominence=0.5,
                                          distance=int(self.fps * 0.1))
            
            # Calculate tapping metrics
            if len(peaks) > 1:
                # Tapping frequency
                tap_intervals = np.diff(merged.iloc[peaks]['frame'].values) / self.fps
                tap_frequency = 1.0 / np.mean(tap_intervals) if len(tap_intervals) > 0 else 0
                
                # Amplitude metrics
                tap_amplitudes = distances[peaks]
                mean_amplitude = np.mean(tap_amplitudes)
                
                # Amplitude decrement (linear regression slope)
                if len(tap_amplitudes) > 3:
                    x = np.arange(len(tap_amplitudes))
                    slope, _ = np.polyfit(x, tap_amplitudes, 1)
                    amplitude_decrement = -slope / mean_amplitude * 100  # Percentage decrease
                else:
                    amplitude_decrement = 0
                    
                # Rhythm regularity (coefficient of variation)
                if len(tap_intervals) > 1:
                    rhythm_cv = np.std(tap_intervals) / np.mean(tap_intervals)
                else:
                    rhythm_cv = 0
                    
                # Detect hesitations (long pauses)
                hesitations = np.sum(tap_intervals > 2 * np.mean(tap_intervals))
                
                # Speed changes over time (divide into thirds)
                n_taps = len(peaks)
                third = n_taps // 3
                if third > 0:
                    early_speed = np.mean(1.0 / tap_intervals[:third]) if third > 0 else 0
                    late_speed = np.mean(1.0 / tap_intervals[-third:]) if third > 0 else 0
                    speed_change = (late_speed - early_speed) / early_speed * 100 if early_speed > 0 else 0
                else:
                    speed_change = 0
                    
            else:
                # No valid taps detected
                tap_frequency = 0
                mean_amplitude = np.mean(distances)
                amplitude_decrement = 0
                rhythm_cv = 0
                hesitations = 0
                speed_change = 0
                tap_intervals = []
                tap_amplitudes = []
                
            results[hand_name] = {
                'tap_count': len(peaks),
                'tap_frequency': tap_frequency,
                'mean_amplitude': mean_amplitude,
                'amplitude_decrement': amplitude_decrement,
                'rhythm_cv': rhythm_cv,
                'hesitations': hesitations,
                'speed_change': speed_change,
                'tap_intervals': tap_intervals,
                'tap_amplitudes': tap_amplitudes,
                'distances': distances,
                'peaks': peaks,
                'valleys': valleys
            }
            
        return results
    
    def calculate_movement_asymmetry(self) -> Dict:
        """
        Calculate asymmetry between left and right hand movements
        Significant asymmetry is a hallmark of PD
        
        Returns:
            Dictionary with asymmetry metrics
        """
        # Get tremor analysis for both hands
        tremor = self.calculate_tremor_frequency('both')
        tapping = self.analyze_finger_tapping('both')
        
        results = {
            'has_both_hands': 'left' in tremor and 'right' in tremor
        }
        
        if results['has_both_hands']:
            # Tremor asymmetry
            left_amp = tremor['left']['tremor_amplitude']
            right_amp = tremor['right']['tremor_amplitude']
            results['tremor_asymmetry'] = abs(left_amp - right_amp) / max(left_amp, right_amp, 0.001)
            
            # Frequency asymmetry
            left_freq = tremor['left']['peak_frequency']
            right_freq = tremor['right']['peak_frequency']
            results['frequency_asymmetry'] = abs(left_freq - right_freq)
            
            # Tapping asymmetry
            if 'left' in tapping and 'right' in tapping:
                left_tap_freq = tapping['left']['tap_frequency']
                right_tap_freq = tapping['right']['tap_frequency']
                if max(left_tap_freq, right_tap_freq) > 0:
                    results['tapping_asymmetry'] = abs(left_tap_freq - right_tap_freq) / max(left_tap_freq, right_tap_freq)
                else:
                    results['tapping_asymmetry'] = 0
                    
                # Amplitude asymmetry
                left_tap_amp = tapping['left']['mean_amplitude']
                right_tap_amp = tapping['right']['mean_amplitude']
                results['amplitude_asymmetry'] = abs(left_tap_amp - right_tap_amp) / max(left_tap_amp, right_tap_amp, 0.001)
            
            # Determine more affected side
            if results.get('tremor_asymmetry', 0) > 0.2:  # 20% difference threshold
                results['more_affected'] = 'left' if left_amp > right_amp else 'right'
            else:
                results['more_affected'] = 'bilateral'
                
        return results
    
    def detect_bradykinesia(self, hand: str = 'both') -> Dict:
        """
        Detect bradykinesia (slowness of movement) patterns
        
        Returns:
            Dictionary with bradykinesia indicators
        """
        tapping = self.analyze_finger_tapping(hand)
        results = {}
        
        for hand_name, tap_data in tapping.items():
            bradykinesia_score = 0
            indicators = []
            
            # Check for slow tapping (< 3 Hz is considered slow)
            if tap_data['tap_frequency'] < 3 and tap_data['tap_frequency'] > 0:
                bradykinesia_score += 1
                indicators.append('slow_tapping')
                
            # Check for amplitude decrement (> 30% is significant)
            if tap_data['amplitude_decrement'] > 30:
                bradykinesia_score += 1
                indicators.append('amplitude_decrement')
                
            # Check for irregular rhythm (CV > 0.3)
            if tap_data['rhythm_cv'] > 0.3:
                bradykinesia_score += 1
                indicators.append('irregular_rhythm')
                
            # Check for progressive slowing (> 20% speed decrease)
            if tap_data['speed_change'] < -20:
                bradykinesia_score += 1
                indicators.append('progressive_slowing')
                
            # Check for hesitations
            if tap_data['hesitations'] > 2:
                bradykinesia_score += 1
                indicators.append('hesitations')
                
            results[hand_name] = {
                'bradykinesia_score': bradykinesia_score,
                'severity': self._classify_severity(bradykinesia_score),
                'indicators': indicators
            }
            
        return results
    
    def _classify_severity(self, score: int) -> str:
        """Classify bradykinesia severity based on score"""
        if score == 0:
            return 'none'
        elif score <= 2:
            return 'mild'
        elif score <= 3:
            return 'moderate'
        else:
            return 'severe'
    
    def detect_movement_type_per_hand(self, window_size: int = 60, overlap: float = 0.5) -> Dict:
        """
        Classify each hand's movement type in sliding windows
        Detects whether hand is performing tapping, tremor, or static movement
        
        Args:
            window_size: Number of frames per window (60 = 0.5s at 120fps)
            overlap: Overlap fraction between windows
        
        Returns:
            Dictionary with movement classification for each hand over time
        """
        results = {'left': [], 'right': []}
        step_size = int(window_size * (1 - overlap))
        
        for hand_name, hand_data in [('left', self.left_hand), ('right', self.right_hand)]:
            if len(hand_data) == 0:
                continue
                
            # Get thumb and index positions for tapping detection
            thumb = hand_data[hand_data['landmark_id'] == self.THUMB_TIP].sort_values('frame')
            index = hand_data[hand_data['landmark_id'] == self.INDEX_TIP].sort_values('frame')
            merged = pd.merge(thumb, index, on='frame', suffixes=('_thumb', '_index'))
            
            if len(merged) < window_size:
                continue
            
            # Process sliding windows
            for start_idx in range(0, len(merged) - window_size, step_size):
                window = merged.iloc[start_idx:start_idx + window_size]
                
                # Calculate thumb-index distance
                distances = np.sqrt(
                    (window['x_thumb'] - window['x_index'])**2 +
                    (window['y_thumb'] - window['y_index'])**2 +
                    (window['z_thumb'] - window['z_index'])**2
                ).values
                
                # Detect movement type
                movement_type = self._classify_movement_pattern(distances)
                
                results[hand_name].append({
                    'start_frame': window.iloc[0]['frame'],
                    'end_frame': window.iloc[-1]['frame'],
                    'movement_type': movement_type,
                    'confidence': self._calculate_pattern_confidence(distances, movement_type)
                })
        
        return results
    
    def _classify_movement_pattern(self, distances: np.ndarray) -> str:
        """
        Classify movement pattern based on signal morphology and frequency
        Prioritizes discrete events (tapping) over continuous oscillation (tremor)
        
        Args:
            distances: Array of thumb-index distances
            
        Returns:
            'tapping', 'tremor', or 'static'
        """
        # Detrend the signal
        distances_detrended = signal.detrend(distances)
        
        # 1. DETECT DISCRETE EVENTS (Primary criterion for tapping)
        # Find peaks (finger opening) and valleys (finger closing)
        peaks, peak_props = signal.find_peaks(
            distances, 
            prominence=np.std(distances)*0.25,  # Sensitive to smaller movements
            distance=int(self.fps * 0.1)  # Min 0.1s between peaks
        )
        
        valleys, valley_props = signal.find_peaks(
            -distances, 
            prominence=np.std(distances)*0.25,
            distance=int(self.fps * 0.1)
        )
        
        # 2. ANALYZE SIGNAL MORPHOLOGY
        # Check if signal returns to baseline between peaks (tapping characteristic)
        has_clear_valleys = len(valleys) >= len(peaks) * 0.5
        
        # Calculate discreteness: how much of the signal is "at rest" vs "in motion"
        if len(peaks) > 1 and len(peak_props.get('widths', [])) > 0:
            avg_peak_width = np.mean(peak_props['widths']) / self.fps  # Convert to seconds
            avg_interval = np.mean(np.diff(peaks)) / self.fps
            discreteness_ratio = 1 - (avg_peak_width / avg_interval) if avg_interval > 0 else 0
        else:
            discreteness_ratio = 0
        
        # 3. FREQUENCY ANALYSIS (Secondary criterion)
        freqs, psd = signal.welch(distances_detrended, self.fps, nperseg=min(128, len(distances)))
        
        # Define frequency bands
        tapping_band = (freqs >= 0.5) & (freqs <= 3.5)  # Typical tapping: 0.5-3.5 Hz
        pd_tremor_band = (freqs >= 4) & (freqs <= 6)    # PD resting tremor: 4-6 Hz
        action_tremor_band = (freqs >= 6) & (freqs <= 12)  # Action/essential tremor
        
        tapping_power = np.sum(psd[tapping_band])
        pd_tremor_power = np.sum(psd[pd_tremor_band])
        action_tremor_power = np.sum(psd[action_tremor_band])
        total_power = np.sum(psd)
        
        # Find dominant frequency
        peak_freq_idx = np.argmax(psd)
        dominant_freq = freqs[peak_freq_idx]
        
        # 4. SIGNAL STATISTICS
        signal_variance = np.var(distances_detrended)
        is_static = signal_variance < 0.00005  # Very little movement
        
        # 5. CLASSIFICATION DECISION TREE
        
        # TAPPING: Discrete events with clear peaks and valleys
        if len(peaks) >= 2 and has_clear_valleys:
            # Calculate tap frequency from peaks
            tap_intervals = np.diff(peaks) / self.fps
            tap_frequency = 1.0 / np.mean(tap_intervals) if len(tap_intervals) > 0 else 0
            
            # Confirm it's in reasonable tapping range
            if 0.3 <= tap_frequency <= 4:  # Extended range for slow tapping
                # Additional check: tapping should have most power in low frequencies
                if tapping_power > (pd_tremor_power + action_tremor_power) * 0.7:
                    return 'tapping'
                # Even if frequency is mixed, discrete peaks indicate tapping
                elif discreteness_ratio > 0.3:
                    return 'tapping'
        
        # STATIC: Very little movement
        if is_static:
            return 'static'
        
        # TREMOR: Continuous oscillation without discrete events
        # Check for PD tremor frequency
        if pd_tremor_power / total_power > 0.25 and dominant_freq >= 4:
            return 'tremor'
        
        # Check for action/essential tremor
        if action_tremor_power / total_power > 0.25 and dominant_freq >= 6:
            return 'tremor'
        
        # If there's movement but no clear peaks, likely low-frequency tremor
        if len(peaks) < 2 and signal_variance > 0.0001:
            if (pd_tremor_power + action_tremor_power) / total_power > 0.15:
                return 'tremor'
        
        # DEFAULT: If movement exists but doesn't fit clear patterns
        # Use frequency to make final decision
        if signal_variance > 0.00005:
            # Most power in tapping range = likely slow/irregular tapping
            if tapping_power / total_power > 0.5:
                return 'tapping'
            # Otherwise consider it tremor
            else:
                return 'tremor'
        
        return 'static'
    
    def _calculate_pattern_confidence(self, distances: np.ndarray, pattern_type: str) -> float:
        """
        Calculate confidence score for detected pattern
        
        Args:
            distances: Array of thumb-index distances
            pattern_type: Detected movement type
            
        Returns:
            Confidence score 0-1
        """
        if pattern_type == 'tapping':
            # Check regularity of peaks
            peaks, _ = signal.find_peaks(distances, prominence=np.std(distances)*0.5)
            if len(peaks) > 2:
                intervals = np.diff(peaks)
                cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1
                return max(0, 1 - cv)  # More regular = higher confidence
            return 0.5
            
        elif pattern_type == 'tremor':
            # Check power concentration in tremor band
            distances_detrended = signal.detrend(distances)
            freqs, psd = signal.welch(distances_detrended, self.fps, nperseg=min(64, len(distances)))
            tremor_band = (freqs >= 4) & (freqs <= 6)
            return np.sum(psd[tremor_band]) / np.sum(psd)
            
        else:  # static
            # Check variance (lower = more static)
            return 1 - min(1, np.std(distances) * 10)
    
    def analyze_simultaneous_movements(self, start_frame: int = None, end_frame: int = None) -> Dict:
        """
        Analyze simultaneous different movements between hands
        Specifically for detecting one hand tapping while other is tremoring
        
        Args:
            start_frame: Optional start frame for analysis
            end_frame: Optional end frame for analysis
            
        Returns:
            Dictionary with differential movement analysis
        """
        results = {}
        
        # Get movement classifications
        movement_types = self.detect_movement_type_per_hand()
        
        # Filter by frame range if specified
        if start_frame is not None or end_frame is not None:
            for hand in ['left', 'right']:
                movement_types[hand] = [
                    m for m in movement_types[hand]
                    if (start_frame is None or m['start_frame'] >= start_frame) and
                       (end_frame is None or m['end_frame'] <= end_frame)
                ]
        
        # Identify simultaneous different movements
        differential_periods = []
        
        for left_window in movement_types.get('left', []):
            for right_window in movement_types.get('right', []):
                # Check for temporal overlap
                overlap_start = max(left_window['start_frame'], right_window['start_frame'])
                overlap_end = min(left_window['end_frame'], right_window['end_frame'])
                
                if overlap_start < overlap_end:
                    # Found overlapping period
                    if left_window['movement_type'] != right_window['movement_type']:
                        differential_periods.append({
                            'start_frame': overlap_start,
                            'end_frame': overlap_end,
                            'left_movement': left_window['movement_type'],
                            'right_movement': right_window['movement_type'],
                            'duration_seconds': (overlap_end - overlap_start) / self.fps
                        })
        
        # Analyze specific tapping vs tremor scenario
        tapping_tremor_periods = [
            p for p in differential_periods
            if ('tapping' in [p['left_movement'], p['right_movement']] and
                'tremor' in [p['left_movement'], p['right_movement']])
        ]
        
        if tapping_tremor_periods:
            # Detailed analysis of tapping vs tremor
            period = tapping_tremor_periods[0]  # Analyze first occurrence
            
            # Determine which hand is doing what
            if period['left_movement'] == 'tapping':
                tapping_hand = 'left'
                tremor_hand = 'right'
            else:
                tapping_hand = 'right'
                tremor_hand = 'left'
            
            # Get detailed metrics for each hand during this period
            frame_range = (period['start_frame'], period['end_frame'])
            
            # Analyze tapping performance
            tapping_metrics = self._analyze_tapping_in_range(tapping_hand, frame_range)
            
            # Analyze tremor characteristics
            tremor_metrics = self._analyze_tremor_in_range(tremor_hand, frame_range)
            
            # Calculate independence/coupling
            coupling_score = self._calculate_movement_coupling(tapping_hand, tremor_hand, frame_range)
            
            results['differential_movement'] = {
                'detected': True,
                'tapping_hand': tapping_hand,
                'tremor_hand': tremor_hand,
                'duration': period['duration_seconds'],
                'tapping_metrics': tapping_metrics,
                'tremor_metrics': tremor_metrics,
                'movement_independence': 1 - coupling_score,  # Higher = more independent
                'clinical_significance': self._assess_clinical_significance(
                    tapping_metrics, tremor_metrics, coupling_score
                )
            }
        else:
            results['differential_movement'] = {
                'detected': False,
                'note': 'No simultaneous tapping-tremor pattern detected'
            }
        
        # Summary statistics
        results['movement_summary'] = {
            'left_hand_patterns': self._summarize_patterns(movement_types.get('left', [])),
            'right_hand_patterns': self._summarize_patterns(movement_types.get('right', [])),
            'differential_periods_count': len(differential_periods),
            'tapping_tremor_periods_count': len(tapping_tremor_periods)
        }
        
        return results
    
    def _analyze_tapping_in_range(self, hand: str, frame_range: Tuple[int, int]) -> Dict:
        """Analyze tapping metrics within specific frame range"""
        hand_data = self.left_hand if hand == 'left' else self.right_hand
        
        # Filter data to frame range
        mask = (hand_data['frame'] >= frame_range[0]) & (hand_data['frame'] <= frame_range[1])
        filtered_data = hand_data[mask]
        
        if len(filtered_data) == 0:
            return {}
        
        # Get thumb-index distance
        thumb = filtered_data[filtered_data['landmark_id'] == self.THUMB_TIP].sort_values('frame')
        index = filtered_data[filtered_data['landmark_id'] == self.INDEX_TIP].sort_values('frame')
        merged = pd.merge(thumb, index, on='frame', suffixes=('_thumb', '_index'))
        
        distances = np.sqrt(
            (merged['x_thumb'] - merged['x_index'])**2 +
            (merged['y_thumb'] - merged['y_index'])**2 +
            (merged['z_thumb'] - merged['z_index'])**2
        ).values
        
        # Detect taps
        peaks, properties = signal.find_peaks(distances, prominence=np.std(distances)*0.5)
        
        if len(peaks) > 1:
            tap_intervals = np.diff(peaks) / self.fps
            tap_frequency = 1.0 / np.mean(tap_intervals)
            tap_regularity = 1 - (np.std(tap_intervals) / np.mean(tap_intervals))
        else:
            tap_frequency = 0
            tap_regularity = 0
        
        return {
            'tap_count': len(peaks),
            'tap_frequency': tap_frequency,
            'tap_regularity': tap_regularity,
            'mean_amplitude': np.mean(distances[peaks]) if len(peaks) > 0 else 0
        }
    
    def _analyze_tremor_in_range(self, hand: str, frame_range: Tuple[int, int]) -> Dict:
        """Analyze tremor metrics within specific frame range"""
        hand_data = self.left_hand if hand == 'left' else self.right_hand
        
        # Filter data to frame range
        mask = (hand_data['frame'] >= frame_range[0]) & (hand_data['frame'] <= frame_range[1])
        filtered_data = hand_data[mask]
        
        if len(filtered_data) == 0:
            return {}
        
        # Analyze index finger tremor
        index_data = filtered_data[filtered_data['landmark_id'] == self.INDEX_TIP].sort_values('frame')
        
        if len(index_data) < 30:  # Need minimum data for FFT
            return {}
        
        # Detrend positions
        x_detrended = signal.detrend(index_data['x'].values)
        y_detrended = signal.detrend(index_data['y'].values)
        z_detrended = signal.detrend(index_data['z'].values)
        
        # Calculate power spectrum
        freqs, psd_x = signal.welch(x_detrended, self.fps, nperseg=min(64, len(x_detrended)))
        _, psd_y = signal.welch(y_detrended, self.fps, nperseg=min(64, len(y_detrended)))
        _, psd_z = signal.welch(z_detrended, self.fps, nperseg=min(64, len(z_detrended)))
        
        combined_psd = psd_x + psd_y + psd_z
        
        # Find peak in tremor range
        tremor_band = (freqs >= 3) & (freqs <= 12)
        peak_idx = np.argmax(combined_psd[tremor_band])
        peak_freq = freqs[tremor_band][peak_idx]
        peak_power = combined_psd[tremor_band][peak_idx]
        
        # Tremor amplitude
        tremor_amplitude = np.sqrt(np.mean(x_detrended**2 + y_detrended**2 + z_detrended**2))
        
        return {
            'peak_frequency': peak_freq,
            'peak_power': peak_power,
            'tremor_amplitude': tremor_amplitude,
            'in_pd_range': 4 <= peak_freq <= 6
        }
    
    def _calculate_movement_coupling(self, hand1: str, hand2: str, 
                                    frame_range: Tuple[int, int]) -> float:
        """
        Calculate coupling between two hands' movements
        Lower coupling suggests more independent control
        
        Returns:
            Coupling score 0-1 (0 = independent, 1 = coupled)
        """
        data1 = self.left_hand if hand1 == 'left' else self.right_hand
        data2 = self.left_hand if hand2 == 'left' else self.right_hand
        
        # Filter to frame range and get index finger data
        mask1 = (data1['frame'] >= frame_range[0]) & (data1['frame'] <= frame_range[1])
        mask2 = (data2['frame'] >= frame_range[0]) & (data2['frame'] <= frame_range[1])
        
        index1 = data1[mask1 & (data1['landmark_id'] == self.INDEX_TIP)].sort_values('frame')
        index2 = data2[mask2 & (data2['landmark_id'] == self.INDEX_TIP)].sort_values('frame')
        
        # Merge on common frames
        merged = pd.merge(index1, index2, on='frame', suffixes=('_1', '_2'))
        
        if len(merged) < 10:
            return 0.5  # Default to moderate coupling if insufficient data
        
        # Calculate cross-correlation
        x1 = signal.detrend(merged['x_1'].values)
        x2 = signal.detrend(merged['x_2'].values)
        
        # Normalize signals
        x1 = (x1 - np.mean(x1)) / (np.std(x1) + 1e-10)
        x2 = (x2 - np.mean(x2)) / (np.std(x2) + 1e-10)
        
        # Calculate maximum cross-correlation
        correlation = signal.correlate(x1, x2, mode='same')
        max_corr = np.max(np.abs(correlation)) / len(x1)
        
        return min(1, max_corr)
    
    def _assess_clinical_significance(self, tapping_metrics: Dict, 
                                     tremor_metrics: Dict, 
                                     coupling_score: float) -> Dict:
        """Assess clinical significance of differential movement pattern"""
        significance = {
            'asymmetry_detected': True,
            'indicators': []
        }
        
        # Check tapping performance
        if tapping_metrics.get('tap_frequency', 0) < 2:
            significance['indicators'].append('slow_tapping')
        if tapping_metrics.get('tap_regularity', 0) < 0.7:
            significance['indicators'].append('irregular_tapping')
        
        # Check tremor characteristics
        if tremor_metrics.get('in_pd_range', False):
            significance['indicators'].append('pd_tremor_frequency')
        if tremor_metrics.get('tremor_amplitude', 0) > 0.01:
            significance['indicators'].append('significant_tremor')
        
        # Check movement independence
        if coupling_score < 0.3:
            significance['indicators'].append('independent_hand_control')
        else:
            significance['indicators'].append('coupled_movements')
        
        # Overall assessment
        significance['severity'] = len(significance['indicators'])
        significance['interpretation'] = self._interpret_findings(significance['indicators'])
        
        return significance
    
    def _interpret_findings(self, indicators: List[str]) -> str:
        """Provide clinical interpretation of findings"""
        if 'pd_tremor_frequency' in indicators and 'slow_tapping' in indicators:
            return "Pattern consistent with Parkinson's disease: tremor in typical frequency range with bradykinesia"
        elif 'independent_hand_control' in indicators:
            return "Good differential hand control maintained"
        elif 'coupled_movements' in indicators:
            return "Movements appear coupled, suggesting difficulty with independent hand control"
        else:
            return "Mixed findings requiring further clinical assessment"
    
    def _summarize_patterns(self, patterns: List[Dict]) -> Dict:
        """Summarize movement patterns detected"""
        if not patterns:
            return {'no_data': True}
        
        total_duration = sum(
            (p['end_frame'] - p['start_frame']) / self.fps 
            for p in patterns
        )
        
        pattern_counts = {}
        for p in patterns:
            pattern_type = p['movement_type']
            if pattern_type not in pattern_counts:
                pattern_counts[pattern_type] = 0
            pattern_counts[pattern_type] += 1
        
        return {
            'total_duration': total_duration,
            'pattern_counts': pattern_counts,
            'dominant_pattern': max(pattern_counts, key=pattern_counts.get) if pattern_counts else None
        }
            
    def generate_clinical_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive clinical assessment report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*60)
        print("PARKINSON'S DISEASE MOVEMENT ANALYSIS REPORT")
        print("="*60)
        
        # Perform all analyses
        tremor = self.calculate_tremor_frequency('both')
        tapping = self.analyze_finger_tapping('both')
        asymmetry = self.calculate_movement_asymmetry()
        bradykinesia = self.detect_bradykinesia('both')
        simultaneous = self.analyze_simultaneous_movements()
        
        report = {
            'tremor_analysis': tremor,
            'tapping_analysis': tapping,
            'asymmetry_analysis': asymmetry,
            'bradykinesia_analysis': bradykinesia,
            'differential_movement_analysis': simultaneous
        }
        
        # Print summary
        print("\n1. TREMOR ANALYSIS")
        print("-" * 40)
        for hand in ['left', 'right']:
            if hand in tremor:
                t = tremor[hand]
                print(f"{hand.upper()} HAND:")
                print(f"  Peak frequency: {t['peak_frequency']:.2f} Hz")
                print(f"  Tremor detected: {t['has_tremor']}")
                if t['has_tremor']:
                    print(f"  Tremor type: {t['tremor_type']}")
                print(f"  Amplitude: {t['tremor_amplitude']:.4f}")
                
        print("\n2. FINGER TAPPING ANALYSIS")
        print("-" * 40)
        for hand in ['left', 'right']:
            if hand in tapping:
                t = tapping[hand]
                print(f"{hand.upper()} HAND:")
                print(f"  Tap count: {t['tap_count']}")
                print(f"  Frequency: {t['tap_frequency']:.2f} Hz")
                print(f"  Amplitude decrement: {t['amplitude_decrement']:.1f}%")
                print(f"  Rhythm CV: {t['rhythm_cv']:.2f}")
                print(f"  Hesitations: {t['hesitations']}")
                
        print("\n3. MOVEMENT ASYMMETRY")
        print("-" * 40)
        if asymmetry.get('has_both_hands'):
            print(f"  Tremor asymmetry: {asymmetry.get('tremor_asymmetry', 0):.2%}")
            print(f"  Tapping asymmetry: {asymmetry.get('tapping_asymmetry', 0):.2%}")
            print(f"  More affected side: {asymmetry.get('more_affected', 'unknown')}")
            
        print("\n4. BRADYKINESIA ASSESSMENT")
        print("-" * 40)
        for hand in ['left', 'right']:
            if hand in bradykinesia:
                b = bradykinesia[hand]
                print(f"{hand.upper()} HAND:")
                print(f"  Score: {b['bradykinesia_score']}/5")
                print(f"  Severity: {b['severity']}")
                if b['indicators']:
                    print(f"  Indicators: {', '.join(b['indicators'])}")
        
        print("\n5. DIFFERENTIAL MOVEMENT ANALYSIS")
        print("-" * 40)
        if simultaneous.get('differential_movement', {}).get('detected'):
            diff = simultaneous['differential_movement']
            print(f"  Tapping hand: {diff['tapping_hand'].upper()}")
            print(f"  Tremor hand: {diff['tremor_hand'].upper()}")
            print(f"  Duration: {diff['duration']:.2f} seconds")
            print(f"  Movement independence: {diff['movement_independence']:.2%}")
            
            # Tapping metrics
            if 'tapping_metrics' in diff and diff['tapping_metrics']:
                print(f"\n  Tapping Performance:")
                print(f"    - Frequency: {diff['tapping_metrics'].get('tap_frequency', 0):.2f} Hz")
                print(f"    - Tap count: {diff['tapping_metrics'].get('tap_count', 0)}")
                print(f"    - Regularity: {diff['tapping_metrics'].get('tap_regularity', 0):.2%}")
            
            # Tremor metrics
            if 'tremor_metrics' in diff and diff['tremor_metrics']:
                print(f"\n  Tremor Characteristics:")
                print(f"    - Peak frequency: {diff['tremor_metrics'].get('peak_frequency', 0):.2f} Hz")
                print(f"    - In PD range (4-6 Hz): {diff['tremor_metrics'].get('in_pd_range', False)}")
                print(f"    - Amplitude: {diff['tremor_metrics'].get('tremor_amplitude', 0):.4f}")
            
            # Clinical interpretation
            if 'clinical_significance' in diff:
                sig = diff['clinical_significance']
                print(f"\n  Clinical Significance:")
                print(f"    - {sig.get('interpretation', 'No interpretation available')}")
                if sig.get('indicators'):
                    print(f"    - Key findings: {', '.join(sig['indicators'])}")
        else:
            print("  No simultaneous tapping-tremor pattern detected")
            
        # Movement summary
        if 'movement_summary' in simultaneous:
            summary = simultaneous['movement_summary']
            print(f"\n  Overall Movement Patterns:")
            for hand in ['left', 'right']:
                pattern_key = f'{hand}_hand_patterns'
                if pattern_key in summary and not summary[pattern_key].get('no_data'):
                    patterns = summary[pattern_key]
                    print(f"    {hand.upper()} hand:")
                    if 'pattern_counts' in patterns:
                        for ptype, count in patterns['pattern_counts'].items():
                            print(f"      - {ptype}: {count} windows")
                    
        print("\n" + "="*60)
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {output_path}")
            
        return report
    
    def visualize_analysis(self, save_path: str = None):
        """
        Create comprehensive visualization of PD analysis
        
        Args:
            save_path: Optional path to save figure
        """
        tremor = self.calculate_tremor_frequency('both')
        tapping = self.analyze_finger_tapping('both')
        
        fig = plt.figure(figsize=(16, 12))
        
        # Tremor frequency spectrum
        for i, hand in enumerate(['left', 'right']):
            if hand in tremor:
                ax = plt.subplot(3, 2, i+1)
                t = tremor[hand]
                
                # Plot PSD
                ax.plot(t['frequencies'], t['psd'], 'b-', alpha=0.7)
                ax.axvline(t['peak_frequency'], color='r', linestyle='--', 
                          label=f'Peak: {t["peak_frequency"]:.2f} Hz')
                
                # Highlight PD frequency ranges
                ax.axvspan(4, 6, alpha=0.2, color='red', label='PD resting (4-6 Hz)')
                ax.axvspan(6, 12, alpha=0.2, color='orange', label='PD action (6-12 Hz)')
                
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density')
                ax.set_title(f'{hand.upper()} Hand - Tremor Spectrum')
                ax.set_xlim([0, 20])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        # Finger tapping patterns
        for i, hand in enumerate(['left', 'right']):
            if hand in tapping:
                ax = plt.subplot(3, 2, i+3)
                t = tapping[hand]
                
                if len(t['distances']) > 0:
                    time = np.arange(len(t['distances'])) / self.fps
                    ax.plot(time, t['distances'], 'b-', alpha=0.5)
                    
                    # Mark peaks and valleys
                    if len(t['peaks']) > 0:
                        ax.plot(time[t['peaks']], t['distances'][t['peaks']], 
                               'ro', markersize=8, label='Tap peaks')
                    if len(t['valleys']) > 0:
                        ax.plot(time[t['valleys']], t['distances'][t['valleys']], 
                               'go', markersize=6, label='Tap valleys')
                    
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Thumb-Index Distance')
                    ax.set_title(f'{hand.upper()} Hand - Finger Tapping Pattern')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
        # Amplitude progression
        for i, hand in enumerate(['left', 'right']):
            if hand in tapping:
                ax = plt.subplot(3, 2, i+5)
                t = tapping[hand]
                
                if len(t['tap_amplitudes']) > 0:
                    tap_numbers = np.arange(len(t['tap_amplitudes']))
                    ax.plot(tap_numbers, t['tap_amplitudes'], 'bo-', markersize=8)
                    
                    # Fit trend line
                    if len(tap_numbers) > 1:
                        z = np.polyfit(tap_numbers, t['tap_amplitudes'], 1)
                        p = np.poly1d(z)
                        ax.plot(tap_numbers, p(tap_numbers), 'r--', 
                               label=f'Decrement: {t["amplitude_decrement"]:.1f}%')
                    
                    ax.set_xlabel('Tap Number')
                    ax.set_ylabel('Tap Amplitude')
                    ax.set_title(f'{hand.upper()} Hand - Amplitude Progression')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
        plt.suptitle('Parkinson\'s Disease Movement Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        
def main():
    """Example usage of PD analyzer"""
    # Initialize analyzer
    analyzer = PDMovementAnalyzer('/home/yl2428/hand-land-mark/output_holistic/Y00078.1_trimmed_1080p_landmarks.csv')
    
    # Generate clinical report
    report = analyzer.generate_clinical_report('pd_analysis_report.json')
    
    # Create visualizations
    analyzer.visualize_analysis('pd_analysis_visualization.png')
    
    return report


if __name__ == "__main__":
    report = main()