#!/usr/bin/env python
"""
Main script for running complete PD movement analysis on a video
"""
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pd_analysis import PDMovementAnalyzer
from visualize_differential_patterns import create_differential_visualization
from visualize_movements import create_movement_visualization
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Run PD movement analysis on landmark data')
    parser.add_argument('csv_path', help='Path to landmarks CSV file')
    parser.add_argument('--fps', type=float, default=119.0, help='Video frame rate')
    parser.add_argument('--output-dir', default='analysis_outputs', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading landmark data from: {args.csv_path}")
    analyzer = PDMovementAnalyzer(args.csv_path, fps=args.fps)
    
    # Generate clinical report
    report_path = os.path.join(args.output_dir, 'pd_clinical_report.json')
    print("\nGenerating clinical report...")
    report = analyzer.generate_clinical_report(report_path)
    
    # Analyze simultaneous movements
    print("\nAnalyzing differential movements...")
    simultaneous = analyzer.analyze_simultaneous_movements()
    
    if simultaneous.get('differential_movement', {}).get('detected'):
        diff = simultaneous['differential_movement']
        print(f"\n✓ Differential movement detected!")
        print(f"  - {diff['tapping_hand'].upper()} hand: Tapping")
        print(f"  - {diff['tremor_hand'].upper()} hand: Tremor")
        print(f"  - Duration: {diff['duration']:.2f} seconds")
        print(f"  - Independence: {diff['movement_independence']:.1%}")
    else:
        print("\nNo clear tapping vs tremor differential detected")
        summary = simultaneous.get('movement_summary', {})
        if summary:
            print("\nMovement patterns detected:")
            for hand in ['left', 'right']:
                patterns = summary.get(f'{hand}_hand_patterns', {})
                if patterns and not patterns.get('no_data'):
                    print(f"  {hand.upper()} hand: {patterns.get('dominant_pattern', 'mixed')}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        viz_path = os.path.join(args.output_dir, 'pd_visualization.png')
        analyzer.visualize_analysis(viz_path)
        print(f"Visualizations saved to: {args.output_dir}")
    
    print("\n✓ Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    return report


if __name__ == "__main__":
    main()