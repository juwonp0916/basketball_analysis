"""
Video Shot Localization Tool

Automated shot detection and localization system for basketball game videos.
Combines shot detection with homography-based court localization to track all shots
throughout a video and generate comprehensive shot charts.

Usage:
    python video_shot_localization.py --video path/to/video.mp4

Features:
- Interactive 6-point FIBA court calibration
- Automatic shot detection (makes and misses)
- Homography transformation to court coordinates
- Zone classification (1-9) and shot type (2PT/3PT)
- Cumulative shot visualization on court diagram
- JSON export with complete shot data
- Hybrid shooter detection (shoot class + IoU fallback)
"""

import cv2
import numpy as np
import sys
import argparse
import json
from pathlib import Path

# Add backend path
sys.path.insert(0, 'backend/score_detection')

from shot_detector import ShotDetector
from constants import CALIBRATION_LABELS


class VideoShotLocalizer:
    def __init__(self, video_path, show_vid=False, auto_calibrate=False, save_debug_frames=False):
        """
        Initialize video shot localizer

        Args:
            video_path: Path to input video
            show_vid: Show CV2 window during processing (slows down processing)
            auto_calibrate: Automatically use existing calibration without prompting
            save_debug_frames: Save frames when shoot class is detected for debugging
        """
        self.video_path = video_path
        self.show_vid = show_vid
        self.auto_calibrate = auto_calibrate
        self.save_debug_frames = save_debug_frames

        # Load first frame for calibration
        print(f"\nLoading video: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        ret, self.first_frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read first frame from video")

        self.height, self.width = self.first_frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Duration: {self.duration:.1f} seconds ({self.total_frames} frames)")

        # Close video (will be reopened by ShotDetector)
        self.cap.release()

        # Calibration
        self.calibration_points = None
        self.calibration_file = 'court_calibration_6pt.json'

    def load_or_calibrate(self):
        """Load existing calibration or run interactive calibration"""
        if Path(self.calibration_file).exists():
            print(f"\n{'='*60}")
            print(f"Found existing calibration: {self.calibration_file}")

            # Auto-use existing calibration if flag is set
            if self.auto_calibrate:
                response = 'y'
                print("Auto-using existing calibration (--auto-calibrate flag)")
            else:
                response = input("Use existing calibration? (y/n): ").lower().strip()

            if response in ['y', 'yes']:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)

                    # Validate it's a 6-point calibration
                    if len(data['points']) != 6:
                        print(f"ERROR: Calibration file has {len(data['points'])} points, expected 6")
                        print("Running new calibration...")
                    else:
                        self.calibration_points = data['points']
                        print("Loaded 6-point calibration:")
                        for i, (point, label) in enumerate(zip(self.calibration_points, CALIBRATION_LABELS)):
                            print(f"  {i+1}. {label}: ({point[0]:.1f}, {point[1]:.1f})")
                        return

        # Run interactive calibration
        print("\n" + "="*60)
        print("6-POINT FIBA COURT CALIBRATION")
        print("="*60)
        self.calibrate_court()

    def calibrate_court(self):
        """Interactive 6-point court calibration"""
        points = []
        display_frame = self.first_frame.copy()
        window_name = "Court Calibration - Click 6 Points"

        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display_frame

            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append([x, y])
                print(f"Point {len(points)}: {CALIBRATION_LABELS[len(points)-1]} = ({x}, {y})")

                # Redraw
                display_frame = self.first_frame.copy()
                self._draw_calibration_points(display_frame, points)
                cv2.imshow(window_name, display_frame)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\nClick 6 court reference points in order:")
        for i, label in enumerate(CALIBRATION_LABELS, 1):
            print(f"  {i}. {label}")
        print("\nPress ENTER when done, 'R' to reset")

        self._draw_calibration_points(display_frame, points)
        cv2.imshow(window_name, display_frame)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(points) == 6:  # Enter
                break
            elif key == ord('r') or key == ord('R'):
                points = []
                display_frame = self.first_frame.copy()
                self._draw_calibration_points(display_frame, points)
                cv2.imshow(window_name, display_frame)
                print("\nReset. Click 6 points again.")

        cv2.destroyAllWindows()

        self.calibration_points = points

        # Save calibration
        calibration_data = {
            'points': [[float(p[0]), float(p[1])] for p in points],
            'labels': CALIBRATION_LABELS,
            'image_dimensions': [self.width, self.height],
            'court_type': 'FIBA_half_court_6pt'
        }

        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"\n✓ Calibration saved to: {self.calibration_file}")

    def _draw_calibration_points(self, frame, points):
        """Draw calibration points and instructions"""
        colors = [
            (0, 255, 255),  # Yellow - Baseline L Sideline
            (0, 255, 0),    # Green - Baseline L Penalty
            (0, 255, 0),    # Green - Baseline R Penalty
            (0, 255, 255),  # Yellow - Baseline R Sideline
            (255, 0, 255),  # Magenta - FT L
            (255, 0, 255),  # Magenta - FT R
        ]

        # Draw points
        for i, point in enumerate(points):
            color = colors[i]
            cv2.circle(frame, tuple(point), 10, (255, 255, 255), -1)  # White border
            cv2.circle(frame, tuple(point), 8, color, -1)  # Colored center

            # Label
            label_short = f"{i+1}"
            cv2.putText(frame, label_short, (point[0] + 15, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, label_short, (point[0] + 15, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw baseline (points 0-3)
        if len(points) >= 2:
            for i in range(min(len(points), 4) - 1):
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), (0, 255, 255), 2)

        # Draw FT line (points 4-5)
        if len(points) == 6:
            cv2.line(frame, tuple(points[4]), tuple(points[5]), (255, 0, 255), 2)

            # Draw penalty box sides
            cv2.line(frame, tuple(points[1]), tuple(points[4]), (0, 255, 0), 2)
            cv2.line(frame, tuple(points[2]), tuple(points[5]), (0, 255, 0), 2)

        # Instructions overlay
        instructions = [
            "Click 6 court calibration points:",
            f"Progress: {len(points)}/6",
            "",
            "BASELINE (4 points): L Sideline, L Penalty, R Penalty, R Sideline",
            "FREE THROW (2 points): L Penalty, R Penalty",
            "",
            "ENTER: Confirm | R: Reset"
        ]

        y_offset = 30
        for instruction in instructions:
            # Background rectangle for readability
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (5, y_offset - 20), (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)

            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

    def run(self):
        """Main pipeline execution"""
        print("\n" + "="*60)
        print("VIDEO SHOT LOCALIZATION")
        print("="*60)

        try:
            # Step 1: Calibration
            self.load_or_calibrate()

            if not self.calibration_points:
                print("ERROR: Calibration failed")
                return

            # Step 2: Run shot detection with localization
            print("\n" + "="*60)
            print("PROCESSING VIDEO")
            print("="*60)
            print("Running shot detection and localization...")
            print("This may take a few minutes depending on video length.")
            print()

            # Callbacks
            def on_shot_detected(timestamp, success, video_id, shot_location, court_position):
                """Called when a shot is detected"""
                pass  # Logging is handled in ShotDetector

            def on_complete():
                """Called when processing completes"""
                print("\n" + "="*60)
                print("PROCESSING COMPLETE")
                print("="*60)

            # Run detector
            detector = ShotDetector(
                video_path=self.video_path,
                on_detect=on_shot_detected,
                on_complete=on_complete,
                calibration_points=self.calibration_points,
                enable_localization=True,
                show_vid=self.show_vid,
                video_id=1,
                save_debug_frames=self.save_debug_frames
            )

            print("\n✓ All done!")
            print("\nCheck the 'output/' directory for:")
            print(f"  - {Path(self.video_path).stem}_shots_data.json (shot data)")
            print(f"  - {Path(self.video_path).stem}_shot_chart.png (final shot chart)")
            print("\nCheck 'shot_visualizations/' for progressive shot images")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Video Shot Localization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_shot_localization.py --video game_footage.mp4
  python video_shot_localization.py --video game.mp4 --show-vid

Output:
  - output/{video}_shots_data.json: Complete shot data with timestamps, locations, zones
  - output/{video}_shot_chart.png: Final shot chart visualization
  - shot_visualizations/: Progressive shot images (shot_001.png, shot_002.png, ...)
        """
    )

    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--show-vid', action='store_true',
                       help='Show video window during processing (slower)')
    parser.add_argument('--auto-calibrate', action='store_true',
                       help='Automatically use existing calibration without prompting')
    parser.add_argument('--debug-frames', action='store_true',
                       help='Save frames when shoot class is detected (for debugging shot locations)')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        return

    # Run localizer
    localizer = VideoShotLocalizer(
        args.video,
        show_vid=args.show_vid,
        auto_calibrate=args.auto_calibrate,
        save_debug_frames=args.debug_frames
    )
    localizer.run()


if __name__ == "__main__":
    main()
