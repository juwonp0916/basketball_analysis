"""
Static Image Shot Localization Tool

Simplified shot localization system for testing and validation on single images.
Uses homography transformation to map player positions from video to court coordinates.

Usage:
    python static_shot_localization.py --image path/to/image.jpg

Features:
- Interactive penalty box calibration
- Automatic shooter detection (using ball-person overlap)
- Homography transformation to court coordinates
- Zone classification (1-9)
- Side-by-side visualization
- JSON output with metadata
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import argparse
import json
from pathlib import Path

# Add backend path
sys.path.insert(0, 'backend/score_detection')

from localization import ShotLocalizer
from statistics import TeamStatistics
from constants import MAP_WIDTH, MAP_HEIGHT

class StaticShotLocalizer:
    def __init__(self, image_path, model_path, court_type='half_court'):
        """
        Initialize static shot localizer

        Args:
            image_path: Path to input image
            model_path: Path to YOLO model weights
            court_type: 'half_court' or 'full_court'
        """
        self.image_path = image_path
        self.model_path = model_path
        self.court_type = court_type

        # Load image
        print(f"\nLoading image: {image_path}")
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.height, self.width = self.image.shape[:2]
        print(f"Resolution: {self.width}x{self.height}")

        # Load YOLO model
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path, verbose=False)
        print("Model loaded successfully")

        # Court configuration
        self.MAP_WIDTH = MAP_WIDTH
        self.MAP_HEIGHT = 140 if court_type == 'half_court' else 280
        self.court_img_path = f'backend/score_detection/court_img_{"halfcourt" if court_type == "half_court" else ""}.png'

        # State variables
        self.calibration_points = None  # Changed from penalty_box_points
        self.calibration_file = 'court_calibration_6pt.json'
        self.ball_box = None
        self.shooter_box = None
        self.shooter_position = None
        self.court_position = None
        self.zone = None
        self.detections = None

    def load_or_calibrate(self):
        """Load existing calibration or run interactive calibration"""
        if Path(self.calibration_file).exists():
            print(f"\nFound existing calibration: {self.calibration_file}")
            response = input("Use existing calibration? (y/n): ").lower().strip()

            if response in ['y', 'yes']:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)

                    # Validate it's a 6-point calibration
                    if len(data['points']) != 6:
                        print(f"ERROR: Calibration file has {len(data['points'])} points, expected 6")
                        print("Running new calibration...")
                    else:
                        self.calibration_points = np.array(data['points'], dtype=np.float32)
                        print("Loaded 6-point calibration:")
                        sys.path.insert(0, 'backend/score_detection')
                        from constants import CALIBRATION_LABELS
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
        sys.path.insert(0, 'backend/score_detection')
        from constants import CALIBRATION_LABELS

        points = []
        display_frame = self.image.copy()
        window_name = "Court Calibration - Click 6 Points"

        def mouse_callback(event, x, y, flags, param):
            nonlocal points, display_frame

            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append([x, y])
                print(f"Point {len(points)}: {CALIBRATION_LABELS[len(points)-1]} = ({x}, {y})")

                # Redraw
                display_frame = self.image.copy()
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
                display_frame = self.image.copy()
                self._draw_calibration_points(display_frame, points)
                cv2.imshow(window_name, display_frame)
                print("\nReset. Click 6 points again.")

        cv2.destroyAllWindows()

        self.calibration_points = np.array(points, dtype=np.float32)

        # Validate: Check that baseline points are roughly collinear
        # and FT line points are roughly collinear
        self._validate_calibration()

        # Save calibration
        calibration_data = {
            'points': [[float(p[0]), float(p[1])] for p in points],
            'labels': CALIBRATION_LABELS,
            'image_dimensions': [self.width, self.height],
            'court_type': 'FIBA_half_court_6pt'
        }

        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"\nCalibration saved to: {self.calibration_file}")

    def _validate_calibration(self):
        """Validate that calibration points form reasonable geometry"""
        if self.calibration_points is None or len(self.calibration_points) != 6:
            return False

        # Check baseline points (0-3) are roughly collinear
        baseline_pts = self.calibration_points[0:4]
        # Simple check: all Y coordinates should be similar
        y_coords = baseline_pts[:, 1]
        y_variance = np.var(y_coords)

        if y_variance > 100:  # Threshold in pixels squared
            print(f"WARNING: Baseline points may not be collinear (Y variance: {y_variance:.1f})")

        # Check FT line points (4-5) form a horizontal line
        ft_pts = self.calibration_points[4:6]
        ft_y_diff = abs(ft_pts[1, 1] - ft_pts[0, 1])

        if ft_y_diff > 10:  # Threshold in pixels
            print(f"WARNING: Free throw line points may not be aligned (Y difference: {ft_y_diff:.1f})")

        # Check parallel constraint: baseline and FT line should have similar slopes
        # Both should be approximately horizontal
        baseline_slope = (baseline_pts[3, 1] - baseline_pts[0, 1]) / (baseline_pts[3, 0] - baseline_pts[0, 0] + 1e-6)
        ft_slope = (ft_pts[1, 1] - ft_pts[0, 1]) / (ft_pts[1, 0] - ft_pts[0, 0] + 1e-6)

        slope_diff = abs(baseline_slope - ft_slope)
        if slope_diff > 0.1:
            print(f"WARNING: Baseline and FT line may not be parallel (slope difference: {slope_diff:.3f})")

        print("✓ Calibration geometry validation passed")
        return True

    def _draw_calibration_points(self, frame, points):
        """Draw calibration points and instructions"""
        sys.path.insert(0, 'backend/score_detection')
        from constants import CALIBRATION_LABELS

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

    def detect_objects(self):
        """Run YOLO detection to find ball and people"""
        print("\n" + "="*60)
        print("OBJECT DETECTION")
        print("="*60)
        print("Running YOLO model...")

        results = self.model(self.image, verbose=False)[0]
        self.detections = results

        ball_boxes = []
        person_boxes = []

        # Extract detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = xyxy
            class_name = results.names[cls]

            box_data = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class': class_name,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'bottom_center': ((x1 + x2) / 2, y2)
            }

            if class_name == 'ball' and conf > 0.3:
                ball_boxes.append(box_data)
            elif class_name == 'person' and conf > 0.5:
                person_boxes.append(box_data)

        print(f"\nDetections:")
        print(f"  Ball: {len(ball_boxes)}")
        print(f"  Person: {len(person_boxes)}")

        # Select best ball detection
        if ball_boxes:
            self.ball_box = max(ball_boxes, key=lambda x: x['confidence'])
            print(f"\nBest ball detection: confidence={self.ball_box['confidence']:.2f}")
        else:
            print("\nWARNING: No ball detected!")

        return ball_boxes, person_boxes

    def select_shooter(self, ball_boxes, person_boxes):
        """Auto-select shooter as person with most overlap with ball"""
        print("\n" + "="*60)
        print("SHOOTER DETECTION")
        print("="*60)

        if not self.ball_box:
            print("ERROR: No ball detected, cannot determine shooter")
            return None

        if not person_boxes:
            print("ERROR: No people detected")
            return None

        print("Finding person with most ball overlap...")

        # Calculate IoU for each person
        best_person = None
        best_iou = 0

        ball_bbox = self.ball_box['bbox']

        for i, person in enumerate(person_boxes):
            iou = self._calculate_iou(ball_bbox, person['bbox'])
            print(f"  Person {i+1}: IoU = {iou:.3f}, conf = {person['confidence']:.2f}")

            if iou > best_iou:
                best_iou = iou
                best_person = person

        if best_person:
            self.shooter_box = best_person
            print(f"\n✓ Shooter detected: IoU = {best_iou:.3f}")
            return best_person
        else:
            print("\nERROR: Could not match ball to any person")
            return None

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def extract_position(self):
        """Extract foot position from shooter bounding box"""
        print("\n" + "="*60)
        print("POSITION EXTRACTION")
        print("="*60)

        if not self.shooter_box:
            print("ERROR: No shooter detected")
            return None

        # Get bottom center of person box (feet position)
        x1, y1, x2, y2 = self.shooter_box['bbox']

        foot_x = x1 + (x2 - x1) / 2
        foot_y = y2  # Bottom of bounding box

        self.shooter_position = (foot_x, foot_y)

        print(f"Shooter bounding box: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Foot position: ({foot_x:.1f}, {foot_y:.1f})")

        return self.shooter_position

    def transform_to_court(self):
        """Apply homography transformation"""
        print("\n" + "="*60)
        print("HOMOGRAPHY TRANSFORMATION")
        print("="*60)

        if self.shooter_position is None:
            print("ERROR: No shooter position available")
            return None

        # Initialize ShotLocalizer with 6-point calibration
        localizer = ShotLocalizer(
            calibration_points=self.calibration_points.tolist(),
            image_dimensions=(self.width, self.height),
            court_img_path=self.court_img_path,
            enable_visualization=False
        )

        # Use pixel coordinates directly
        foot_x, foot_y = self.shooter_position

        print(f"\n--- DEBUG INFO ---")
        print(f"Image dimensions: {self.width}x{self.height}")
        print(f"Shooter bbox: {self.shooter_box['bbox']}")
        print(f"Foot pixel position: ({foot_x:.1f}, {foot_y:.1f})")
        print(f"\nCalibration points (pixels):")
        sys.path.insert(0, 'backend/score_detection')
        from constants import CALIBRATION_LABELS
        for i, (pt, label) in enumerate(zip(self.calibration_points, CALIBRATION_LABELS)):
            print(f"  {i+1}. {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

        # Transform to court coordinates (in meters)
        court_coords = localizer.map_to_court((foot_x, foot_y))

        print(f"\nHomography matrix:")
        print(localizer.homography_matrix)

        # Test homography with a known calibration point
        test_pt_pixel = self.calibration_points[2]  # Right penalty box baseline
        test_pt_court = localizer.map_to_court((test_pt_pixel[0], test_pt_pixel[1]))
        print(f"\nHomography verification (testing calibration point 3):")
        print(f"  Input pixel: ({test_pt_pixel[0]:.1f}, {test_pt_pixel[1]:.1f})")
        print(f"  Output court: ({test_pt_court[0]:.2f}m, {test_pt_court[1]:.2f}m)")
        print(f"  Expected: (2.9m, 0.0m)")
        print(f"  Error: ({abs(test_pt_court[0] - 2.9):.2f}m, {abs(test_pt_court[1] - 0.0):.2f}m)")
        print(f"--- END DEBUG ---\n")

        if court_coords[0] is None:
            print("ERROR: Transformation failed (out of bounds)")
            return None

        self.court_position = court_coords

        print(f"Court position: ({court_coords[0]:.2f}m, {court_coords[1]:.2f}m)")

        # Validate bounds (FIBA half-court: X ∈ [-7.5, 7.5], Y ∈ [0, 14])
        sys.path.insert(0, 'backend/score_detection')
        from constants import COURT_WIDTH, COURT_HALF_LENGTH
        half_width = COURT_WIDTH / 2

        if not (-half_width <= court_coords[0] <= half_width and 0 <= court_coords[1] <= COURT_HALF_LENGTH):
            print(f"WARNING: Position out of FIBA court bounds!")
            print(f"  Valid range: X ∈ [{-half_width:.1f}, {half_width:.1f}]m, Y ∈ [0, {COURT_HALF_LENGTH}]m")
        else:
            print("✓ Position within valid FIBA court bounds")

        return court_coords

    def classify_zone(self):
        """Classify shot into zone"""
        if self.court_position is None:
            return None

        # Use TeamStatistics for zone classification
        stats = TeamStatistics(quarters=[float('inf')])
        zone = stats.determine_zone(self.court_position[0], self.court_position[1])
        self.zone = zone

        if zone:
            is_three_pt = stats.determine_is_three_pt(zone)
            shot_type = "3PT" if is_three_pt else "2PT"

            print(f"\nZone Classification:")
            print(f"  Zone: {zone}")
            print(f"  Shot Type: {shot_type}")
        else:
            print("\nZone Classification: Unable to determine zone")

        return zone

    def visualize(self):
        """Create side-by-side visualization"""
        print("\n" + "="*60)
        print("VISUALIZATION")
        print("="*60)

        # Left panel: Original image with annotations
        img_display = self.image.copy()

        # Draw 6-point calibration
        if self.calibration_points is not None:
            pts = self.calibration_points.astype(np.int32)

            # Draw baseline (points 0-3)
            cv2.polylines(img_display, [pts[0:4]], False, (0, 255, 255), 2)

            # Draw FT line (points 4-5)
            if len(pts) >= 6:
                cv2.line(img_display, tuple(pts[4]), tuple(pts[5]), (255, 0, 255), 2)

                # Draw penalty box sides
                cv2.line(img_display, tuple(pts[1]), tuple(pts[4]), (0, 255, 0), 2)
                cv2.line(img_display, tuple(pts[2]), tuple(pts[5]), (0, 255, 0), 2)

            # Draw calibration points
            for i, pt in enumerate(pts):
                cv2.circle(img_display, tuple(pt), 5, (255, 255, 255), -1)
                cv2.putText(img_display, str(i+1), (pt[0] + 8, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw ball
        if self.ball_box:
            x1, y1, x2, y2 = self.ball_box['bbox']
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img_display, "BALL", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw shooter
        if self.shooter_box:
            x1, y1, x2, y2 = self.shooter_box['bbox']
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img_display, "SHOOTER", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

        # Draw foot position
        if self.shooter_position:
            foot_x, foot_y = int(self.shooter_position[0]), int(self.shooter_position[1])
            # Draw large marker for foot position
            cv2.circle(img_display, (foot_x, foot_y), 12, (0, 0, 255), -1)
            cv2.circle(img_display, (foot_x, foot_y), 15, (255, 255, 255), 2)
            cv2.putText(img_display, "FEET", (foot_x + 20, foot_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw crosshair to make it very visible
            cv2.line(img_display, (foot_x - 30, foot_y), (foot_x + 30, foot_y), (0, 0, 255), 3)
            cv2.line(img_display, (foot_x, foot_y - 30), (foot_x, foot_y + 30), (0, 0, 255), 3)

            # Show pixel coordinates
            cv2.putText(img_display, f"({foot_x}, {foot_y})", (foot_x + 20, foot_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img_display, f"({foot_x}, {foot_y})", (foot_x + 20, foot_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Right panel: Court diagram
        court_img = cv2.imread(self.court_img_path)
        if court_img is None:
            print(f"WARNING: Could not load court image: {self.court_img_path}")
            return img_display

        court_height, court_width = court_img.shape[:2]

        # Draw shot location on court
        if self.court_position:
            court_x, court_y = self.court_position

            # Convert court coordinates (meters) to image pixel coordinates
            # Court: X ∈ [-7.5, 7.5]m (left to right), Y ∈ [0, 14]m (baseline to half-court)
            # Image: (0,0) at top-left, baseline at bottom

            # X: Shift from [-7.5, 7.5]m to [0, 15]m, then scale to image width
            from constants import COURT_WIDTH, COURT_HALF_LENGTH
            img_x = int((court_x + COURT_WIDTH/2) / COURT_WIDTH * court_width)

            # Y: Invert because y=0 (baseline) should be at bottom (court_height)
            #    and y=14m (half-court) should be at top (0)
            img_y = int((COURT_HALF_LENGTH - court_y) / COURT_HALF_LENGTH * court_height)

            # Draw marker
            cv2.circle(court_img, (img_x, img_y), 20, (255, 255, 255), -1)
            cv2.circle(court_img, (img_x, img_y), 15, (0, 255, 0), -1)
            cv2.circle(court_img, (img_x, img_y), 20, (255, 255, 255), 3)

        # Add text info
        info_text = []
        if self.shooter_position:
            info_text.append(f"Video: ({self.shooter_position[0]:.0f}, {self.shooter_position[1]:.0f})")
        if self.court_position:
            info_text.append(f"Court: ({self.court_position[0]:.1f}, {self.court_position[1]:.1f})")
        if self.zone:
            stats = TeamStatistics(quarters=[float('inf')])
            is_three = stats.determine_is_three_pt(self.zone)
            info_text.append(f"Zone: {self.zone} ({'3PT' if is_three else '2PT'})")

        y_offset = 40
        for text in info_text:
            cv2.putText(court_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(court_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            y_offset += 40

        # Resize to match heights
        if img_display.shape[0] != court_img.shape[0]:
            target_height = min(img_display.shape[0], court_img.shape[0])
            img_display = cv2.resize(img_display,
                                    (int(img_display.shape[1] * target_height / img_display.shape[0]), target_height))
            court_img = cv2.resize(court_img,
                                  (int(court_img.shape[1] * target_height / court_img.shape[0]), target_height))

        # Combine side-by-side
        combined = np.hstack([img_display, court_img])

        return combined

    def save_results(self, visualization, output_name=None):
        """Save visualization and JSON data"""
        if output_name is None:
            output_name = Path(self.image_path).stem

        # Save visualization
        vis_path = f"output_{output_name}.png"
        cv2.imwrite(vis_path, visualization)
        print(f"Visualization saved: {vis_path}")

        # Save JSON data
        json_path = f"output_{output_name}.json"

        output_data = {
            'image_path': self.image_path,
            'image_dimensions': {
                'width': self.width,
                'height': self.height
            },
            'shooter_position': {
                'pixel_x': float(self.shooter_position[0]) if self.shooter_position else None,
                'pixel_y': float(self.shooter_position[1]) if self.shooter_position else None,
                'normalized_x': float(self.shooter_position[0] / self.width) if self.shooter_position else None,
                'normalized_y': float(self.shooter_position[1] / self.height) if self.shooter_position else None
            },
            'court_position_meters': {
                'x': float(self.court_position[0]) if self.court_position else None,
                'y': float(self.court_position[1]) if self.court_position else None,
                'description': 'FIBA half-court coordinates: X ∈ [-7.5, 7.5]m (left to right), Y ∈ [0, 14]m (baseline to half-court)'
            },
            'zone': int(self.zone) if self.zone else None,
            'shot_type': None,
            'calibration_6pt_FIBA': self.calibration_points.tolist() if self.calibration_points is not None else None,
            'detections': {
                'ball_confidence': float(self.ball_box['confidence']) if self.ball_box else None,
                'shooter_confidence': float(self.shooter_box['confidence']) if self.shooter_box else None
            }
        }

        if self.zone:
            stats = TeamStatistics(quarters=[float('inf')])
            is_three = stats.determine_is_three_pt(self.zone)
            output_data['shot_type'] = '3PT' if is_three else '2PT'

        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"JSON data saved: {json_path}")

        return vis_path, json_path

    def run(self):
        """Main pipeline execution"""
        print("\n" + "="*60)
        print("STATIC SHOT LOCALIZATION")
        print("="*60)
        print(f"Image: {self.image_path}")
        print(f"Court Type: {self.court_type}")

        try:
            # Step 1: Calibration
            self.load_or_calibrate()

            # Step 2: Object detection
            ball_boxes, person_boxes = self.detect_objects()

            # Step 3: Select shooter
            if not self.select_shooter(ball_boxes, person_boxes):
                print("\nFAILED: Could not detect shooter")
                return

            # Step 4: Extract position
            if not self.extract_position():
                print("\nFAILED: Could not extract position")
                return

            # Step 5: Transform to court
            if not self.transform_to_court():
                print("\nFAILED: Could not transform to court coordinates")
                return

            # Step 6: Classify zone
            self.classify_zone()

            # Step 7: Visualize
            visualization = self.visualize()

            # Step 8: Display and save
            print("\nDisplaying results...")
            print("Press 'S' to save, 'Q' to quit")

            window_name = "Shot Localization - Left: Image | Right: Court"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1600, 800)
            cv2.imshow(window_name, visualization)

            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') or key == ord('S'):
                    self.save_results(visualization)
                    print("\nResults saved!")
                elif key == ord('q') or key == ord('Q'):
                    break

            cv2.destroyAllWindows()

            print("\n" + "="*60)
            print("COMPLETED SUCCESSFULLY")
            print("="*60)

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Static Image Shot Localization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python static_shot_localization.py --image shot_frame.jpg
  python static_shot_localization.py --image shot.jpg --court-type full_court
        """
    )

    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model',
                       default='backend/score_detection/weights/03102025_best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--court-type',
                       default='half_court',
                       choices=['half_court', 'full_court'],
                       help='Court type (default: half_court)')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return

    # Run localizer
    localizer = StaticShotLocalizer(args.image, args.model, args.court_type)
    localizer.run()


if __name__ == "__main__":
    main()
