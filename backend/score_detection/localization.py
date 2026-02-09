# shot_localizer.py

import numpy as np
import cv2
import os
from pathlib import Path
from datetime import datetime

from constants import (
    MAP_WIDTH,
    MAP_HEIGHT,
)

from logger import (
    INFO,
    SOCKET,
    Logger
)

logger = Logger([
    INFO
])


class ShotLocalizer:
    def __init__(self, calibration_points, image_dimensions, court_img_path='court_img.png', enable_visualization=False):
        """
        Initialize localizer with 6 calibration points

        Args:
            calibration_points: List of 6 points [(x,y), ...] in video coordinates
            image_dimensions: Image dimensions as (width, height)
            court_img_path: Path to court diagram image for visualization
            enable_visualization: Whether to enable shot visualization
        """
        self.calibration_points = self._parse_points(calibration_points)
        self.image_width, self.image_height = self._parse_dimensions(image_dimensions)

        # FIBA court 6-point reference (in meters)
        from constants import CALIBRATION_POINTS_FIBA
        self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA, dtype=float)

        # Calculate the homography matrix when initialized
        self.shot_data = []
        self.homography_matrix = self._calculate_homography()

        # Visualization setup
        self.enable_visualization = enable_visualization
        self.court_img_path = court_img_path
        self.court_img = None
        self.court_img_width = None
        self.court_img_height = None
        self.shot_markers = []  # Store all shot locations for cumulative display

        if self.enable_visualization:
            self._load_court_image()
    
    def _parse_points(self, points_input):
        """Parse calibration points into numpy array"""
        if isinstance(points_input, str):
            # Format: "x1,y1,x2,y2,...,x6,y6"
            coords = list(map(float, points_input.split(',')))
            if len(coords) != 12:
                raise ValueError(f"Expected 12 coordinates (6 points), got {len(coords)}")
            return np.array([
                [coords[i], coords[i+1]] for i in range(0, 12, 2)
            ], dtype=float)
        elif isinstance(points_input, list):
            points_array = np.array(points_input, dtype=np.float32)
            if points_array.shape[0] != 6 or points_array.shape[1] != 2:
                raise ValueError(f"Expected 6 points with (x,y), got shape {points_array.shape}")
            return points_array
        else:
            raise ValueError("Invalid format for calibration points")
    
    def _parse_dimensions(self, dimensions_str):
        """Parse dimensions string into width and height"""
        if isinstance(dimensions_str, str):
            # Format: "width,height"
            return tuple(map(float, dimensions_str.split(',')))
        elif isinstance(dimensions_str, tuple) or isinstance(dimensions_str, list):
            # Already in tuple/list format
            return dimensions_str
        else:
            raise ValueError("Invalid format for image dimensions")
    
    def _calculate_homography(self):
        """
        Calculate homography matrix from 6 video points to 6 court points

        Uses cv2.findHomography with RANSAC for robustness (6 points = overdetermined system)
        """
        if self.calibration_points.shape[0] != 6:
            raise ValueError(f"Need exactly 6 calibration points, got {self.calibration_points.shape[0]}")

        # findHomography expects at least 4 points; 6 provides better constraints
        H, status = cv2.findHomography(
            self.calibration_points,
            self.court_calibration_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.1  # 10cm tolerance in meters
        )

        if H is None:
            raise ValueError("Failed to compute homography matrix. Check calibration points.")

        # Validate homography quality
        self._validate_homography(H)

        return H

    def _validate_homography(self, H):
        """
        Validate homography matrix quality by back-projecting calibration points

        Args:
            H: The homography matrix to validate
        """
        max_error = 0.0
        errors = []

        for i, (orig_pt, court_pt) in enumerate(zip(self.calibration_points, self.court_calibration_points)):
            # Transform original point using homography
            point_array = np.array([[orig_pt[0], orig_pt[1]]], dtype=float)
            transformed = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), H)

            # Calculate error
            error = np.sqrt((transformed[0][0][0] - court_pt[0])**2 +
                          (transformed[0][0][1] - court_pt[1])**2)
            errors.append(error)
            max_error = max(max_error, error)

            # Log warnings for high errors
            if error > 0.2:  # 20cm tolerance
                logger.log(INFO, f"WARNING: High calibration error at point {i+1}: {error:.3f}m")

        avg_error = np.mean(errors)
        logger.log(INFO, f"Homography validation - Avg error: {avg_error:.3f}m, Max error: {max_error:.3f}m")

        if max_error > 0.5:  # 50cm is too high
            logger.log(INFO, f"WARNING: Maximum calibration error ({max_error:.3f}m) exceeds recommended threshold (0.5m)")
            logger.log(INFO, "Consider recalibrating for better accuracy")

        return max_error
    
    def map_to_court(self, point):
        """
        Map a point from video coordinates to court coordinates (in meters)

        Args:
            point: (x, y) coordinates in the video (in pixels)

        Returns:
            (x, y) coordinates in meters on the FIBA half-court
        """
        if not point[0] or not point[1]:
            return (None, None)

        logger.log(INFO, f"Mapping point: {point}")

        # Convert point to proper format (pixel coordinates)
        point_array = np.array([[point[0], point[1]]], dtype=float)

        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), self.homography_matrix)

        # Extract result
        court_x = transformed_point[0][0][0]
        court_y = transformed_point[0][0][1]

        # Validate bounds (court half-court)
        from constants import COURT_WIDTH, COURT_HALF_LENGTH

        if court_x < 0 or court_x > COURT_WIDTH or court_y < 0 or court_y > COURT_HALF_LENGTH:
            logger.log(INFO, f"WARNING: Transformed point ({court_x:.2f}m, {court_y:.2f}m) is out of court bounds " +
                            f"(0 to {COURT_WIDTH}m, 0 to {COURT_HALF_LENGTH}m)")

        return (court_x, court_y)

    def _load_court_image(self):
        """Load the court diagram image for visualization"""
        if not os.path.exists(self.court_img_path):
            logger.log(INFO, f"Warning: Court image not found at {self.court_img_path}")
            self.enable_visualization = False
            return

        self.court_img = cv2.imread(self.court_img_path)
        if self.court_img is None:
            logger.log(INFO, f"Warning: Failed to load court image from {self.court_img_path}")
            self.enable_visualization = False
            return

        self.court_img_height, self.court_img_width = self.court_img.shape[:2]
        logger.log(INFO, f"Court image loaded: {self.court_img_width}x{self.court_img_height}")

    def _court_to_image_coords(self, court_x, court_y):
        """
        Convert court coordinates to image pixel coordinates

        Args:
            court_x: X coordinate in meters [0, 15] (left to right)
            court_y: Y coordinate in meters [0, 14] (baseline to half-court)

        Returns:
            (img_x, img_y): Pixel coordinates in the court image
        """
        if self.court_img_width is None or self.court_img_height is None:
            return None, None

        from constants import COURT_WIDTH, COURT_HALF_LENGTH

        # Convert from court coordinates to image coordinates
        # Court: X ∈ [0, 15]m, Y ∈ [0, 14]m (baseline to half-court)
        # Image: top-left = baseline-left, bottom-left = half-court-left (Y-axis is FLIPPED!)

        # X: [0, 15] -> [0, img_width] (left to right)
        normalized_x = court_x / COURT_WIDTH

        # Y: [0, 14] -> [0, img_height] (NOT flipped - 0 at top, 14 at bottom)
        normalized_y = court_y / COURT_HALF_LENGTH

        img_x = int(normalized_x * self.court_img_width)
        img_y = int(normalized_y * self.court_img_height)

        return img_x, img_y

    def visualize_shot(self, court_location, is_made=True, timestamp=None, save_path='shot_visualizations'):
        """
        Visualize a shot on the court diagram

        Args:
            court_location: (x, y) coordinates in court space
            is_made: Whether the shot was made (green) or missed (red)
            timestamp: Optional timestamp string for labeling
            save_path: Directory to save visualization images

        Returns:
            Path to saved visualization image, or None if visualization is disabled
        """
        if not self.enable_visualization or self.court_img is None:
            return None

        court_x, court_y = court_location
        if court_x is None or court_y is None:
            logger.log(INFO, "Cannot visualize shot: Invalid court location")
            return None

        # Convert to image coordinates
        img_x, img_y = self._court_to_image_coords(court_x, court_y)
        if img_x is None or img_y is None:
            return None

        # Store shot marker for cumulative visualization
        self.shot_markers.append({
            'position': (img_x, img_y),
            'is_made': is_made,
            'timestamp': timestamp
        })

        # Create a fresh copy of the court image
        court_display = self.court_img.copy()

        # Draw all shot markers with smaller dots and labels
        for idx, marker in enumerate(self.shot_markers):
            pos = marker['position']
            made = marker['is_made']
            shot_num = idx + 1

            # Color: Green for made shots, Red for misses
            color = (0, 255, 0) if made else (0, 0, 255)

            # Draw smaller dots (radius 5 instead of 8)
            cv2.circle(court_display, pos, 5, (255, 255, 255), -1)  # White border
            cv2.circle(court_display, pos, 4, color, -1)            # Colored center

            # Add shot number label (every 5th shot or if it's the latest)
            if shot_num % 5 == 0 or shot_num == len(self.shot_markers):
                label = f"{shot_num}"
                label_pos = (pos[0] + 8, pos[1] - 8)
                cv2.putText(court_display, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
                cv2.putText(court_display, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Highlight the most recent shot (slightly larger)
        latest_pos = self.shot_markers[-1]['position']
        latest_made = self.shot_markers[-1]['is_made']
        latest_color = (0, 255, 0) if latest_made else (0, 0, 255)

        # Draw larger marker for latest shot
        cv2.circle(court_display, latest_pos, 8, (255, 255, 255), 2)   # Outer ring
        cv2.circle(court_display, latest_pos, 6, latest_color, -1)     # Filled center

        # Add text label
        label = f"{'MADE' if latest_made else 'MISS'}"
        if timestamp:
            label += f" @ {timestamp}"

        # Position text above the shot marker
        text_pos = (latest_pos[0] - 40, latest_pos[1] - 20)
        cv2.putText(court_display, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 2)
        cv2.putText(court_display, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, latest_color, 1)

        # Add shot count
        made_count = sum(1 for m in self.shot_markers if m['is_made'])
        total_count = len(self.shot_markers)
        stats_text = f"Shots: {made_count}/{total_count}"
        cv2.putText(court_display, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        cv2.putText(court_display, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 0), 1)

        # Save visualization
        Path(save_path).mkdir(parents=True, exist_ok=True)
        timestamp_str = timestamp.replace(':', '-') if timestamp else datetime.now().strftime('%H-%M-%S')
        output_filename = f"shot_{total_count:03d}_{timestamp_str}.png"
        output_path = os.path.join(save_path, output_filename)

        cv2.imwrite(output_path, court_display)
        logger.log(INFO, f"Shot visualization saved to: {output_path}")

        return output_path

    def get_shot_chart(self, save_path='shot_chart.png'):
        """
        Generate a final shot chart with all shots

        Args:
            save_path: Path to save the final shot chart

        Returns:
            Path to saved shot chart image
        """
        if not self.enable_visualization or self.court_img is None:
            return None

        if not self.shot_markers:
            logger.log(INFO, "No shots to visualize")
            return None

        # Create a fresh copy of the court image
        court_display = self.court_img.copy()

        # Draw all shot markers
        made_shots = []
        missed_shots = []

        for marker in self.shot_markers:
            if marker['is_made']:
                made_shots.append(marker['position'])
            else:
                missed_shots.append(marker['position'])

        # Draw missed shots (red) with smaller dots
        for pos in missed_shots:
            cv2.circle(court_display, pos, 5, (255, 255, 255), -1)
            cv2.circle(court_display, pos, 4, (0, 0, 255), -1)

        # Draw made shots (green) with smaller dots
        for pos in made_shots:
            cv2.circle(court_display, pos, 5, (255, 255, 255), -1)
            cv2.circle(court_display, pos, 4, (0, 255, 0), -1)

        # Add statistics
        made_count = len(made_shots)
        total_count = len(self.shot_markers)
        percentage = (made_count / total_count * 100) if total_count > 0 else 0

        stats_text = f"Shot Chart: {made_count}/{total_count} ({percentage:.1f}%)"
        cv2.putText(court_display, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 3)
        cv2.putText(court_display, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 2)

        # Add legend
        legend_y = 60
        cv2.circle(court_display, (20, legend_y), 6, (0, 255, 0), -1)
        cv2.putText(court_display, f"Made: {made_count}", (35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        legend_y += 25
        cv2.circle(court_display, (20, legend_y), 6, (0, 0, 255), -1)
        cv2.putText(court_display, f"Missed: {total_count - made_count}", (35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save shot chart
        cv2.imwrite(save_path, court_display)
        logger.log(INFO, f"Final shot chart saved to: {save_path}")

        return save_path