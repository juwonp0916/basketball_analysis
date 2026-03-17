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
    def __init__(self, calibration_points, image_dimensions, court_img_path='court_img.png', enable_visualization=False, calibration_mode='6-point'):
        """
        Initialize localizer with 4 or 6 calibration points

        Args:
            calibration_points: List of 4 or 6 points [(x,y), ...] in video coordinates
            image_dimensions: Image dimensions as (width, height)
            court_img_path: Path to court diagram image for visualization
            enable_visualization: Whether to enable shot visualization
            calibration_mode: "4-point" (paint box) or "6-point" (full baseline)
        """
        self.calibration_mode = calibration_mode
        self.calibration_points = self._parse_points(calibration_points, calibration_mode)
        self.image_width, self.image_height = self._parse_dimensions(image_dimensions)

        # FIBA court reference points (in meters)
        from constants import CALIBRATION_POINTS_FIBA, CALIBRATION_POINTS_FIBA_PAINT
        if calibration_mode == '4-point':
            self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA_PAINT, dtype=float)
        else:
            self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA, dtype=float)

        # Calculate the homography matrix when initialized
        self.shot_data = []
        self.homography_matrix = self._calculate_homography()

        # Visualization setup (now uses SVG-based renderer instead of PNG)
        self.enable_visualization = enable_visualization
        # Legacy attributes kept for backward compatibility but not used
        self.court_img_path = court_img_path
        self.court_img = None
        self.court_img_width = None
        self.court_img_height = None
        self.shot_markers = []  # Deprecated - using shot_data instead

        # No need to load court image - using SVG renderer now
        if self.enable_visualization:
            logger.log(INFO, "Shot visualization enabled (SVG-based renderer)")
    
    def _parse_points(self, points_input, calibration_mode='6-point'):
        """Parse calibration points into numpy array"""
        expected_points = 4 if calibration_mode == '4-point' else 6

        if isinstance(points_input, str):
            # Format: "x1,y1,x2,y2,..."
            coords = list(map(float, points_input.split(',')))
            expected_coords = expected_points * 2
            if len(coords) != expected_coords:
                raise ValueError(f"Expected {expected_coords} coordinates ({expected_points} points), got {len(coords)}")
            return np.array([
                [coords[i], coords[i+1]] for i in range(0, len(coords), 2)
            ], dtype=float)
        elif isinstance(points_input, list):
            points_array = np.array(points_input, dtype=np.float32)
            if points_array.shape[0] != expected_points or points_array.shape[1] != 2:
                raise ValueError(f"Expected {expected_points} points with (x,y), got shape {points_array.shape}")
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
        Calculate homography matrix from video points to court points

        Supports both 4-point (paint box) and 6-point (full baseline) calibration.
        Uses cv2.findHomography with RANSAC for robustness.
        """
        expected_points = 4 if self.calibration_mode == '4-point' else 6
        if self.calibration_points.shape[0] != expected_points:
            raise ValueError(f"Need exactly {expected_points} calibration points for {self.calibration_mode} mode, got {self.calibration_points.shape[0]}")

        # findHomography works with both 4 and 6 points
        # 4 points = minimum for homography (exactly determined)
        # 6 points = overdetermined (better constraints, more robust)
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
        Visualize a shot on the court diagram (deprecated - shots are now stored in shot_data)

        This method is kept for backward compatibility but no longer generates individual
        shot images. Use get_shot_chart() to generate the final shot chart with all shots.

        Args:
            court_location: (x, y) coordinates in court space
            is_made: Whether the shot was made (green) or missed (red)
            timestamp: Optional timestamp string for labeling
            save_path: Directory to save visualization images (not used)

        Returns:
            None (no longer generates individual shot images)
        """
        if not self.enable_visualization:
            return None

        court_x, court_y = court_location
        if court_x is None or court_y is None:
            logger.log(INFO, "Cannot visualize shot: Invalid court location")
            return None

        # Shots are now stored in shot_data during detection
        # Individual shot visualization is deprecated - use get_shot_chart() instead
        logger.log(INFO, f"Shot logged at ({court_x:.2f}m, {court_y:.2f}m) - {'Made' if is_made else 'Missed'}")

        return None

    def get_shot_chart(self, save_path='shot_chart.png'):
        """
        Generate a final shot chart with all shots using SVG-based renderer

        Args:
            save_path: Path to save the final shot chart

        Returns:
            Path to saved shot chart image
        """
        if not self.enable_visualization:
            return None

        if not self.shot_data:
            logger.log(INFO, "No shots to visualize")
            return None

        # Use the new SVG-based court renderer
        from court_renderer import CourtRenderer

        # Extract shot data: (x_m, y_m, is_made)
        shots = [
            (shot['court_position_meters']['x'],
             shot['court_position_meters']['y'],
             shot['is_made'])
            for shot in self.shot_data
            if shot.get('court_position_meters')
        ]

        if not shots:
            logger.log(INFO, "No valid shots with court coordinates")
            return None

        # Generate statistics for title
        made_count = sum(1 for _, _, is_made in shots if is_made)
        total_count = len(shots)
        pct = (made_count / total_count * 100) if total_count > 0 else 0
        title = f"Shot Chart: {made_count}/{total_count} ({pct:.1f}%)"

        # Create shot chart using SVG renderer
        CourtRenderer.create_shot_chart(
            shots=shots,
            title=title,
            save_path=save_path,
            show_stats=True
        )

        logger.log(INFO, f"Final shot chart saved to: {save_path}")
        return save_path