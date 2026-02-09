"""
TeamDetector - Identifies shooter's team by jersey color analysis.

Adapted from Organized_code/team_detector.py for streaming use.
Called only when a shot is detected to minimize overhead.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# Color name to HSV mapping
# HSV ranges: H (0-179), S (0-255), V (0-255)
COLOR_NAME_MAP = {
    'red': {'h': 0, 's': 200, 'v': 200},
    'blue': {'h': 110, 's': 200, 'v': 200},
    'green': {'h': 60, 's': 200, 'v': 150},
    'yellow': {'h': 30, 's': 200, 'v': 200},
    'orange': {'h': 15, 's': 220, 'v': 220},
    'purple': {'h': 140, 's': 180, 'v': 180},
    'pink': {'h': 160, 's': 150, 'v': 220},
    'cyan': {'h': 90, 's': 200, 'v': 200},
    'white': {'h': 0, 's': 20, 'v': 230},
    'black': {'h': 0, 's': 20, 'v': 40},
    'gray': {'h': 0, 's': 20, 'v': 128},
    'brown': {'h': 10, 's': 150, 'v': 100},
    'navy': {'h': 120, 's': 200, 'v': 100},
    'maroon': {'h': 0, 's': 180, 'v': 80},
    'lime': {'h': 70, 's': 240, 'v': 240},
    'teal': {'h': 95, 's': 180, 'v': 150},
    'gold': {'h': 25, 's': 200, 'v': 200},
}


def parse_color_name(color_str: str) -> Optional[np.ndarray]:
    """Parse a color name string to HSV values."""
    color_lower = color_str.lower().strip()
    if color_lower in COLOR_NAME_MAP:
        color_dict = COLOR_NAME_MAP[color_lower]
        return np.array([color_dict['h'], color_dict['s'], color_dict['v']], dtype=np.uint8)
    return None


def get_available_colors() -> List[str]:
    """Get list of available color names."""
    return sorted(COLOR_NAME_MAP.keys())


class StreamingTeamDetector:
    """
    Identifies shooter's team by analyzing jersey color.

    Designed to be called only when a shot is detected.
    Reuses the shooter bounding box already found by StreamingShotDetector.
    """

    def __init__(self, team0_color: Optional[str] = None, team1_color: Optional[str] = None):
        """
        Initialize team detector.

        Args:
            team0_color: Color name for team 0 (e.g., 'red')
            team1_color: Color name for team 1 (e.g., 'blue')
        """
        self.team0_hsv: Optional[np.ndarray] = None
        self.team1_hsv: Optional[np.ndarray] = None
        self._configured = False

        if team0_color and team1_color:
            self.set_team_colors(team0_color, team1_color)

    def set_team_colors(self, team0_color: str, team1_color: str) -> bool:
        """
        Set team colors. Can be called to update colors at runtime.

        Args:
            team0_color: Color name for team 0
            team1_color: Color name for team 1

        Returns:
            True if colors were set successfully
        """
        team0_hsv = parse_color_name(team0_color)
        team1_hsv = parse_color_name(team1_color)

        if team0_hsv is None:
            print(f"[TeamDetector] Invalid color '{team0_color}'. Available: {', '.join(get_available_colors())}")
            return False
        if team1_hsv is None:
            print(f"[TeamDetector] Invalid color '{team1_color}'. Available: {', '.join(get_available_colors())}")
            return False

        self.team0_hsv = team0_hsv
        self.team1_hsv = team1_hsv
        self._configured = True

        print(f"[TeamDetector] Team 0: '{team0_color}' -> HSV{tuple(team0_hsv)}")
        print(f"[TeamDetector] Team 1: '{team1_color}' -> HSV{tuple(team1_hsv)}")

        # Warn if colors are too similar
        color_dist = np.linalg.norm(team0_hsv.astype(float) - team1_hsv.astype(float))
        if color_dist < 30:
            print(f"[TeamDetector] WARNING: Team colors are very similar (distance: {color_dist:.1f})")

        return True

    @property
    def is_configured(self) -> bool:
        """Check if team colors have been set."""
        return self._configured

    def classify_from_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[int], float]:
        """
        Classify team from a player bounding box.

        Args:
            frame: BGR video frame
            bbox: (x1, y1, x2, y2) bounding box of the shooter

        Returns:
            (team_id, confidence) where team_id is 0 or 1,
            or (None, 0.0) if classification fails
        """
        if not self._configured:
            return None, 0.0

        jersey_region = self._get_jersey_region(bbox, frame)

        if jersey_region is None or jersey_region.size == 0:
            return None, 0.0

        jersey_hsv = self._extract_jersey_color(jersey_region)

        if jersey_hsv is None:
            return None, 0.0

        # Compare to team colors
        dist_team0 = np.linalg.norm(jersey_hsv.astype(float) - self.team0_hsv.astype(float))
        dist_team1 = np.linalg.norm(jersey_hsv.astype(float) - self.team1_hsv.astype(float))

        team = 0 if dist_team0 < dist_team1 else 1

        # Confidence: how much closer to one team vs the other
        min_dist = min(dist_team0, dist_team1)
        max_dist = max(dist_team0, dist_team1)

        if max_dist > 0:
            confidence = 1.0 - (min_dist / max_dist)
        else:
            confidence = 0.5

        return team, confidence

    def classify_from_position(
        self,
        frame: np.ndarray,
        shooter_pos: dict,
        model_shoot,
        class_names_shoot: List[str],
        inference_dims: Tuple[int, int],
        frame_dims: Tuple[int, int],
        device: str = 'cpu'
    ) -> Tuple[Optional[int], float, Optional[Tuple[int, int, int, int]]]:
        """
        Classify team by finding the person bbox near shooter position.

        Args:
            frame: BGR video frame
            shooter_pos: {'x': float, 'y': float} shooter position
            model_shoot: YOLO model for person detection
            class_names_shoot: Class names for the model
            inference_dims: (width, height) for inference
            frame_dims: (width, height) of original frame
            device: Device for inference

        Returns:
            (team_id, confidence, bbox) or (None, 0.0, None)
        """
        if not self._configured:
            return None, 0.0, None

        inf_w, inf_h = inference_dims
        frame_w, frame_h = frame_dims

        det_frame = cv2.resize(frame, (inf_w, inf_h))

        results = model_shoot(
            det_frame,
            stream=True,
            verbose=False,
            imgsz=inf_w,
            device=device,
            conf=0.3,
            max_det=15
        )

        best_bbox = None
        best_dist = float('inf')

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls >= len(class_names_shoot):
                    continue
                class_name = class_names_shoot[cls]

                if class_name != 'person':
                    continue

                conf = float(box.conf[0])
                if conf < 0.3:
                    continue

                x1 = int(box.xyxy[0][0] * frame_w / inf_w)
                y1 = int(box.xyxy[0][1] * frame_h / inf_h)
                x2 = int(box.xyxy[0][2] * frame_w / inf_w)
                y2 = int(box.xyxy[0][3] * frame_h / inf_h)

                # Check distance from shooter position to person center-bottom
                center_x = (x1 + x2) / 2
                bottom_y = y2
                dist = np.sqrt((center_x - shooter_pos['x'])**2 + (bottom_y - shooter_pos['y'])**2)

                if dist < best_dist:
                    best_dist = dist
                    best_bbox = (x1, y1, x2, y2)

        if best_bbox is None:
            return None, 0.0, None

        team_id, confidence = self.classify_from_bbox(frame, best_bbox)
        return team_id, confidence, best_bbox

    def _get_jersey_region(
        self,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract the torso/jersey region from a player bounding box.
        Uses 25%-55% height (torso area) and center 50% width (avoids arms).
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1

        if height <= 0 or width <= 0:
            return None

        # Vertical: 25%-55% from top (torso area, avoids head and legs)
        jersey_y1 = y1 + int(height * 0.25)
        jersey_y2 = y1 + int(height * 0.55)

        # Horizontal: center 50% (avoids arms)
        margin_x = int(width * 0.25)
        jersey_x1 = x1 + margin_x
        jersey_x2 = x2 - margin_x

        # Fallback if bbox is too narrow
        if jersey_x2 - jersey_x1 < 10:
            margin_x = int(width * 0.1)
            jersey_x1 = x1 + margin_x
            jersey_x2 = x2 - margin_x

        # Bounds checking
        h, w = frame.shape[:2]
        jersey_y1 = max(0, min(jersey_y1, h))
        jersey_y2 = max(0, min(jersey_y2, h))
        jersey_x1 = max(0, min(jersey_x1, w))
        jersey_x2 = max(0, min(jersey_x2, w))

        region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

        if region.size == 0:
            return None

        return region

    def _extract_jersey_color(self, jersey_region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the dominant jersey color using median HSV after skin filtering.
        """
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)

        # Filter out skin tones
        non_skin_mask = self._filter_skin(jersey_region)

        # Apply mask
        pixels_hsv = hsv.reshape(-1, 3)
        mask_flat = non_skin_mask.reshape(-1)
        filtered_pixels = pixels_hsv[mask_flat]

        if len(filtered_pixels) < 10:
            # Fall back to all pixels if not enough non-skin pixels
            filtered_pixels = pixels_hsv

        if len(filtered_pixels) < 5:
            return None

        # Use median color (robust to outliers)
        median_hsv = np.median(filtered_pixels, axis=0).astype(np.uint8)

        return median_hsv

    def _filter_skin(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Filter skin tones using HSV and YCrCb color spaces.
        Returns boolean mask where True = non-skin pixel (keep).
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

        # HSV skin detection
        lower_skin_hsv = np.array([0, 10, 30], dtype=np.uint8)
        upper_skin_hsv = np.array([30, 180, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

        # YCrCb skin detection
        lower_skin_ycrcb = np.array([0, 130, 70], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)

        # Combine (skin if detected in either space)
        skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        # Invert: True = not skin (keep)
        return skin_mask == 0
