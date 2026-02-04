import cv2
import numpy as np


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


def parse_color_name(color_str):
    """
    Parse a color name string to HSV values.

    Args:
        color_str: Color name (e.g., 'red', 'blue')

    Returns:
        HSV array [H, S, V] or None if invalid
    """
    color_lower = color_str.lower().strip()

    if color_lower in COLOR_NAME_MAP:
        color_dict = COLOR_NAME_MAP[color_lower]
        return np.array([color_dict['h'], color_dict['s'], color_dict['v']], dtype=np.uint8)
    else:
        return None


def get_available_colors():
    """Get list of available color names."""
    return sorted(COLOR_NAME_MAP.keys())


class TeamDetector:
    """
    Identifies which team the shooter belongs to by analyzing jersey color.

    Uses person bounding boxes provided by the shot detector and the rim
    position to find the person closest to the rim, then extracts jersey
    color and compares to the two user-provided team colors.
    """

    def __init__(self, team0_color, team1_color):
        """
        Args:
            team0_color: Color name string for team 0 (e.g., 'white')
            team1_color: Color name string for team 1 (e.g., 'red')
        """
        # Parse user-provided team colors
        self.team0_hsv = parse_color_name(team0_color)
        self.team1_hsv = parse_color_name(team1_color)

        if self.team0_hsv is None:
            raise ValueError(
                f"Invalid color '{team0_color}' for Team 0. "
                f"Available: {', '.join(get_available_colors())}"
            )
        if self.team1_hsv is None:
            raise ValueError(
                f"Invalid color '{team1_color}' for Team 1. "
                f"Available: {', '.join(get_available_colors())}"
            )

        print(f"Team 0 color: '{team0_color}' -> HSV{tuple(self.team0_hsv)}")
        print(f"Team 1 color: '{team1_color}' -> HSV{tuple(self.team1_hsv)}")

        # Warn if colors are too similar
        color_dist = np.linalg.norm(
            self.team0_hsv.astype(float) - self.team1_hsv.astype(float)
        )
        if color_dist < 30:
            print(f"WARNING: Team colors are very similar (distance: {color_dist:.1f}). "
                  f"Classification may be unreliable.")

    def detect(self, frame, person_boxes, rim_position):
        """
        Identify the team of the player taking the shot.

        Finds the person closest to the rim from the provided bounding boxes,
        extracts jersey color, and compares to team colors.

        Args:
            frame: BGR video frame
            person_boxes: list of [x1, y1, x2, y2] from shot_detector
            rim_position: rim position tuple ((x,y), frame, w, h, conf)

        Returns:
            tuple: (team_id, confidence) where team_id is 0 or 1,
                   or (None, 0.0) if shooter cannot be identified
        """
        if not person_boxes or rim_position is None:
            return None, 0.0

        # Find person closest to the rim
        rim_cx, rim_cy = rim_position[0]
        best_bbox = None
        best_dist = float('inf')

        for bbox in person_boxes:
            x1, y1, x2, y2 = bbox
            pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.sqrt((pcx - rim_cx) ** 2 + (pcy - rim_cy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_bbox = bbox

        if best_bbox is None:
            return None, 0.0

        # Classify team based on jersey color
        return self._classify_team(best_bbox, frame)

    def _classify_team(self, bbox, frame):
        """
        Classify a player's team based on jersey color.

        Args:
            bbox: [x1, y1, x2, y2] of the player
            frame: BGR video frame

        Returns:
            tuple: (team_id, confidence)
        """
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

    def _get_jersey_region(self, bbox, frame):
        """
        Extract the torso/jersey region from a player bounding box.
        Uses 25%-55% height (torso area) and center 50% width (avoids arms).

        Args:
            bbox: [x1, y1, x2, y2]
            frame: BGR video frame

        Returns:
            Cropped BGR image of the jersey region, or None
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

    def _extract_jersey_color(self, jersey_region):
        """
        Extract the dominant jersey color using median HSV after skin filtering.

        Args:
            jersey_region: BGR image of the jersey area

        Returns:
            HSV array [H, S, V] or None
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

    def _filter_skin(self, image_bgr):
        """
        Filter skin tones using HSV and YCrCb color spaces.

        Args:
            image_bgr: BGR image

        Returns:
            Boolean mask where True = non-skin pixel (keep)
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
