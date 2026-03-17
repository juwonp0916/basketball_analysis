"""
TeamDetector - Identifies shooter's team by auto-calibrated jersey color analysis.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.cluster import KMeans
import colorsys


class StreamingTeamDetector:
    """
    Identifies shooter's team by analyzing jersey color.
    Uses automatic K-Means clustering to discover team colors.
    """

    def __init__(self):
        self.team0_color: Optional[np.ndarray] = None  # Feature representation
        self.team1_color: Optional[np.ndarray] = None
        self._configured = False

        # Track IDs to Team assignments for temporal smoothing
        self.track_history: Dict[int, List[int]] = {}

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_jersey_region(
        self,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract the torso/jersey region from a player bounding box.
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1

        if height <= 0 or width <= 0:
            return None

        # Vertical: 25%-55% from top
        jersey_y1 = y1 + int(height * 0.25)
        jersey_y2 = y1 + int(height * 0.55)

        # Horizontal: center 50%
        margin_x = int(width * 0.25)
        jersey_x1 = x1 + margin_x
        jersey_x2 = x2 - margin_x

        if jersey_x2 - jersey_x1 < 10:
            margin_x = int(width * 0.1)
            jersey_x1 = x1 + margin_x
            jersey_x2 = x2 - margin_x

        h, w = frame.shape[:2]
        jersey_y1 = max(0, min(jersey_y1, h))
        jersey_y2 = max(0, min(jersey_y2, h))
        jersey_x1 = max(0, min(jersey_x1, w))
        jersey_x2 = max(0, min(jersey_x2, w))

        region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

        if region.size == 0:
            return None

        return region

    def _filter_skin(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Filter skin tones using HSV and YCrCb color spaces.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

        lower_skin_hsv = np.array([0, 10, 30], dtype=np.uint8)
        upper_skin_hsv = np.array([30, 180, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

        lower_skin_ycrcb = np.array([0, 130, 70], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)

        skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        return skin_mask == 0

    def _extract_color_features(self, jersey_region: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features (e.g. median HSV) from non-skin pixels.
        """
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        non_skin_mask = self._filter_skin(jersey_region)

        pixels_hsv = hsv.reshape(-1, 3)
        mask_flat = non_skin_mask.reshape(-1)
        filtered_pixels = pixels_hsv[mask_flat]

        if len(filtered_pixels) < 10:
            filtered_pixels = pixels_hsv
        if len(filtered_pixels) < 5:
            return None

        # Feature: Median HSV
        median_hsv = np.median(filtered_pixels, axis=0).astype(np.float32)
        return median_hsv

    def _hue_distance(self, h1: float, h2: float) -> float:
        """Calculate circular distance between two hues (0-179 in OpenCV)."""
        diff = abs(h1 - h2)
        return min(diff, 180 - diff)

    def _color_distance(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """
        Distance between two HSV features.
        Weights hue more heavily, properly handles circular hue.
        """
        h_dist = self._hue_distance(f1[0], f2[0])
        s_dist = abs(f1[1] - f2[1])
        v_dist = abs(f1[2] - f2[2])
        # Weights: Hue=2.0 (scaled to 180), Sat=1.0, Val=0.5
        return np.sqrt((h_dist * 2.0)**2 + (s_dist * 1.0)**2 + (v_dist * 0.5)**2)

    def auto_calibrate(self, frame: np.ndarray, model, class_names: List[str], inference_dims: Tuple[int, int], device: str) -> bool:
        """
        Run person detection on the given frame, extract colors, cluster into 2 teams.
        Returns True if successful.
        """
        inf_w, inf_h = inference_dims
        frame_h, frame_w = frame.shape[:2]
        det_frame = cv2.resize(frame, (inf_w, inf_h))

        results = model(
            det_frame,
            stream=False,
            verbose=False,
            imgsz=inf_w,
            device=device,
            conf=0.3,
            max_det=30
        )

        features = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls >= len(class_names) or class_names[cls] != 'person':
                    continue
                if float(box.conf[0]) < 0.4:
                    continue

                x1 = int(box.xyxy[0][0] * frame_w / inf_w)
                y1 = int(box.xyxy[0][1] * frame_h / inf_h)
                x2 = int(box.xyxy[0][2] * frame_w / inf_w)
                y2 = int(box.xyxy[0][3] * frame_h / inf_h)

                jersey_region = self._get_jersey_region((x1, y1, x2, y2), frame)
                if jersey_region is not None:
                    feat = self._extract_color_features(jersey_region)
                    if feat is not None:
                        features.append(feat)

        if len(features) < 4:
            print(f"[TeamDetector] Not enough players for auto-calibration ({len(features)} found)")
            return False

        # Convert to features suitable for KMeans.
        # Since hue is circular, we can map it to sin/cos.
        # H is 0-179 in OpenCV, so angle = H * 2 * pi / 180
        X = []
        for f in features:
            h, s, v = f
            angle = h * 2.0 * np.pi / 180.0
            # weight S and V so they have some effect, but hue dominates
            X.append([np.cos(angle) * 100, np.sin(angle) * 100, s, v * 0.5])

        X = np.array(X)

        # Use K=3 to filter out referees, then pick top 2 clusters
        n_clusters = min(3, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)

        # Count elements in each cluster
        counts = np.bincount(kmeans.labels_)

        # Get top 2 clusters by size
        top_2_idx = np.argsort(counts)[-2:]

        # The centroids in the original feature space (median HSV)
        # We can find the average HSV of the items in those clusters.
        team_features = []
        for cluster_idx in top_2_idx:
            cluster_items = [features[i] for i in range(len(features)) if kmeans.labels_[i] == cluster_idx]
            # Average the features, but for Hue we need to be careful.
            # Median is safer.
            median_feat = np.median(cluster_items, axis=0)
            team_features.append(median_feat)

        self.team0_color = team_features[0]
        self.team1_color = team_features[1]
        self._configured = True

        def hsv_to_hex(hsv):
            # hsv: H(0-179), S(0-255), V(0-255)
            h, s, v = hsv
            r, g, b = colorsys.hsv_to_rgb(h / 179.0, s / 255.0, v / 255.0)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        print("[TeamDetector] Auto-calibrated teams:")
        print(f"  Team 0 HSV: {self.team0_color} -> {hsv_to_hex(self.team0_color)}")
        print(f"  Team 1 HSV: {self.team1_color} -> {hsv_to_hex(self.team1_color)}")

        return True

    def get_team_colors_hex(self) -> Tuple[str, str]:
        """Return the auto-calibrated colors as hex strings for the frontend."""
        if not self._configured:
            return "#ff0000", "#0000ff"  # fallback

        def hsv_to_hex(hsv):
            h, s, v = hsv
            r, g, b = colorsys.hsv_to_rgb(h / 179.0, s / 255.0, v / 255.0)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        return hsv_to_hex(self.team0_color), hsv_to_hex(self.team1_color)

    def classify_from_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: Optional[int] = None
    ) -> Tuple[Optional[int], float]:
        """
        Classify team from a player bounding box.
        Optionally uses track_id for temporal smoothing.
        """
        if not self._configured:
            return None, 0.0

        jersey_region = self._get_jersey_region(bbox, frame)
        if jersey_region is None:
            return None, 0.0

        feat = self._extract_color_features(jersey_region)
        if feat is None:
            return None, 0.0

        dist0 = self._color_distance(feat, self.team0_color)
        dist1 = self._color_distance(feat, self.team1_color)

        raw_team = 0 if dist0 < dist1 else 1

        min_dist = min(dist0, dist1)
        max_dist = max(dist0, dist1)
        confidence = 1.0 - (min_dist / max_dist) if max_dist > 0 else 0.5

        # Temporal smoothing
        if track_id is not None:
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(raw_team)

            # Keep last 5 frames
            if len(self.track_history[track_id]) > 5:
                self.track_history[track_id].pop(0)

            # Majority vote
            history = self.track_history[track_id]
            final_team = max(set(history), key=history.count)
            # Increase confidence if history agrees
            if final_team == raw_team:
                confidence = min(1.0, confidence + 0.1)
        else:
            final_team = raw_team

        return final_team, confidence

    def classify_from_position(
        self,
        frame: np.ndarray,
        shooter_pos: dict,
        model,
        class_names: List[str],
        inference_dims: Tuple[int, int],
        frame_dims: Tuple[int, int],
        device: str = 'cpu'
    ) -> Tuple[Optional[int], float, Optional[Tuple[int, int, int, int]]]:
        """
        Finds the player closest to the shooter position and classifies their team.
        """
        if not self._configured:
            return None, 0.0, None

        inf_w, inf_h = inference_dims
        frame_w, frame_h = frame_dims

        det_frame = cv2.resize(frame, (inf_w, inf_h))

        results = model(
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
        # Note: If we had a tracker, we'd get track_id from results.
        # But YOLO standard results don't have track IDs unless model.track() is used.
        # We will fallback to track_id=None since it's just one shot event.
        best_track_id = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls >= len(class_names) or class_names[cls] != 'person':
                    continue

                if float(box.conf[0]) < 0.3:
                    continue

                x1 = int(box.xyxy[0][0] * frame_w / inf_w)
                y1 = int(box.xyxy[0][1] * frame_h / inf_h)
                x2 = int(box.xyxy[0][2] * frame_w / inf_w)
                y2 = int(box.xyxy[0][3] * frame_h / inf_h)

                center_x = (x1 + x2) / 2
                bottom_y = y2
                dist = np.sqrt((center_x - shooter_pos['x'])**2 + (bottom_y - shooter_pos['y'])**2)

                if dist < best_dist:
                    best_dist = dist
                    best_bbox = (x1, y1, x2, y2)
                    if box.id is not None:
                        best_track_id = int(box.id[0])

        if best_bbox is None:
            return None, 0.0, None

        team_id, confidence = self.classify_from_bbox(frame, best_bbox, track_id=best_track_id)
        return team_id, confidence, best_bbox
