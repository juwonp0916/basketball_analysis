"""
TeamDetector v2 - Robust team auto-calibration for amateur basketball.

Key improvements over v1:
- Resolution-aware filtering (relative bbox size instead of fixed pixel area)
- Multi-frame feature accumulation (not single-frame all-or-nothing)
- K=2 clustering + MAD-based outlier removal (no referee color assumption)
- Supports 1v1, 3v3, and 5v5 (minimum 2 players to calibrate)
- Silent periodic re-check after initial calibration
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.cluster import KMeans
import colorsys
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Resolution-aware bbox filters (fraction of frame dimensions)
MIN_BBOX_HEIGHT_RATIO = 0.06   # Player must be >= 6% of frame height
MIN_BBOX_WIDTH_RATIO = 0.02    # Player must be >= 2% of frame width

# Confidence: match YOLO's own conf parameter — let size filter reject noise
MIN_PERSON_CONFIDENCE = 0.3

# Accumulation limits
MAX_ACCUMULATED_FEATURES = 80  # Cap memory; enough for robust clustering

# Clustering quality gate
MIN_INTER_CLUSTER_DISTANCE = 25.0  # In HSV-weighted space; reject if teams too similar

# MAD outlier removal multiplier (higher = less aggressive)
MAD_OUTLIER_FACTOR = 2.5


class StreamingTeamDetector:
    """
    Identifies shooter's team by analyzing jersey color.
    Uses multi-frame accumulation + K-Means clustering to discover team colors.
    """

    def __init__(self):
        self.team0_color: Optional[np.ndarray] = None  # HSV centroid
        self.team1_color: Optional[np.ndarray] = None
        self._configured = False

        # Multi-frame feature accumulation
        self._accumulated_features: List[np.ndarray] = []

        # Track IDs to Team assignments for temporal smoothing
        self.track_history: Dict[int, List[int]] = {}

    def reset(self) -> None:
        """
        Reset detector to unconfigured state for a new session.
        Clears team colors, accumulated features, and tracking history.
        """
        self.team0_color = None
        self.team1_color = None
        self._configured = False
        self._accumulated_features.clear()
        self.track_history.clear()

    @property
    def is_configured(self) -> bool:
        return self._configured

    # ------------------------------------------------------------------
    # Jersey / color extraction (unchanged from v1)
    # ------------------------------------------------------------------

    def _get_jersey_region(
        self,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract the torso/jersey region from a player bounding box."""
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
        """Filter skin tones using HSV and YCrCb color spaces."""
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
        """Extract median HSV from non-skin pixels."""
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        non_skin_mask = self._filter_skin(jersey_region)

        pixels_hsv = hsv.reshape(-1, 3)
        mask_flat = non_skin_mask.reshape(-1)
        filtered_pixels = pixels_hsv[mask_flat]

        if len(filtered_pixels) < 10:
            filtered_pixels = pixels_hsv
        if len(filtered_pixels) < 5:
            return None

        median_hsv = np.median(filtered_pixels, axis=0).astype(np.float32)
        return median_hsv

    # ------------------------------------------------------------------
    # Distance helpers (unchanged from v1)
    # ------------------------------------------------------------------

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
        return np.sqrt((h_dist * 2.0)**2 + (s_dist * 1.0)**2 + (v_dist * 0.5)**2)

    # ------------------------------------------------------------------
    # HSV <-> clustering feature conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_to_cluster_feature(hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to a KMeans-friendly feature (cos/sin hue + S + V)."""
        h, s, v = hsv
        angle = h * 2.0 * np.pi / 180.0
        return np.array([np.cos(angle) * 100, np.sin(angle) * 100, s, v * 0.5])

    # ------------------------------------------------------------------
    # Core: extract features from a single frame
    # ------------------------------------------------------------------

    def _extract_frame_features(
        self,
        frame: np.ndarray,
        model,
        class_names: List[str],
        inference_dims: Tuple[int, int],
        device: str
    ) -> List[np.ndarray]:
        """
        Run person detection on one frame and return a list of HSV features.
        Uses resolution-aware bbox filtering.
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
            conf=MIN_PERSON_CONFIDENCE,
            max_det=30
        )

        # Resolution-aware thresholds
        min_height = frame_h * MIN_BBOX_HEIGHT_RATIO
        min_width = frame_w * MIN_BBOX_WIDTH_RATIO

        features: List[np.ndarray] = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls >= len(class_names) or class_names[cls] != 'person':
                    continue

                if float(box.conf[0]) < MIN_PERSON_CONFIDENCE:
                    continue

                # Scale coords back to full frame
                x1 = int(box.xyxy[0][0] * frame_w / inf_w)
                y1 = int(box.xyxy[0][1] * frame_h / inf_h)
                x2 = int(box.xyxy[0][2] * frame_w / inf_w)
                y2 = int(box.xyxy[0][3] * frame_h / inf_h)

                bbox_h = y2 - y1
                bbox_w = x2 - x1

                if bbox_h < min_height or bbox_w < min_width:
                    continue

                jersey_region = self._get_jersey_region((x1, y1, x2, y2), frame)
                if jersey_region is not None:
                    feat = self._extract_color_features(jersey_region)
                    if feat is not None:
                        features.append(feat)

        return features

    # ------------------------------------------------------------------
    # Core: K=2 clustering + MAD outlier removal
    # ------------------------------------------------------------------

    def _cluster_and_validate(
        self, features: List[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Cluster accumulated features into 2 teams using K=2 + MAD outlier removal.

        Returns (team0_hsv, team1_hsv) on success, or None if quality gate fails.
        """
        if len(features) < 2:
            return None

        # Build clustering feature matrix
        X = np.array([self._hsv_to_cluster_feature(f) for f in features])

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        labels = kmeans.labels_

        # --- MAD-based outlier removal per cluster ---
        clean_indices: List[int] = []
        for cluster_id in range(2):
            member_idx = [i for i in range(len(features)) if labels[i] == cluster_id]
            if len(member_idx) == 0:
                continue

            centroid = kmeans.cluster_centers_[cluster_id]
            dists = np.array([np.linalg.norm(X[i] - centroid) for i in member_idx])

            median_dist = np.median(dists)
            mad = np.median(np.abs(dists - median_dist))
            # If MAD is 0 (all same distance), use a small epsilon to avoid
            # rejecting everything
            threshold = median_dist + MAD_OUTLIER_FACTOR * max(mad, 1.0)

            for j, idx in enumerate(member_idx):
                if dists[j] <= threshold:
                    clean_indices.append(idx)

        if len(clean_indices) < 2:
            logger.debug(
                f"[TeamDetector] Too few features after outlier removal "
                f"({len(clean_indices)} remaining)"
            )
            return None

        # Re-cluster on cleaned features
        clean_features = [features[i] for i in clean_indices]
        X_clean = np.array([self._hsv_to_cluster_feature(f) for f in clean_features])

        kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_clean)
        labels2 = kmeans2.labels_

        # Compute per-cluster median HSV (in original HSV space, not cluster space)
        team_colors: List[np.ndarray] = []
        for cluster_id in range(2):
            cluster_items = [
                clean_features[i]
                for i in range(len(clean_features))
                if labels2[i] == cluster_id
            ]
            if len(cluster_items) == 0:
                return None  # Degenerate: one cluster is empty
            team_colors.append(np.median(cluster_items, axis=0).astype(np.float32))

        # Quality gate: inter-cluster distance must be large enough
        inter_dist = self._color_distance(team_colors[0], team_colors[1])
        if inter_dist < MIN_INTER_CLUSTER_DISTANCE:
            logger.debug(
                f"[TeamDetector] Clusters too similar "
                f"(distance={inter_dist:.1f} < {MIN_INTER_CLUSTER_DISTANCE}), "
                f"continuing accumulation"
            )
            return None

        return (team_colors[0], team_colors[1])

    # ------------------------------------------------------------------
    # Public API: auto_calibrate  (same signature as v1)
    # ------------------------------------------------------------------

    def auto_calibrate(
        self,
        frame: np.ndarray,
        model,
        class_names: List[str],
        inference_dims: Tuple[int, int],
        device: str
    ) -> bool:
        """
        Accumulate player color features from this frame and attempt clustering.

        This method is designed to be called repeatedly across frames:
        - Each call extracts features from the current frame and appends them.
        - After each call, tries K=2 clustering + outlier removal on all
          accumulated features.
        - Returns True the first time clustering succeeds with sufficient quality.

        The caller does NOT need to change — same interface as v1.
        """
        # Phase 1: extract features from this frame and accumulate
        new_features = self._extract_frame_features(
            frame, model, class_names, inference_dims, device
        )

        if new_features:
            remaining_capacity = MAX_ACCUMULATED_FEATURES - len(self._accumulated_features)
            self._accumulated_features.extend(new_features[:remaining_capacity])

        total = len(self._accumulated_features)
        logger.info(
            f"[TeamDetector] Frame yielded {len(new_features)} features, "
            f"accumulated {total}/{MAX_ACCUMULATED_FEATURES}"
        )

        if total < 2:
            logger.info(
                f"[TeamDetector] Not enough accumulated features for calibration "
                f"({total}/2 minimum)"
            )
            return False

        # Phase 2: attempt clustering
        result = self._cluster_and_validate(self._accumulated_features)
        if result is None:
            return False

        self.team0_color, self.team1_color = result
        self._configured = True

        logger.info(
            f"[TeamDetector] Auto-calibrated teams "
            f"(from {total} accumulated features):"
        )
        logger.info(
            f"  Team 0 HSV: {self.team0_color} -> {self._hsv_to_hex(self.team0_color)}"
        )
        logger.info(
            f"  Team 1 HSV: {self.team1_color} -> {self._hsv_to_hex(self.team1_color)}"
        )

        return True

    # ------------------------------------------------------------------
    # Public API: silent re-check (does NOT overwrite if significantly different)
    # ------------------------------------------------------------------

    def recheck(
        self,
        frame: np.ndarray,
        model,
        class_names: List[str],
        inference_dims: Tuple[int, int],
        device: str
    ) -> None:
        """
        Silent periodic re-check. Extracts features from the current frame,
        clusters them together with a recent subset of accumulated features,
        and applies drift correction if the new result is consistent.

        Does NOT change is_configured or broadcast anything — purely internal.
        """
        if not self._configured:
            return

        new_features = self._extract_frame_features(
            frame, model, class_names, inference_dims, device
        )
        if len(new_features) < 2:
            return

        # Use new features + recent tail of accumulated (for stability)
        recent = self._accumulated_features[-40:] if len(self._accumulated_features) > 40 else self._accumulated_features[:]
        combined = recent + new_features

        result = self._cluster_and_validate(combined)
        if result is None:
            return

        new_c0, new_c1 = result

        # Match new clusters to existing teams (closest assignment)
        d00 = self._color_distance(new_c0, self.team0_color)
        d01 = self._color_distance(new_c0, self.team1_color)
        d10 = self._color_distance(new_c1, self.team0_color)
        d11 = self._color_distance(new_c1, self.team1_color)

        if (d00 + d11) <= (d01 + d10):
            matched_c0, matched_c1 = new_c0, new_c1
        else:
            matched_c0, matched_c1 = new_c1, new_c0

        drift0 = self._color_distance(matched_c0, self.team0_color)
        drift1 = self._color_distance(matched_c1, self.team1_color)
        max_drift = max(drift0, drift1)

        if max_drift > 80.0:
            # Large drift — likely a bad frame or scene change; ignore
            logger.warning(
                f"[TeamDetector] Re-check drift too large "
                f"({drift0:.1f}, {drift1:.1f}), ignoring"
            )
            return

        # Apply gentle drift correction (exponential moving average)
        alpha = 0.2  # Blend factor: 20% new, 80% old
        self.team0_color = (1 - alpha) * self.team0_color + alpha * matched_c0
        self.team1_color = (1 - alpha) * self.team1_color + alpha * matched_c1

        logger.debug(
            f"[TeamDetector] Re-check applied drift correction "
            f"(drift={drift0:.1f}, {drift1:.1f})"
        )

    # ------------------------------------------------------------------
    # Public API: color output
    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_to_hex(hsv: np.ndarray) -> str:
        """Convert OpenCV HSV (H:0-179, S:0-255, V:0-255) to hex string."""
        h, s, v = hsv
        r, g, b = colorsys.hsv_to_rgb(h / 179.0, s / 255.0, v / 255.0)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def get_team_colors_hex(self) -> Tuple[str, str]:
        """Return the auto-calibrated colors as hex strings for the frontend."""
        if not self._configured:
            return "#ff0000", "#0000ff"  # fallback

        return self._hsv_to_hex(self.team0_color), self._hsv_to_hex(self.team1_color)

    # ------------------------------------------------------------------
    # Public API: per-shot classification (unchanged from v1)
    # ------------------------------------------------------------------

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
                dist = np.sqrt(
                    (center_x - shooter_pos['x'])**2 +
                    (bottom_y - shooter_pos['y'])**2
                )

                if dist < best_dist:
                    best_dist = dist
                    best_bbox = (x1, y1, x2, y2)
                    if box.id is not None:
                        best_track_id = int(box.id[0])

        if best_bbox is None:
            return None, 0.0, None

        team_id, confidence = self.classify_from_bbox(
            frame, best_bbox, track_id=best_track_id
        )
        return team_id, confidence, best_bbox
