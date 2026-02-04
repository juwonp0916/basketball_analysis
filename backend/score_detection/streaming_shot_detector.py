"""
StreamingShotDetector - Stateful shot detector for real-time frame-by-frame processing.

Adapted from ShotDetector for streaming use cases where frames arrive one at a time
via WebRTC rather than from a video file.
"""

import cv2
import math
import numpy as np
import os
import yaml
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from ultralytics import YOLO

# Load config relative to this file's location
_config_dir = os.path.dirname(__file__)
_config_path = os.path.join(_config_dir, 'config.yaml')
env = yaml.load(open(_config_path, 'r'), Loader=yaml.SafeLoader)

# Resolve relative paths
if not os.path.isabs(env['weights_path']):
    env['weights_path'] = os.path.join(_config_dir, env['weights_path'])
if not os.path.isabs(env['weights_path_shoot']):
    env['weights_path_shoot'] = os.path.join(_config_dir, env['weights_path_shoot'])


@dataclass
class ShotEvent:
    """Represents a detected shot event"""
    shot_id: int
    timestamp_ms: int
    sequence_id: int
    is_made: bool
    shooter_position: Optional[Dict[str, float]]  # {'x': float, 'y': float}
    court_position: Optional[Tuple[float, float]]  # (x_meters, y_meters)
    shot_type: Optional[str]  # '2pt' or '3pt'
    zone: Optional[int]


@dataclass
class DetectorState:
    """Holds mutable state for the streaming detector"""
    ball_pos: List = field(default_factory=list)
    hoop_pos: List = field(default_factory=list)
    shoot_pos: List = field(default_factory=list)
    pending_shot_group: List = field(default_factory=list)
    recent_shots: List = field(default_factory=list)

    last_group_processed_frame: int = -1
    last_shot_detection_frame: int = -1
    rim_last_detected: int = -1

    makes: int = 0
    attempts: int = 0
    attempt_cooldown: int = 0
    attempt_time: int = 0
    ball_entered: bool = False
    last_point_in_region: Any = None

    shot_id_counter: int = 0
    last_frame: Any = None  # Most recent frame for fallback person detection
    frame_history: List = field(default_factory=list)  # Rolling buffer of (det_frame, sequence_id, timestamp_ms)


class StreamingShotDetector:
    """
    Stateful shot detector for frame-by-frame processing.

    Unlike ShotDetector which processes entire video files, this class:
    - Receives frames via process_frame() instead of cv2.VideoCapture
    - Uses sequence_id from caller instead of internal frame_count
    - Returns shot events immediately when detected
    - Maintains detection state between frames
    """

    def __init__(
        self,
        calibration_points: Optional[List[List[float]]] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        frame_rate: float = 30.0
    ):
        """
        Initialize the streaming shot detector.

        Args:
            calibration_points: Optional 6-point calibration for court localization
            frame_width: Expected frame width in pixels
            frame_height: Expected frame height in pixels
            frame_rate: Expected frame rate (for timing calculations)
        """
        # Load YOLO models once
        self.model = YOLO(env['weights_path'], verbose=False)
        self.model_shoot = YOLO(env['weights_path_shoot'], verbose=False)

        self.class_names = env['classes']
        self.class_names_shoot = env['classes_shoot']

        # Frame dimensions
        self.width = frame_width
        self.height = frame_height
        self.frame_rate = frame_rate

        # Inference dimensions (rounded to multiples of 32)
        # Use 30% resolution for maximum CPU performance
        self.inference_width = ((int(frame_width * 0.3) + 31) // 32) * 32
        self.inference_height = ((int(frame_height * 0.3) + 31) // 32) * 32

        # Timing constants (based on frame rate) - reduced to avoid blocking legitimate shots
        self.MISS_ATTEMPT_COOLDOWN = int(frame_rate * 1.5)  # Reduced from 2.5s to 1.5s
        self.MADE_ATTEMPT_COOLDOWN = int(frame_rate * 2.0)  # Reduced from 3.0s to 2.0s
        self.ATTEMPT_DETECTION_INTERVAL = int(frame_rate * 0.3)

        # Deduplication constants
        from constants import (
            DEDUPLICATION_FRAME_THRESHOLD,
            DEDUPLICATION_POSITION_THRESHOLD,
            DEDUPLICATION_SAFETY_COOLDOWN_SEC
        )
        self.DEDUPLICATION_FRAME_THRESHOLD = DEDUPLICATION_FRAME_THRESHOLD
        self.DEDUPLICATION_POSITION_THRESHOLD = DEDUPLICATION_POSITION_THRESHOLD
        self.DEDUPLICATION_SAFETY_COOLDOWN = int(frame_rate * DEDUPLICATION_SAFETY_COOLDOWN_SEC)

        # Shot localization
        self.calibration_points = calibration_points
        self.shot_localizer = None
        self._setup_localization()

        # Mutable state
        self.state = DetectorState()

        # Device configuration
        self.device = env.get('device', 0)

    def _setup_localization(self) -> None:
        """Initialize shot localizer if calibration points are provided"""
        if self.calibration_points is None:
            return

        try:
            from localization import ShotLocalizer
            court_img_path = os.path.join(_config_dir, 'court_img_halfcourt.png')

            self.shot_localizer = ShotLocalizer(
                calibration_points=self.calibration_points,
                image_dimensions=(self.width, self.height),
                court_img_path=court_img_path,
                enable_visualization=False  # Disable file-based visualization for streaming
            )
        except Exception as e:
            print(f"[StreamingShotDetector] Failed to initialize localizer: {e}")
            self.shot_localizer = None

    def set_calibration(
        self,
        points: List[List[float]],
        dimensions: Tuple[int, int]
    ) -> bool:
        """
        Update calibration for live calibration support.

        Args:
            points: List of 6 calibration points [[x1,y1], ...]
            dimensions: (width, height) of the calibration frame

        Returns:
            True if calibration was successful
        """
        self.calibration_points = points
        self.width, self.height = dimensions
        # Use 30% resolution for maximum CPU performance
        self.inference_width = ((int(self.width * 0.3) + 31) // 32) * 32
        self.inference_height = ((int(self.height * 0.3) + 31) // 32) * 32

        try:
            self._setup_localization()
            return self.shot_localizer is not None
        except Exception:
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        sequence_id: int
    ) -> Optional[ShotEvent]:
        """
        Process a single frame and return shot event if detected.

        Args:
            frame: BGR frame from video/camera
            timestamp_ms: Timestamp in milliseconds
            sequence_id: Frame sequence number (provided by caller)

        Returns:
            ShotEvent if a shot was detected this frame, None otherwise
        """
        # Store frame for fallback person detection
        self.state.last_frame = frame

        # Skip every other frame for CPU performance (process only odd frames)
        # This is in addition to pipeline-level skipping
        if sequence_id % 2 != 0:
            # Still decrement cooldown even on skipped frames
            if self.state.attempt_cooldown > 0:
                self.state.attempt_cooldown -= 1
            return None

        # Resize frame for inference
        det_frame = cv2.resize(frame, (self.inference_width, self.inference_height))

        # Maintain a small rolling frame buffer for re-inference when
        # a shot is detected but shooter position is missing
        # Reduced buffer size from 30 to 15 to save memory
        self.state.frame_history.append((det_frame, sequence_id, timestamp_ms))
        if len(self.state.frame_history) > 15:
            self.state.frame_history.pop(0)

        # Run ball/rim detection with CPU optimizations
        results = self.model(
            det_frame,
            stream=True,
            verbose=False,
            imgsz=self.inference_width,
            device=self.device,
            conf=0.25,  # Filter low-confidence detections early (saves post-processing)
            iou=0.4,    # NMS threshold
            max_det=10  # Limit maximum detections per frame
        )

        ball_detected = False
        rim_detected = False

        for r in results:
            boxes = sorted(
                [(box.xyxy[0], box.conf, box.cls) for box in r.boxes],
                key=lambda x: -x[1]
            )

            for box in boxes:
                if ball_detected and rim_detected:
                    break

                x1, y1, x2, y2 = box[0]
                x1 = int(x1 * self.width / self.inference_width)
                y1 = int(y1 * self.height / self.inference_height)
                x2 = int(x2 * self.width / self.inference_width)
                y2 = int(y2 * self.height / self.inference_height)
                w, h = x2 - x1, y2 - y1

                conf = float(box[1])
                cls = int(box[2])
                current_class = self.class_names[cls]

                center = (int(x1 + w / 2), int(y1 + h / 2))

                if conf > 0.3:  # Lowered to match static_shot_localization.py threshold
                    if current_class == 'rim' and not rim_detected:
                        rim_detected = True
                        self.state.rim_last_detected = sequence_id
                        self.state.hoop_pos.append((center, sequence_id, w, h, conf))
                    elif current_class == 'ball' and not ball_detected:
                        ball_detected = True
                        self.state.ball_pos.append((center, sequence_id, w, h, conf))

        # Run shoot detection every 5 frames for CPU performance
        # Ball/rim tracking runs every frame (faster and more critical)
        shoot_detected = False
        if sequence_id % 5 == 0:
            shoot_detected = self._detect_shoot_class(
                det_frame, timestamp_ms, sequence_id
            )

        # Clean detection history
        self._clean_motion(sequence_id)

        # Check for shot events
        shot_event = self._score_detection(
            sequence_id, timestamp_ms, ball_detected
        )

        # Decrement cooldown
        if self.state.attempt_cooldown > 0:
            self.state.attempt_cooldown -= 1

        return shot_event

    def _detect_shoot_class(
        self,
        det_frame: np.ndarray,
        timestamp_ms: int,
        sequence_id: int
    ) -> bool:
        """Detect shoot class and shooter position"""
        results_shoot = self.model_shoot(
            det_frame,
            stream=True,
            verbose=False,
            imgsz=self.inference_width,
            device=self.device,
            conf=0.1,   # Low threshold for early filtering
            iou=0.4,
            max_det=15  # Limit detections (shoot + people)
        )

        shoot_detected = False

        for r in results_shoot:
            boxes_shoot = sorted(
                [(box.xyxy[0], box.conf, box.cls) for box in r.boxes],
                key=lambda x: -x[1]
            )

            for box in boxes_shoot:
                conf = float(box[1])
                cls = int(box[2])
                current_class = self.class_names_shoot[cls]

                if current_class == 'shoot' and conf > 0.15 and not shoot_detected:  # Lowered to 0.15 to catch more shooting motions
                    shoot_detected = True

                    # Scale shoot box
                    x1_shoot = int(box[0][0] * self.width / self.inference_width)
                    y1_shoot = int(box[0][1] * self.height / self.inference_height)
                    x2_shoot = int(box[0][2] * self.width / self.inference_width)
                    y2_shoot = int(box[0][3] * self.height / self.inference_height)

                    shooter_position = self._find_shooter_position(
                        boxes_shoot,
                        (x1_shoot, y1_shoot, x2_shoot, y2_shoot)
                    )

                    shot_data = {
                        'frame': sequence_id,
                        'timestamp': timestamp_ms,
                        'shoot_confidence': conf,
                        'shooter_position': shooter_position
                    }
                    self.state.shoot_pos.append(shot_data)
                    self.state.pending_shot_group.append(shot_data)
                    self.state.last_shot_detection_frame = sequence_id
                    break

        return shoot_detected

    def _find_shooter_position(
        self,
        boxes_shoot: List,
        shoot_box_coords: Tuple[int, int, int, int]
    ) -> Optional[Dict[str, float]]:
        """Find shooter position from person boxes based on overlap with shoot box"""
        x1_shoot, y1_shoot, x2_shoot, y2_shoot = shoot_box_coords
        shoot_area = (x2_shoot - x1_shoot) * (y2_shoot - y1_shoot)

        best_person = None
        best_overlap = 0.0

        for box_data in boxes_shoot:
            cls = int(box_data[2])
            class_name = self.class_names_shoot[cls]

            if class_name != 'person':
                continue

            x1_person = int(box_data[0][0] * self.width / self.inference_width)
            y1_person = int(box_data[0][1] * self.height / self.inference_height)
            x2_person = int(box_data[0][2] * self.width / self.inference_width)
            y2_person = int(box_data[0][3] * self.height / self.inference_height)

            # Calculate intersection
            x_left = max(x1_shoot, x1_person)
            y_top = max(y1_shoot, y1_person)
            x_right = min(x2_shoot, x2_person)
            y_bottom = min(y2_shoot, y2_person)

            if x_right < x_left or y_bottom < y_top:
                continue

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            overlap_ratio = intersection_area / shoot_area if shoot_area > 0 else 0

            if overlap_ratio >= 0.3 and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                center_x = (x1_person + x2_person) / 2
                bottom_y = y2_person
                best_person = {'x': center_x, 'y': bottom_y}

        return best_person

    def _find_nearest_person_to_ball(
        self,
        frame: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Fallback shooter detection: run the main model and find the person
        closest to the last known ball position (IoU-based, like static_shot_localization).
        Returns foot position (center-x, bottom-y) of the best match.
        """
        if len(self.state.ball_pos) == 0:
            return None

        ball_center = self.state.ball_pos[-1][0]
        ball_w = self.state.ball_pos[-1][2]
        ball_h = self.state.ball_pos[-1][3]
        ball_box = (
            ball_center[0] - ball_w // 2,
            ball_center[1] - ball_h // 2,
            ball_center[0] + ball_w // 2,
            ball_center[1] + ball_h // 2,
        )

        det_frame = cv2.resize(frame, (self.inference_width, self.inference_height))
        results = self.model(det_frame, stream=True, verbose=False,
                             imgsz=self.inference_width, device=self.device,
                             conf=0.25, max_det=10)

        best_person = None
        best_iou = 0.0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if self.class_names[cls] != 'person':
                    continue
                conf = float(box.conf[0])
                if conf < 0.3:  # Lowered from 0.4 to match static tool sensitivity
                    continue

                coords = box.xyxy[0]
                x1 = int(coords[0] * self.width / self.inference_width)
                y1 = int(coords[1] * self.height / self.inference_height)
                x2 = int(coords[2] * self.width / self.inference_width)
                y2 = int(coords[3] * self.height / self.inference_height)

                # IoU between ball box and person box
                ix1 = max(ball_box[0], x1)
                iy1 = max(ball_box[1], y1)
                ix2 = min(ball_box[2], x2)
                iy2 = min(ball_box[3], y2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter = (ix2 - ix1) * (iy2 - iy1)
                area1 = (ball_box[2] - ball_box[0]) * (ball_box[3] - ball_box[1])
                area2 = (x2 - x1) * (y2 - y1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    center_x = (x1 + x2) / 2
                    bottom_y = y2  # foot position
                    best_person = {'x': center_x, 'y': bottom_y}

        return best_person

    def _reinfer_shooter_from_history(self) -> Optional[Dict[str, float]]:
        """
        Scan recent frame history to find the shooter position.

        Mirrors the video detector's _process_shot_detection: re-runs the shoot
        model on stored frames to find a 'shoot' class detection, then matches
        to a person box to get foot position.

        Only checks the 10 most recent frames to avoid blocking the pipeline.
        """
        # Limit to last 5 frames to keep latency reasonable (reduced from 10)
        frames_to_check = self.state.frame_history[-5:]
        for det_frame, seq_id, ts in reversed(frames_to_check):
            try:
                results_shoot = self.model_shoot(
                    det_frame, stream=True, verbose=False,
                    imgsz=self.inference_width, device=self.device,
                    conf=0.1, max_det=15
                )
                for r in results_shoot:
                    boxes_shoot = sorted(
                        [(box.xyxy[0], box.conf, box.cls) for box in r.boxes],
                        key=lambda x: -x[1]
                    )
                    for box in boxes_shoot:
                        conf = float(box[1])
                        cls = int(box[2])
                        if self.class_names_shoot[cls] == 'shoot' and conf > 0.15:  # Lowered to match main detection threshold
                            x1 = int(box[0][0] * self.width / self.inference_width)
                            y1 = int(box[0][1] * self.height / self.inference_height)
                            x2 = int(box[0][2] * self.width / self.inference_width)
                            y2 = int(box[0][3] * self.height / self.inference_height)
                            pos = self._find_shooter_position(
                                boxes_shoot, (x1, y1, x2, y2)
                            )
                            if pos:
                                return pos
            except Exception:
                continue
        return None

    def _clean_motion(self, sequence_id: int) -> None:
        """Clean detection history to remove stale/erroneous points"""
        from utils import clean_ball_pos, clean_hoop_pos

        self.state.ball_pos = clean_ball_pos(self.state.ball_pos, sequence_id)
        if len(self.state.hoop_pos) > 1:
            self.state.hoop_pos = clean_hoop_pos(self.state.hoop_pos)

    def _score_detection(
        self,
        sequence_id: int,
        timestamp_ms: int,
        ball_detected: bool
    ) -> Optional[ShotEvent]:
        """Check for shot events using both shoot class and fallback detection"""
        from utils import in_score_region, detect_score

        frames_since_last_detection = sequence_id - self.state.last_shot_detection_frame

        # Process accumulated shot group when enough time has passed
        if (self.state.attempt_cooldown == 0 and
            len(self.state.pending_shot_group) > 0 and
            frames_since_last_detection > self.DEDUPLICATION_FRAME_THRESHOLD):

            representative_shot = self._deduplicate_shot_group(
                self.state.pending_shot_group
            )

            if representative_shot:
                # If shooter position missing, try re-inference on recent frame history
                # (mirrors the video detector's _process_shot_detection frame_track scan)
                if not representative_shot.get('shooter_position'):
                    shooter_pos = self._reinfer_shooter_from_history()
                    if shooter_pos:
                        representative_shot['shooter_position'] = shooter_pos
                    else:
                        # Last resort: IoU on the current frame
                        if self.state.last_frame is not None:
                            fallback_pos = self._find_nearest_person_to_ball(self.state.last_frame)
                            if fallback_pos:
                                representative_shot['shooter_position'] = fallback_pos
                    if not representative_shot.get('shooter_position'):
                        self.state.pending_shot_group = []
                        return None

                # Stale detection filter
                frames_old = sequence_id - representative_shot['frame']
                if frames_old > self.DEDUPLICATION_FRAME_THRESHOLD * 1.5:
                    self.state.pending_shot_group = []
                    return None

                # Cross-method deduplication
                if self._is_duplicate_of_recent_shot(representative_shot):
                    self.state.pending_shot_group = []
                    return None

                # Determine make/miss
                is_scored = self._check_score_region()

                # Create shot event
                shot_event = self._create_shot_event(
                    representative_shot, is_scored, sequence_id
                )

                # Update state
                if is_scored:
                    self.state.makes += 1
                    self.state.attempt_cooldown = self.MADE_ATTEMPT_COOLDOWN
                else:
                    self.state.attempt_cooldown = self.MISS_ATTEMPT_COOLDOWN

                self.state.attempts += 1
                self.state.attempt_cooldown = max(
                    self.state.attempt_cooldown,
                    self.DEDUPLICATION_SAFETY_COOLDOWN
                )
                self.state.last_point_in_region = None
                self._add_to_recent_shots(representative_shot, sequence_id)
                self.state.pending_shot_group = []
                self.state.last_group_processed_frame = sequence_id

                return shot_event

        # Fallback: score region detection
        elif len(self.state.hoop_pos) > 0 and len(self.state.ball_pos) > 0:
            shot_event = self._fallback_score_detection(
                sequence_id, timestamp_ms, ball_detected
            )
            if shot_event:
                return shot_event

        return None

    def _check_score_region(self) -> bool:
        """Check if ball is in score region and determine if made"""
        from utils import in_score_region, detect_score

        if len(self.state.hoop_pos) == 0 or len(self.state.ball_pos) == 0:
            return False

        if in_score_region(self.state.ball_pos, self.state.hoop_pos):
            if self.state.last_point_in_region:
                is_scored = detect_score(
                    self.state.ball_pos,
                    self.state.hoop_pos,
                    self.state.last_point_in_region
                )
                self.state.last_point_in_region = self.state.ball_pos[-1]
                return is_scored
            else:
                self.state.last_point_in_region = self.state.ball_pos[-1]

        return False

    def _fallback_score_detection(
        self,
        sequence_id: int,
        timestamp_ms: int,
        ball_detected: bool
    ) -> Optional[ShotEvent]:
        """Fallback detection using ball trajectory in score region"""
        from utils import in_score_region, detect_score

        if (sequence_id - self.state.rim_last_detected >= self.frame_rate or
            not ball_detected or
            self.state.attempt_cooldown != 0):
            return None

        if in_score_region(self.state.ball_pos, self.state.hoop_pos):
            if self.state.ball_entered:
                self.state.attempt_time += 1
            else:
                self.state.ball_entered = True
                self.state.attempt_time = 1

            if not self.state.last_point_in_region:
                self.state.last_point_in_region = self.state.ball_pos[-1]
                scored = False
            else:
                scored = detect_score(
                    self.state.ball_pos,
                    self.state.hoop_pos,
                    self.state.last_point_in_region
                )

            if scored:
                self.state.makes += 1
                self.state.attempts += 1

                # Find shooter via nearest person to ball (foot position)
                shooter_pos = None
                if self.state.last_frame is not None:
                    shooter_pos = self._find_nearest_person_to_ball(self.state.last_frame)
                if shooter_pos is None and self.state.ball_pos:
                    ball_x, ball_y = self.state.ball_pos[-1][0]
                    shooter_pos = {'x': ball_x, 'y': ball_y}

                # Create shot event for made shot
                shot_data = {
                    'frame': sequence_id,
                    'timestamp': timestamp_ms,
                    'shooter_position': shooter_pos
                }
                shot_event = self._create_shot_event(shot_data, True, sequence_id)

                self.state.attempt_cooldown = self.MADE_ATTEMPT_COOLDOWN
                self.state.last_point_in_region = None
                self.state.ball_entered = False
                self.state.attempt_time = 0

                # Clear pending to prevent duplicates
                if len(self.state.pending_shot_group) > 0:
                    self.state.pending_shot_group = []

                self._add_to_recent_shots(shot_data, sequence_id)
                return shot_event
            else:
                self.state.last_point_in_region = self.state.ball_pos[-1]
        else:
            if self.state.ball_entered:
                self.state.attempt_time += 1

            if self.state.attempt_time >= self.ATTEMPT_DETECTION_INTERVAL:
                self.state.attempts += 1

                # Find shooter via nearest person to ball (foot position)
                shooter_pos = None
                if self.state.last_frame is not None:
                    shooter_pos = self._find_nearest_person_to_ball(self.state.last_frame)
                if shooter_pos is None and self.state.ball_pos:
                    ball_x, ball_y = self.state.ball_pos[-1][0]
                    shooter_pos = {'x': ball_x, 'y': ball_y}

                # Create shot event for missed shot
                shot_data = {
                    'frame': sequence_id,
                    'timestamp': timestamp_ms,
                    'shooter_position': shooter_pos
                }
                shot_event = self._create_shot_event(shot_data, False, sequence_id)

                self.state.attempt_cooldown = self.MISS_ATTEMPT_COOLDOWN
                self.state.attempt_time = 0
                self.state.ball_entered = False
                self.state.last_point_in_region = None

                if len(self.state.pending_shot_group) > 0:
                    self.state.pending_shot_group = []

                self._add_to_recent_shots(shot_data, sequence_id)
                return shot_event

        return None

    def _create_shot_event(
        self,
        shot_data: Dict,
        is_made: bool,
        sequence_id: int
    ) -> ShotEvent:
        """Create a ShotEvent from shot data"""
        self.state.shot_id_counter += 1

        shooter_pos = shot_data.get('shooter_position')
        court_position = None
        zone = None
        shot_type = None

        # Transform to court coordinates if localization enabled
        if self.shot_localizer and shooter_pos:
            try:
                court_position = self.shot_localizer.map_to_court(
                    (shooter_pos['x'], shooter_pos['y'])
                )

                if court_position[0] is not None and court_position[1] is not None:
                    from statistics import TeamStatistics
                    stats = TeamStatistics(quarters=[float('inf')])
                    zone = stats.determine_zone(court_position[0], court_position[1])

                    if zone:
                        is_three = stats.determine_is_three_pt(zone)
                        shot_type = '3pt' if is_three else '2pt'
            except Exception as e:
                print(f"[StreamingShotDetector] Court localization failed for "
                      f"shooter_pos=({shooter_pos['x']:.1f}, {shooter_pos['y']:.1f}): {e}")

        return ShotEvent(
            shot_id=self.state.shot_id_counter,
            timestamp_ms=shot_data.get('timestamp', 0),
            sequence_id=sequence_id,
            is_made=is_made,
            shooter_position=shooter_pos,
            court_position=court_position,
            shot_type=shot_type,
            zone=zone
        )

    def _deduplicate_shot_group(self, shot_group: List) -> Optional[Dict]:
        """Deduplicate shot group and return representative shot"""
        if not shot_group:
            return None

        sorted_shots = sorted(shot_group, key=lambda s: s['frame'])

        subgroups = []
        current_subgroup = [sorted_shots[0]]

        for i in range(1, len(sorted_shots)):
            current = sorted_shots[i]
            last = current_subgroup[-1]

            frame_dist = current['frame'] - last['frame']

            pos_dist = float('inf')
            if current['shooter_position'] and last['shooter_position']:
                p1 = current['shooter_position']
                p2 = last['shooter_position']
                pos_dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

            if (frame_dist <= self.DEDUPLICATION_FRAME_THRESHOLD and
                pos_dist <= self.DEDUPLICATION_POSITION_THRESHOLD):
                current_subgroup.append(current)
            else:
                subgroups.append(current_subgroup)
                current_subgroup = [current]

        if current_subgroup:
            subgroups.append(current_subgroup)

        representative = subgroups[0][0].copy()
        representative['grouped_frames'] = len(subgroups[0])
        representative['total_raw_detections'] = len(shot_group)

        return representative

    def _is_duplicate_of_recent_shot(self, shot_data: Dict) -> bool:
        """Check if shot is duplicate of a recent shot"""
        if not self.state.recent_shots:
            return False

        current_frame = shot_data['frame']
        current_pos = shot_data['shooter_position']

        for recent_shot in self.state.recent_shots:
            recent_frame, recent_timestamp, recent_pos = recent_shot

            if not recent_pos:
                continue

            frame_dist = abs(current_frame - recent_frame)
            if frame_dist > self.DEDUPLICATION_FRAME_THRESHOLD:
                continue

            pos_dist = np.sqrt(
                (current_pos['x'] - recent_pos['x'])**2 +
                (current_pos['y'] - recent_pos['y'])**2
            )

            if pos_dist <= self.DEDUPLICATION_POSITION_THRESHOLD:
                return True

        return False

    def _add_to_recent_shots(self, shot_data: Dict, current_frame: int) -> None:
        """Add shot to recent shots for deduplication"""
        if not shot_data:
            return

        pos = shot_data.get('shooter_position')
        self.state.recent_shots.append((
            shot_data['frame'],
            shot_data.get('timestamp', 0),
            pos
        ))

        # Keep only recent shots
        max_history = self.DEDUPLICATION_FRAME_THRESHOLD * 2
        self.state.recent_shots = [
            shot for shot in self.state.recent_shots
            if current_frame - shot[0] <= max_history
        ]

    def get_current_stats(self) -> Dict[str, Any]:
        """Return accumulated statistics"""
        total = self.state.attempts
        made = self.state.makes

        return {
            'total_shots': total,
            'made_shots': made,
            'missed_shots': total - made,
            'field_goal_percentage': (made / total * 100.0) if total > 0 else 0.0
        }

    def reset(self) -> None:
        """Reset all state for new session"""
        self.state = DetectorState()
