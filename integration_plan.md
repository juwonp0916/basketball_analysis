# Real-time Shot Localization Integration Plan

## Overview

This document provides step-by-step instructions to integrate the shot localization module with the WebRTC streaming infrastructure. The goal is to process live video frames from the frontend, detect shots using YOLO models, localize them to court coordinates via homography, and broadcast results back to the frontend in real-time.

---

## Prerequisites

Before starting, ensure you understand these key files:
- `backend/src/utils/webrtc_manager.py` - WebRTC connection handling and frame buffering
- `backend/score_detection/shot_detector.py` - Batch video shot detection (reference implementation)
- `backend/score_detection/localization.py` - Court coordinate mapping via homography
- `backend/schema.py` - Pydantic models for WebRTC messages
- `backend/main.py` - FastAPI application entry point

---

## Step 1: Fix Existing Bugs

### 1.1 Fix TrackConsumer method call
**File:** `backend/src/utils/webrtc_manager.py`
**Line:** 55

Change:
```python
self.buffer.add_frame(frame)
```
To:
```python
await self.buffer.put_frame(frame)
```

### 1.2 Fix SharedFrameBuffer initialization
**File:** `backend/src/utils/webrtc_manager.py`
**Line:** 71

Change:
```python
self.frame_buffer = SharedFrameBuffer(maxlen=60)
```
To:
```python
self.frame_buffer = SharedFrameBuffer(maxsize=60)
```

### 1.3 Add ShotDetectedPayload alias
**File:** `backend/schema.py`
**After line 50**, add:
```python
# Alias for backward compatibility with webrtc_manager.py
ShotDetectedPayload = ShotPayload
```

---

## Step 2: Create StreamingShotDetector

**Create new file:** `backend/score_detection/streaming_shot_detector.py`

This class adapts `ShotDetector` for frame-by-frame processing. Key differences:
- No `cv2.VideoCapture` - frames passed via `process_frame()` method
- No blocking `run()` loop - caller controls frame rate
- Uses `sequence_id` parameter instead of internal frame counter
- Same YOLO models and detection logic as `ShotDetector`
- Same 3-level deduplication logic

### Class Structure:

```python
import numpy as np
import math
import time
from typing import Optional, List, Tuple, Dict, Any
from ultralytics import YOLO
from pathlib import Path
import os

# Load config
import yaml
_config_dir = os.path.dirname(__file__)
_config_path = os.path.join(_config_dir, 'config.yaml')
env = yaml.load(open(_config_path, 'r'), Loader=yaml.SafeLoader)

# Resolve model paths
if not os.path.isabs(env['weights_path']):
    env['weights_path'] = os.path.join(_config_dir, env['weights_path'])
if not os.path.isabs(env['weights_path_shoot']):
    env['weights_path_shoot'] = os.path.join(_config_dir, env['weights_path_shoot'])


class StreamingShotDetector:
    """
    Real-time shot detector for streaming video frames.
    Adapted from ShotDetector for frame-by-frame processing.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        calibration_points: Optional[List[List[float]]] = None,
        target_fps: float = 30.0
    ):
        """
        Initialize streaming shot detector.

        Args:
            frame_width: Expected frame width
            frame_height: Expected frame height
            calibration_points: 6 calibration points for court localization [[x,y], ...]
            target_fps: Expected frame rate (for cooldown calculations)
        """
        # Frame dimensions
        self.width = frame_width
        self.height = frame_height
        self.frame_rate = target_fps

        # Calculate inference dimensions (multiples of 32)
        self.inference_width = ((frame_width + 31) // 32) * 32
        self.inference_height = ((frame_height + 31) // 32) * 32

        # Load YOLO models (only once)
        self.model = YOLO(env['weights_path'], verbose=False)
        self.model_shoot = YOLO(env['weights_path_shoot'], verbose=False)
        self.class_names = env['classes']
        self.class_names_shoot = env['classes_shoot']

        # Detection state (same as ShotDetector)
        self.ball_pos = []      # ((x, y), frame, w, h, conf)
        self.hoop_pos = []      # ((x, y), frame, w, h, conf)
        self.shoot_pos = []     # [{'frame', 'timestamp', 'shoot_confidence', 'shooter_position'}, ...]
        self.pending_shot_group = []
        self.recent_shots = []  # For cross-method deduplication

        # Frame tracking
        self.sequence_id = 0
        self.last_shot_detection_frame = -1
        self.rim_last_detected = -1

        # Cooldowns
        from constants import (
            DEDUPLICATION_FRAME_THRESHOLD,
            DEDUPLICATION_POSITION_THRESHOLD,
            DEDUPLICATION_SAFETY_COOLDOWN_SEC
        )
        self.DEDUPLICATION_FRAME_THRESHOLD = DEDUPLICATION_FRAME_THRESHOLD
        self.DEDUPLICATION_POSITION_THRESHOLD = DEDUPLICATION_POSITION_THRESHOLD
        self.DEDUPLICATION_SAFETY_COOLDOWN = int(target_fps * DEDUPLICATION_SAFETY_COOLDOWN_SEC)
        self.MISS_ATTEMPT_COOLDOWN = int(target_fps * 2.5)
        self.MADE_ATTEMPT_COOLDOWN = int(target_fps * 3)
        self.attempt_cooldown = 0

        # Score region tracking
        self.ball_entered = False
        self.attempt_time = 0
        self.last_point_in_region = None

        # Statistics
        self.makes = 0
        self.attempts = 0
        self.two_made = 0
        self.two_total = 0
        self.three_made = 0
        self.three_total = 0

        # Localization
        self.calibration_points = calibration_points
        self.shot_localizer = None
        if calibration_points:
            self._init_localizer(calibration_points)

    def _init_localizer(self, calibration_points: List[List[float]]):
        """Initialize shot localizer with calibration points"""
        from localization import ShotLocalizer
        court_img_path = os.path.join(_config_dir, 'court_img_halfcourt.png')
        self.shot_localizer = ShotLocalizer(
            calibration_points=calibration_points,
            image_dimensions=(self.width, self.height),
            court_img_path=court_img_path,
            enable_visualization=False  # Disable for streaming
        )

    def set_calibration(self, points: List[List[float]], dimensions: Tuple[int, int]):
        """Update calibration points (can be called during streaming)"""
        self.calibration_points = points
        self.width, self.height = dimensions
        self._init_localizer(points)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        sequence_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single frame.

        Args:
            frame: BGR numpy array (original resolution)
            timestamp_ms: Timestamp in milliseconds
            sequence_id: Frame sequence number

        Returns:
            Shot event dict if shot detected, else None
        """
        self.sequence_id = sequence_id

        # Resize for inference
        det_frame = cv2.resize(frame, (self.inference_width, self.inference_height))

        # Run ball/rim detection (every frame)
        self._detect_ball_rim(det_frame, sequence_id)

        # Run shoot detection (every 3 frames)
        if sequence_id % 3 == 0:
            self._detect_shoot(det_frame, sequence_id, timestamp_ms)

        # Clean old detections
        self._clean_motion(sequence_id)

        # Check for shot events
        shot_event = self._score_detection(sequence_id, timestamp_ms)

        # Decrement cooldown
        if self.attempt_cooldown > 0:
            self.attempt_cooldown -= 1

        return shot_event

    def _detect_ball_rim(self, det_frame: np.ndarray, sequence_id: int):
        """Run ball/rim detection model"""
        results = self.model(det_frame, stream=True, verbose=False,
                           imgsz=self.inference_width, device=env.get('device', 0))

        ball_detected, rim_detected = False, False

        for r in results:
            boxes = sorted([(box.xyxy[0], box.conf, box.cls) for box in r.boxes],
                          key=lambda x: -x[1])

            for box in boxes:
                if ball_detected and rim_detected:
                    break

                x1, y1, x2, y2 = box[0]
                # Scale to original dimensions
                x1 = int(x1 * self.width / self.inference_width)
                y1 = int(y1 * self.height / self.inference_height)
                x2 = int(x2 * self.width / self.inference_width)
                y2 = int(y2 * self.height / self.inference_height)
                w, h = x2 - x1, y2 - y1
                conf = float(box[1])
                cls = int(box[2])
                current_class = self.class_names[cls]
                center = (int(x1 + w / 2), int(y1 + h / 2))

                if conf > 0.7:
                    if current_class == 'rim' and not rim_detected:
                        rim_detected = True
                        self.rim_last_detected = sequence_id
                        self.hoop_pos.append((center, sequence_id, w, h, conf))
                    elif current_class == 'ball' and not ball_detected:
                        ball_detected = True
                        self.ball_pos.append((center, sequence_id, w, h, conf))

    def _detect_shoot(self, det_frame: np.ndarray, sequence_id: int, timestamp_ms: int):
        """Run shoot detection model"""
        results = self.model_shoot(det_frame, stream=True, verbose=False,
                                  imgsz=self.inference_width, device=env.get('device', 0))

        for r in results:
            boxes = sorted([(box.xyxy[0], box.conf, box.cls) for box in r.boxes],
                          key=lambda x: -x[1])

            for box in boxes:
                conf = float(box[1])
                cls = int(box[2])
                current_class = self.class_names_shoot[cls]

                if current_class == 'shoot' and conf > 0.3:
                    # Scale shoot box
                    x1 = int(box[0][0] * self.width / self.inference_width)
                    y1 = int(box[0][1] * self.height / self.inference_height)
                    x2 = int(box[0][2] * self.width / self.inference_width)
                    y2 = int(box[0][3] * self.height / self.inference_height)

                    # Find shooter position
                    shooter_position = self._find_shooter_position(boxes, (x1, y1, x2, y2))

                    shot_data = {
                        'frame': sequence_id,
                        'timestamp': timestamp_ms,
                        'shoot_confidence': conf,
                        'shooter_position': shooter_position
                    }
                    self.shoot_pos.append(shot_data)
                    self.pending_shot_group.append(shot_data)
                    self.last_shot_detection_frame = sequence_id
                    break

    def _find_shooter_position(self, boxes, shoot_box_coords) -> Optional[Dict[str, float]]:
        """Find shooter position from person boxes (same as ShotDetector)"""
        x1_shoot, y1_shoot, x2_shoot, y2_shoot = shoot_box_coords
        shoot_area = (x2_shoot - x1_shoot) * (y2_shoot - y1_shoot)

        best_person = None
        best_overlap = 0.0

        for box_data in boxes:
            cls = int(box_data[2])
            if self.class_names_shoot[cls] != 'person':
                continue

            x1_p = int(box_data[0][0] * self.width / self.inference_width)
            y1_p = int(box_data[0][1] * self.height / self.inference_height)
            x2_p = int(box_data[0][2] * self.width / self.inference_width)
            y2_p = int(box_data[0][3] * self.height / self.inference_height)

            # Calculate intersection
            x_left = max(x1_shoot, x1_p)
            y_top = max(y1_shoot, y1_p)
            x_right = min(x2_shoot, x2_p)
            y_bottom = min(y2_shoot, y2_p)

            if x_right < x_left or y_bottom < y_top:
                continue

            intersection = (x_right - x_left) * (y_bottom - y_top)
            overlap = intersection / shoot_area if shoot_area > 0 else 0

            if overlap >= 0.7 and overlap > best_overlap:
                best_overlap = overlap
                center_x = (x1_p + x2_p) / 2
                bottom_y = y2_p
                best_person = {'x': center_x, 'y': bottom_y}

        return best_person

    def _clean_motion(self, sequence_id: int):
        """Clean old ball/hoop positions"""
        from utils import clean_ball_pos, clean_hoop_pos
        self.ball_pos = clean_ball_pos(self.ball_pos, sequence_id)
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

    def _score_detection(self, sequence_id: int, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        """Check for shot events (same logic as ShotDetector.score_detection)"""
        # Import utilities
        from utils import in_score_region, detect_score

        frames_since_detection = sequence_id - self.last_shot_detection_frame

        # Process pending shot group after deduplication window
        if (self.attempt_cooldown == 0 and
            len(self.pending_shot_group) > 0 and
            frames_since_detection > self.DEDUPLICATION_FRAME_THRESHOLD):

            representative = self._deduplicate_shot_group(self.pending_shot_group)

            if representative:
                # Quality filter: require shooter position
                if not representative.get('shooter_position'):
                    self.pending_shot_group = []
                    return None

                # Cross-method deduplication
                if self._is_duplicate_of_recent_shot(representative):
                    self.pending_shot_group = []
                    return None

                # Determine make/miss
                is_scored = False
                if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
                    if in_score_region(self.ball_pos, self.hoop_pos):
                        if self.last_point_in_region:
                            is_scored = detect_score(self.ball_pos, self.hoop_pos, self.last_point_in_region)

                # Update stats
                self.attempts += 1
                if is_scored:
                    self.makes += 1
                    self.attempt_cooldown = self.MADE_ATTEMPT_COOLDOWN
                else:
                    self.attempt_cooldown = self.MISS_ATTEMPT_COOLDOWN

                # Track for deduplication
                self._add_to_recent_shots(representative)
                self.pending_shot_group = []

                # Create shot event
                return self._create_shot_event(representative, is_scored, timestamp_ms)

        # Fallback: score region detection (same as ShotDetector)
        # ... (implement fallback logic from ShotDetector.score_detection lines 700-843)

        return None

    def _deduplicate_shot_group(self, shot_group):
        """Deduplicate shot group (same as ShotDetector)"""
        # ... (copy from ShotDetector._deduplicate_shot_group)

    def _is_duplicate_of_recent_shot(self, shot_data) -> bool:
        """Check cross-method duplicates (same as ShotDetector)"""
        # ... (copy from ShotDetector._is_duplicate_of_recent_shot)

    def _add_to_recent_shots(self, shot_data):
        """Track shot for deduplication (same as ShotDetector)"""
        # ... (copy from ShotDetector._add_to_recent_shots)

    def _create_shot_event(self, shot_data: Dict, is_scored: bool, timestamp_ms: int) -> Dict[str, Any]:
        """Create shot event payload"""
        shooter_pos = shot_data.get('shooter_position', {})
        shot_location = (shooter_pos.get('x'), shooter_pos.get('y')) if shooter_pos else (None, None)

        # Map to court coordinates if localization enabled
        court_position = None
        zone = None
        shot_type = "2pt"

        if self.shot_localizer and shot_location[0] is not None:
            court_position = self.shot_localizer.map_to_court(shot_location)

            if court_position[0] is not None:
                from statistics import TeamStatistics
                stats = TeamStatistics(quarters=[float('inf')])
                zone = stats.determine_zone(court_position[0], court_position[1])

                if zone:
                    is_three = stats.determine_is_three_pt(zone)
                    shot_type = "3pt" if is_three else "2pt"

                    # Update shot type stats
                    if shot_type == "3pt":
                        self.three_total += 1
                        if is_scored:
                            self.three_made += 1
                    else:
                        self.two_total += 1
                        if is_scored:
                            self.two_made += 1

        return {
            'shot_detected': True,
            'timestamp_ms': timestamp_ms,
            'is_made': is_scored,
            'shot_type': shot_type,
            'pixel_location': shot_location,
            'court_location': court_position,
            'zone': zone
        }

    def get_current_stats(self) -> Dict[str, Any]:
        """Return current accumulated statistics"""
        return {
            'totalShots': {
                'made': self.makes,
                'total': self.attempts
            },
            'percentages': {
                'fieldGoal': (self.makes / self.attempts * 100) if self.attempts > 0 else 0.0,
                'twoPoint': (self.two_made / self.two_total * 100) if self.two_total > 0 else 0.0,
                'threePoint': (self.three_made / self.three_total * 100) if self.three_total > 0 else 0.0
            }
        }

    def reset(self):
        """Reset all detection state and statistics"""
        self.ball_pos = []
        self.hoop_pos = []
        self.shoot_pos = []
        self.pending_shot_group = []
        self.recent_shots = []
        self.sequence_id = 0
        self.last_shot_detection_frame = -1
        self.makes = 0
        self.attempts = 0
        self.two_made = 0
        self.two_total = 0
        self.three_made = 0
        self.three_total = 0
        self.attempt_cooldown = 0
        self.ball_entered = False
        self.attempt_time = 0
        self.last_point_in_region = None
```

**Note:** The class above is a template. Copy the full implementations of these methods from `shot_detector.py`:
- `_deduplicate_shot_group()` (lines 910-969)
- `_is_duplicate_of_recent_shot()` (lines 971-1022)
- `_add_to_recent_shots()` (lines 1024-1052)
- Fallback score detection logic (lines 700-843)

---

## Step 3: Create Shot Processing Pipeline

**Create new file:** `backend/score_detection/shot_processing_pipeline.py`

```python
import asyncio
import time
import logging
from typing import Optional, Callable, Awaitable, List, Tuple

from streaming_shot_detector import StreamingShotDetector

logger = logging.getLogger(__name__)


class ShotProcessingPipeline:
    """
    Async pipeline that connects SharedFrameBuffer to StreamingShotDetector
    and broadcasts results via WebRTC data channel.
    """

    def __init__(
        self,
        frame_buffer,  # SharedFrameBuffer instance
        broadcaster: Callable[[dict], Awaitable[None]],  # async broadcast function
        calibration_points: Optional[List[List[float]]] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        skip_threshold: int = 30  # Skip frames if buffer backs up beyond this
    ):
        """
        Initialize pipeline.

        Args:
            frame_buffer: SharedFrameBuffer to consume frames from
            broadcaster: Async function to broadcast payloads to frontend
            calibration_points: 6-point calibration for court localization
            frame_width: Expected frame width
            frame_height: Expected frame height
            skip_threshold: Frame buffer size threshold for frame skipping
        """
        self.frame_buffer = frame_buffer
        self.broadcaster = broadcaster
        self.skip_threshold = skip_threshold

        self.detector = StreamingShotDetector(
            frame_width=frame_width,
            frame_height=frame_height,
            calibration_points=calibration_points
        )

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sequence_id = 0
        self._shot_id = 0

    async def start(self):
        """Start the processing loop"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Shot processing pipeline started")

    async def stop(self):
        """Stop the processing loop gracefully"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Shot processing pipeline stopped")

    def set_calibration(self, points: List[List[float]], dimensions: Tuple[int, int]):
        """Update calibration (thread-safe for async context)"""
        self.detector.set_calibration(points, dimensions)

    def reset_stats(self):
        """Reset detector statistics"""
        self.detector.reset()
        self._shot_id = 0

    async def _process_loop(self):
        """Main async processing loop"""
        from schema import FrameSyncPayload, ShotPayload, Shot, Point, GameStats, TotalShots, Percentages

        while self._running:
            try:
                # Get next frame from buffer
                frame = await self.frame_buffer.get_next()

                # Convert to numpy array (BGR format)
                img = frame.to_ndarray(format="bgr24")
                timestamp_ms = int(time.time() * 1000)

                # Frame skipping if falling behind
                buffer_size = self.frame_buffer._q.qsize()
                if buffer_size > self.skip_threshold:
                    logger.debug(f"Skipping frame, buffer size: {buffer_size}")
                    continue

                # Run detection in thread pool (YOLO is CPU-bound)
                result = await asyncio.to_thread(
                    self.detector.process_frame,
                    img,
                    timestamp_ms,
                    self._sequence_id
                )

                self._sequence_id += 1

                # Get current stats
                stats = self.detector.get_current_stats()
                game_stats = GameStats(
                    totalShots=TotalShots(**stats['totalShots']),
                    percentages=Percentages(**stats['percentages'])
                )

                # Broadcast frame sync
                frame_sync = FrameSyncPayload(
                    sequence_id=self._sequence_id,
                    video_timestamp_ms=timestamp_ms,
                    current_stats=game_stats
                )
                await self.broadcaster(frame_sync.model_dump())

                # Broadcast shot event if detected
                if result and result.get('shot_detected'):
                    self._shot_id += 1

                    # Determine location string
                    zone = result.get('zone')
                    location_str = self._zone_to_location_string(zone) if zone else "Unknown"

                    shot = Shot(
                        id=self._shot_id,
                        timestamp_ms=timestamp_ms,
                        type=result.get('shot_type', '2pt'),
                        result="made" if result.get('is_made') else "missed",
                        location=location_str,
                        coord=Point(
                            x=result['court_location'][0] if result.get('court_location') else 0.0,
                            y=result['court_location'][1] if result.get('court_location') else 0.0
                        )
                    )

                    shot_payload = ShotPayload(
                        video_timestamp_ms=timestamp_ms,
                        event_data=shot,
                        updated_stats=game_stats
                    )
                    await self.broadcaster(shot_payload.model_dump())

                    logger.info(f"Shot detected: #{self._shot_id} {shot.type} {shot.result} at {location_str}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop

    def _zone_to_location_string(self, zone: int) -> str:
        """Convert zone number to human-readable location"""
        zone_names = {
            1: "Left Corner",
            2: "Left Wing",
            3: "Top of Key",
            4: "Right Wing",
            5: "Right Corner",
            6: "Left Paint",
            7: "Center Paint",
            8: "Right Paint",
            9: "Restricted Area"
        }
        return zone_names.get(zone, "Unknown")
```

---

## Step 4: Add Calibration API Endpoints

**Modify:** `backend/main.py`

Add these imports at the top:
```python
from typing import List, Optional
from pydantic import BaseModel
```

Add calibration request model:
```python
class CalibrationRequest(BaseModel):
    points: List[List[float]]  # 6 points: [[x1, y1], [x2, y2], ...]
    image_width: int
    image_height: int
```

Add these endpoints:

```python
@app.post("/calibration")
async def set_calibration(request: CalibrationRequest):
    """Set 6-point court calibration for shot localization"""
    if len(request.points) != 6:
        return {"success": False, "error": f"Expected 6 points, got {len(request.points)}"}

    # Validate points are within image bounds
    for i, point in enumerate(request.points):
        if len(point) != 2:
            return {"success": False, "error": f"Point {i+1} must have 2 coordinates"}
        x, y = point
        if x < 0 or x > request.image_width or y < 0 or y > request.image_height:
            return {"success": False, "error": f"Point {i+1} ({x}, {y}) is outside image bounds"}

    manager.set_calibration(request.points, (request.image_width, request.image_height))
    return {"success": True}


@app.get("/calibration")
async def get_calibration():
    """Get current calibration state"""
    return {
        "is_calibrated": manager.is_calibrated,
        "points": manager.calibration_points
    }


@app.post("/detection/start")
async def start_detection():
    """Start real shot detection (requires calibration)"""
    if not manager.is_calibrated:
        return {"success": False, "error": "Calibration required before starting detection"}

    await manager.start_shot_detection()
    return {"success": True, "status": "started"}


@app.post("/detection/stop")
async def stop_detection():
    """Stop shot detection, revert to dummy broadcast"""
    await manager.stop_shot_detection()
    return {"success": True, "status": "stopped"}


@app.post("/detection/reset")
async def reset_detection():
    """Reset detection statistics"""
    manager.reset_stats()
    return {"success": True, "status": "reset"}


@app.get("/detection/stats")
async def get_stats():
    """Get current detection statistics"""
    return manager.get_current_stats()
```

Update startup to NOT auto-start dummy broadcast (optional - can keep dummy as fallback):
```python
@app.on_event("startup")
async def on_startup() -> None:
    # Start dummy broadcaster (will be replaced when detection starts)
    manager.start_dummy_broadcast(interval_sec=0.5)
    logger.info("FastAPI startup complete")
```

---

## Step 5: Integrate Pipeline with WebRTC Manager

**Modify:** `backend/src/utils/webrtc_manager.py`

Add imports at top:
```python
from score_detection.shot_processing_pipeline import ShotProcessingPipeline
```

Add to `ConnectionManager.__init__`:
```python
def __init__(self):
    # ... existing code ...
    self.shot_pipeline: Optional[ShotProcessingPipeline] = None
    self.calibration_points: Optional[List[List[float]]] = None
    self.is_calibrated: bool = False
    self._frame_dimensions: Tuple[int, int] = (1280, 720)
```

Add these methods to `ConnectionManager`:
```python
def set_calibration(self, points: List[List[float]], dimensions: Tuple[int, int]):
    """Set calibration points for shot localization"""
    self.calibration_points = points
    self._frame_dimensions = dimensions
    self.is_calibrated = True

    # Update pipeline if running
    if self.shot_pipeline:
        self.shot_pipeline.set_calibration(points, dimensions)

    logger.info(f"Calibration set: {len(points)} points, dimensions: {dimensions}")


async def start_shot_detection(self):
    """Start real shot detection (replaces dummy broadcast)"""
    # Stop dummy broadcast first
    await self.stop_dummy_broadcast()

    # Create and start pipeline
    self.shot_pipeline = ShotProcessingPipeline(
        frame_buffer=self.frame_buffer,
        broadcaster=self.broadcast,
        calibration_points=self.calibration_points,
        frame_width=self._frame_dimensions[0],
        frame_height=self._frame_dimensions[1]
    )
    await self.shot_pipeline.start()
    logger.info("Shot detection started")


async def stop_shot_detection(self):
    """Stop shot detection, revert to dummy broadcast"""
    if self.shot_pipeline:
        await self.shot_pipeline.stop()
        self.shot_pipeline = None

    # Restart dummy broadcast
    self.start_dummy_broadcast()
    logger.info("Shot detection stopped, reverted to dummy broadcast")


def reset_stats(self):
    """Reset detection statistics"""
    if self.shot_pipeline:
        self.shot_pipeline.reset_stats()


def get_current_stats(self) -> dict:
    """Get current detection statistics"""
    if self.shot_pipeline:
        return self.shot_pipeline.detector.get_current_stats()
    return {
        'totalShots': {'made': 0, 'total': 0},
        'percentages': {'fieldGoal': 0.0, 'twoPoint': 0.0, 'threePoint': 0.0}
    }
```

---

## Step 6: Testing

### 6.1 Test Bug Fixes
```bash
# Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Check health endpoint
curl http://localhost:8000/health
```

### 6.2 Test Calibration API
```bash
# Set calibration (example points)
curl -X POST http://localhost:8000/calibration \
  -H "Content-Type: application/json" \
  -d '{
    "points": [[100,400], [300,400], [500,400], [700,400], [300,200], [500,200]],
    "image_width": 1280,
    "image_height": 720
  }'

# Get calibration
curl http://localhost:8000/calibration
```

### 6.3 Test Detection Control
```bash
# Start detection (after calibration)
curl -X POST http://localhost:8000/detection/start

# Get stats
curl http://localhost:8000/detection/stats

# Stop detection
curl -X POST http://localhost:8000/detection/stop
```

### 6.4 End-to-End Test
1. Start backend: `uvicorn main:app --reload`
2. Open frontend and establish WebRTC connection
3. Set calibration via API or frontend UI
4. Start detection
5. Point camera at basketball court
6. Verify shot events appear on frontend

---

## Data Flow Summary

```
Frontend Camera
      |
      v (WebRTC Video Track)
TrackConsumer.run()
      |
      v (await put_frame)
SharedFrameBuffer (async queue)
      |
      v (await get_next)
ShotProcessingPipeline._process_loop()
      |
      v (asyncio.to_thread)
StreamingShotDetector.process_frame()
      |
      |-- Ball/Rim detection (every frame)
      |-- Shoot detection (every 3 frames)
      |-- Deduplication (3 levels)
      |-- Court localization (if calibrated)
      |
      v
broadcast() via DataChannel
      |
      v
Frontend receives FrameSyncPayload / ShotPayload
```

---

## Troubleshooting

### YOLO model not loading
- Check `backend/score_detection/config.yaml` has correct paths
- Ensure weights files exist: `weights/04242025_shot_detector.pt`, `weights/03102025_best.pt`

### Frame buffer overflow
- Increase `skip_threshold` in ShotProcessingPipeline
- Check YOLO inference speed - consider using GPU

### No shots detected
- Verify calibration points are correct
- Check that shoot model confidence threshold (0.3) is appropriate
- Enable debug logging in StreamingShotDetector

### WebRTC connection issues
- Check CORS settings in main.py
- Verify frontend is connecting to correct backend URL
