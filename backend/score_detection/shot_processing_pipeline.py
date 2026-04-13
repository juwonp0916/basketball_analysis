"""
ShotProcessingPipeline - Orchestrates StreamingShotDetector with WebRTC frame buffer.

Connects the streaming shot detector to the WebRTC infrastructure, handling:
- Async frame consumption from SharedFrameBuffer
- Running YOLO inference in thread pool to avoid blocking
- Broadcasting detection results via data channel
- Frame skipping when falling behind real-time
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Any

from schema import (
    GameStats, TotalShots, Percentages, Shot, Point,
    FrameSyncPayload, ShotPayload
)

logger = logging.getLogger(__name__)

# Debug toggle: set BASKETBALL_DEBUG=1 to enable debug frame saving.
# Default is ON for development; set to 0 for demos/production.
BASKETBALL_DEBUG = os.environ.get("BASKETBALL_DEBUG", "1") == "1"


class ShotProcessingPipeline:
    """
    Async processing pipeline that connects WebRTC frame buffer to shot detector.

    This pipeline:
    - Consumes frames from SharedFrameBuffer asynchronously
    - Runs YOLO detection in a thread pool to avoid blocking the event loop
    - Broadcasts FrameSyncPayload for each processed frame
    - Broadcasts ShotPayload when shots are detected
    - Implements frame skipping when processing falls behind real-time
    """

    # Skip threshold: if more than this many frames queued, start skipping
    # Set to 10 for aggressive frame dropping to keep real-time performance
    SKIP_THRESHOLD = 100000

    def __init__(
        self,
        frame_buffer: Any,  # SharedFrameBuffer
        broadcaster: Callable,
        calibration_points: Optional[List[List[float]]] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        frame_rate: float = 30.0,
        calibration_mode: str = "4-point"
    ):
        """
        Initialize the processing pipeline.

        Args:
            frame_buffer: SharedFrameBuffer instance for consuming frames
            broadcaster: Async callable to broadcast messages (e.g., ConnectionManager.broadcast)
            calibration_points: Optional 6-point calibration for court localization
            frame_width: Expected frame width
            frame_height: Expected frame height
            frame_rate: Expected frame rate
        """
        # Lazy import to avoid circular dependencies
        from streaming_shot_detector import StreamingShotDetector

        self.frame_buffer = frame_buffer
        self.broadcaster = broadcaster
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_rate = frame_rate
        self.calibration_mode = calibration_mode

        # Raw calibration points (in the coordinate space of the calibration frame).
        # On the first incoming frame we detect whether WebRTC delivers a different
        # resolution and rescale these points to the actual frame — no frame upscaling.
        self._calibration_points_raw = [list(p) for p in calibration_points] if calibration_points else None
        self._calibration_rescaled = False

        # Initialize detector
        self.detector = StreamingShotDetector(
            calibration_points=calibration_points,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_rate=frame_rate,
            calibration_mode=calibration_mode
        )

        # Pipeline state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sequence_id = 0

        # Video Recorder module (standalone)
        import score_detection.config as cfg
        import yaml

        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            _env = yaml.safe_load(f)
        weights_path = os.path.join(os.path.dirname(__file__), _env["weights_path"])

        # Debug: save frames on shot detection
        self._debug_frame_dir = Path(__file__).parent.parent.parent / "debug_shot_frames"
        self._debug_frame_dir.mkdir(parents=True, exist_ok=True)
        self._debug_frame_count = 0
        self._first_frame_logged = False

        # Stats tracking
        self._two_pt_made = 0
        self._two_pt_total = 0
        self._three_pt_made = 0
        self._three_pt_total = 0
        self._shot_id_counter = 0

        # Auto-calibration state
        self._team_calibrated = False
        self._calibration_frames = 0
        self._calibration_max_frames = 60   # ~2s at 30fps — accumulation needs more frames
        self._calibration_retry_interval = 30  # After initial window, retry every N frames
        self._recheck_interval = 900  # ~30s at 30fps — silent periodic re-check
        self._last_recheck_seq = 0

        logger.info("ShotProcessingPipeline initialized")

    async def start(self) -> None:
        """Start the async processing loop"""
        if self._running:
            logger.warning("Pipeline already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Shot processing pipeline started")

    async def stop(self) -> None:
        """Stop the processing loop gracefully"""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Shot processing pipeline stopped")

    def set_calibration(
        self,
        points: List[List[float]],
        dimensions: Tuple[int, int],
        mode: str = "4-point"
    ) -> bool:
        """
        Update calibration (for live calibration).

        Args:
            points: 4 or 6 calibration points
            dimensions: (width, height) of calibration frame
            mode: "4-point" or "6-point"

        Returns:
            True if calibration was successful
        """
        success = self.detector.set_calibration(points, dimensions, mode)
        if success:
            self.frame_width, self.frame_height = dimensions
            self._calibration_points_raw = [list(p) for p in points]
            self._calibration_rescaled = False  # Allow rescaling on next frame
            logger.info(f"Calibration updated: {mode} ({len(points)} points), {dimensions}")
        return success

    def get_team_colors_hex(self) -> Tuple[str, str]:
        """Get the hex colors of the auto-calibrated teams"""
        return self.detector.get_team_colors_hex()

    def reset_stats(self) -> None:
        """Reset accumulated statistics"""
        self.detector.reset()
        self._two_pt_made = 0
        self._two_pt_total = 0
        self._three_pt_made = 0
        self._three_pt_total = 0
        self._shot_id_counter = 0
        self._sequence_id = 0
        logger.info("Statistics reset")

    def full_reset(self) -> None:
        """
        Complete reset of all pipeline state for a fresh session.
        Call this when stopping analysis to ensure clean state for next session.
        """
        # Reset stats
        self.reset_stats()

        # Reset team calibration state
        self._team_calibrated = False
        self._calibration_frames = 0
        self._last_recheck_seq = 0

        # Reset team detector if available
        if hasattr(self, 'detector') and hasattr(self.detector, 'team_detector'):
            self.detector.team_detector.reset()

        logger.info("Full pipeline reset complete")

    def get_current_stats(self) -> GameStats:
        """Return current accumulated statistics as GameStats"""
        total = self._two_pt_total + self._three_pt_total
        made = self._two_pt_made + self._three_pt_made

        return GameStats(
            totalShots=TotalShots(made=made, total=total),
            percentages=Percentages(
                fieldGoal=(made / total * 100.0) if total > 0 else 0.0,
                twoPoint=(self._two_pt_made / self._two_pt_total * 100.0) if self._two_pt_total > 0 else 0.0,
                threePoint=(self._three_pt_made / self._three_pt_total * 100.0) if self._three_pt_total > 0 else 0.0
            )
        )

    async def _process_loop(self) -> None:
        """Main processing loop - consumes frames and broadcasts results"""
        logger.info("Processing loop started")
        frames_processed = 0
        frames_skipped = 0

        while self._running:
            try:
                # Get next frame from buffer
                frame = await self.frame_buffer.get_next()

                # Check queue size for frame skipping (check every frame)
                queue_size = self.frame_buffer._q.qsize()
                if queue_size > self.SKIP_THRESHOLD:
                    frames_skipped += 1
                    self._sequence_id += 1
                    if frames_skipped % 100 == 0:  # Log less frequently
                        logger.warning(
                            f"Skipped {frames_skipped} frames (queue: {queue_size})"
                        )
                    continue

                # Convert frame to numpy array
                img = frame.to_ndarray(format="bgr24")
                timestamp_ms = int(time.time() * 1000)

                # Check frame dimensions against calibration dimensions.
                # If WebRTC delivers a different resolution (e.g. 640×360 vs 1920×1080),
                # rescale the calibration points to the actual frame size on the first
                # frame — cheaper and higher-quality than upscaling every raw frame.
                actual_h, actual_w = img.shape[:2]
                if not self._first_frame_logged:
                    self._first_frame_logged = True
                    logger.info(
                        f"[DIM CHECK] First frame: actual={actual_w}x{actual_h}, "
                        f"calibration={self.frame_width}x{self.frame_height}, "
                        f"calibration_mode={self.calibration_mode}"
                    )

                if (actual_w != self.frame_width or actual_h != self.frame_height) and not self._calibration_rescaled:
                    if self._calibration_points_raw:
                        scale_x = actual_w / self.frame_width
                        scale_y = actual_h / self.frame_height
                        scaled_pts = [[p[0] * scale_x, p[1] * scale_y] for p in self._calibration_points_raw]
                        self.detector.set_calibration(scaled_pts, (actual_w, actual_h), self.calibration_mode)
                        logger.info(
                            f"[DIM RESCALE] Calibration points scaled from "
                            f"{self.frame_width}x{self.frame_height} → {actual_w}x{actual_h} "
                            f"(scale_x={scale_x:.3f}, scale_y={scale_y:.3f})"
                        )
                    else:
                        # No calibration points — update detector dimensions only
                        self.detector.width = actual_w
                        self.detector.height = actual_h
                        logger.warning(f"[DIM RESCALE] No calibration points to rescale; updating dims only.")
                    self.frame_width = actual_w
                    self.frame_height = actual_h
                    self._calibration_rescaled = True

                # --- Team auto-calibration ---
                if not self._team_calibrated:
                    # Phase 1: initial window — try every frame
                    # Phase 2: after window — slow retry every N frames
                    should_try = (
                        self._calibration_frames < self._calibration_max_frames
                        or (self._calibration_frames - self._calibration_max_frames)
                        % self._calibration_retry_interval == 0
                    )
                    if should_try:
                        success = await asyncio.to_thread(
                            self.detector.auto_calibrate_teams, img
                        )
                        if success:
                            self._team_calibrated = True
                            self._last_recheck_seq = self._sequence_id
                            color0, color1 = self.detector.get_team_colors_hex()
                            logger.info(
                                f"Auto-calibrated team colors: {color0}, {color1} "
                                f"(after {self._calibration_frames + 1} frames)"
                            )

                            from schema import TeamColorsPayload
                            team_colors_msg = TeamColorsPayload(
                                team0_color=color0, team1_color=color1
                            )
                            await self.broadcaster(team_colors_msg.model_dump())
                    self._calibration_frames += 1

                elif (self._sequence_id - self._last_recheck_seq) >= self._recheck_interval:
                    # Silent periodic re-check (no frontend broadcast)
                    self._last_recheck_seq = self._sequence_id
                    await asyncio.to_thread(
                        self.detector.recheck_teams, img
                    )

                # Run YOLO detection in thread pool to avoid blocking
                shot_event = await asyncio.to_thread(
                    self.detector.process_frame,
                    img,
                    timestamp_ms,
                    self._sequence_id
                )

                # Broadcast frame sync only every 5 frames to reduce overhead
                if self._sequence_id % 5 == 0:
                    current_stats = self.get_current_stats()
                    frame_msg = FrameSyncPayload(
                        sequence_id=self._sequence_id,
                        video_timestamp_ms=timestamp_ms,
                        current_stats=current_stats
                    )
                    await self.broadcaster(frame_msg.model_dump())

                # Broadcast shot event if detected
                if shot_event:
                    self._handle_shot_event(shot_event)

                    updated_stats = self.get_current_stats()
                    shot_msg = self._create_shot_payload(shot_event, updated_stats)

                    # Save debug frame BEFORE broadcasting so the image is on disk
                    # by the time the frontend receives the event and fetches it.
                    # Use the frame stored at shoot-detection time if available —
                    # that frame still has ball + shooter visible together.
                    # Fall back to current frame if not stored (fallback path shots).
                    _det_frame = getattr(shot_event, '_detection_frame', None)
                    debug_img = img if _det_frame is None else _det_frame
                    self._save_debug_frame(debug_img, shot_event)

                    await self.broadcaster(shot_msg.model_dump())

                    logger.info(
                        f"Shot detected: id={shot_event.shot_id}, "
                        f"made={shot_event.is_made}, type={shot_event.shot_type}, "
                        f"shooter_px={shot_event.shooter_position}, "
                        f"court_m={shot_event.court_position}"
                    )

                self._sequence_id += 1
                frames_processed += 1

                # Periodic logging
                if frames_processed % 100 == 0:
                    logger.info(
                        f"Processed {frames_processed} frames, "
                        f"skipped {frames_skipped}, "
                        f"shots: {self._two_pt_total + self._three_pt_total}"
                    )

            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                raise
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on persistent errors

        logger.info(
            f"Processing loop ended. Processed: {frames_processed}, Skipped: {frames_skipped}"
        )

    def _handle_shot_event(self, shot_event: Any) -> None:
        """Update stats based on shot event"""
        self._shot_id_counter += 1

        shot_type = shot_event.shot_type or '2pt'

        if shot_type == '3pt':
            self._three_pt_total += 1
            if shot_event.is_made:
                self._three_pt_made += 1
        else:
            self._two_pt_total += 1
            if shot_event.is_made:
                self._two_pt_made += 1

    def _create_shot_payload(self, shot_event: Any, stats: GameStats) -> ShotPayload:
        """Create ShotPayload from shot event"""
        # Determine location string
        location = self._get_location_string(shot_event)

        # Get coordinates in court space first, then convert to frontend chart space.
        # Court system: X ∈ [0, 15], Y ∈ [0, 14] (origin at left baseline corner)
        # Frontend chart: X ∈ [0, 50], Y ∈ [0, 47] (origin at baseline-left corner)
        # Default to basket position so shots with unknown location still appear on the chart
        court_x = 7.5
        court_y = 1.575
        _COURT_W, _COURT_H = 15.0, 14.0
        if shot_event.court_position and shot_event.court_position[0] is not None:
            cx, cy = shot_event.court_position[0], shot_event.court_position[1]
            if 0.0 <= cx <= _COURT_W and 0.0 <= cy <= _COURT_H:
                court_x, court_y = cx, cy
            else:
                # Out-of-bounds homography result — do not plot at wrong position.
                # Clamp to court boundary so the point appears on the nearest valid
                # edge rather than off the chart, and log for investigation.
                court_x = max(0.0, min(_COURT_W, cx))
                court_y = max(0.0, min(_COURT_H, cy))
                logger.warning(
                    f"[PAYLOAD] Out-of-bounds court coords ({cx:.2f}, {cy:.2f}) — "
                    f"clamped to ({court_x:.2f}, {court_y:.2f}). "
                    f"Shooter px={shot_event.shooter_position}"
                )
        elif shot_event.shooter_position:
            # Normalize pixel position to court coordinates
            court_x = shot_event.shooter_position['x'] / self.frame_width * 15.0
            court_y = shot_event.shooter_position['y'] / self.frame_height * 14.0

        # Convert court meters → frontend chart coordinates.
        # Matches CourtRenderer.meters_to_svg used by the static tool.
        # X: [0, 15]m → [0, 50]
        coord_x = (court_x / 15.0 * 50.0)
        # Y: [0, 14]m → [0, 47]  (NO inversion here)
        # The Svelte chart uses yDomain={[47, 0]} which already inverts the scale
        # (same effect as matplotlib's ylim(47, 0) in the static tool),
        # putting baseline (Y=0) at the top and half-court (Y=47) at the bottom.
        coord_y = (court_y / 14.0 * 47.0)

        shot_type_literal = "3pt" if shot_event.shot_type == '3pt' else "2pt"
        result_literal = "made" if shot_event.is_made else "missed"

        shot = Shot(
            id=shot_event.shot_id,
            timestamp_ms=shot_event.timestamp_ms,
            type=shot_type_literal,
            result=result_literal,
            location=location,
            coord=Point(x=coord_x, y=coord_y),
            team_id=shot_event.team_id,
            team_confidence=shot_event.team_confidence
        )

        return ShotPayload(
            video_timestamp_ms=shot_event.timestamp_ms,
            event_data=shot,
            updated_stats=stats
        )

    def _get_location_string(self, shot_event: Any) -> str:
        """Get human-readable location string from zone"""
        zone = shot_event.zone

        zone_names = {
            1: "Left Baseline",
            2: "In the Paint",
            3: "Right Baseline",
            4: "Free Throw",
            5: "Left Wing 3PT",
            6: "Top of Key 3PT",
            7: "Right Wing 3PT",
            8: "Left Corner 3PT",
            9: "Right Corner 3PT"
        }

        return zone_names.get(zone, "Unknown")

    def _save_debug_frame(self, img: Any, shot_event: Any) -> None:
        """
        Save the frame at shot detection to disk for offline debugging.

        Draws all detected person bounding boxes colored by team classification,
        highlights the identified shooter, and adds a legend mapping cluster
        labels to team names/colors.  A sidecar JSON records full metadata
        including per-player LAB values, hex colors, and a ground_truth field
        for manual annotation.

        Controlled by the BASKETBALL_DEBUG environment variable (default: on).
        """
        if not BASKETBALL_DEBUG:
            return

        try:
            import cv2
            import json
            import numpy as np

            self._debug_frame_count += 1
            stem = f"debug_shot_{shot_event.shot_id:04d}"
            img_path = self._debug_frame_dir / f"{stem}.jpg"
            meta_path = self._debug_frame_dir / f"{stem}.json"
            crops_dir = self._debug_frame_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            vis = img.copy()
            sp = shot_event.shooter_position

            # ----------------------------------------------------------
            # 1. Detect all persons and classify each by team
            # ----------------------------------------------------------
            team_det = self.detector.team_detector
            model = self.detector.model
            class_names = self.detector.class_names
            inf_w = self.detector.inference_width
            inf_h = self.detector.inference_height
            frame_w = self.detector.width
            frame_h = self.detector.height
            device = self.detector.device

            # BGR colors for team 0, team 1, and unclassified
            TEAM_COLORS_BGR = {
                0: (0, 200, 0),    # Green for team 0
                1: (200, 100, 0),  # Blue-ish for team 1
                None: (128, 128, 128),  # Gray for unclassified
            }
            SHOOTER_COLOR_BGR = (0, 0, 255)  # Red for shooter

            person_entries: list = []  # Collected for JSON metadata

            try:
                det_frame = cv2.resize(img, (inf_w, inf_h))
                results = model(
                    det_frame,
                    stream=True,
                    verbose=False,
                    imgsz=inf_w,
                    device=device,
                    conf=0.3,
                    max_det=15,
                )

                person_idx = 0
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls >= len(class_names) or class_names[cls] != "person":
                            continue
                        if float(box.conf[0]) < 0.3:
                            continue

                        x1 = int(box.xyxy[0][0] * frame_w / inf_w)
                        y1 = int(box.xyxy[0][1] * frame_h / inf_h)
                        x2 = int(box.xyxy[0][2] * frame_w / inf_w)
                        y2 = int(box.xyxy[0][3] * frame_h / inf_h)
                        bbox = (x1, y1, x2, y2)

                        track_id = int(box.id[0]) if box.id is not None else None

                        # Classify team
                        tid, conf = None, 0.0
                        extracted_lab = None
                        extracted_hex = None
                        if team_det.is_configured:
                            tid, conf = team_det.classify_from_bbox(
                                img, bbox, track_id=track_id
                            )
                            # Extract LAB feature for this player (for evaluation)
                            jersey_region = team_det._get_jersey_region(bbox, img)
                            if jersey_region is not None:
                                feat = team_det._extract_color_features(jersey_region)
                                if feat is not None:
                                    extracted_lab = feat.tolist()
                                    extracted_hex = team_det._lab_to_hex(feat)
                                # Save jersey crop for annotation/evaluation
                                crop_path = crops_dir / f"{stem}_player_{person_idx}.jpg"
                                cv2.imwrite(str(crop_path), jersey_region)

                        # Check if this person is the shooter
                        is_shooter = False
                        if sp:
                            center_x = (x1 + x2) / 2
                            bottom_y = y2
                            dist = np.sqrt(
                                (center_x - sp["x"]) ** 2
                                + (bottom_y - sp["y"]) ** 2
                            )
                            is_shooter = dist < 60  # px threshold

                        color = SHOOTER_COLOR_BGR if is_shooter else TEAM_COLORS_BGR.get(tid, TEAM_COLORS_BGR[None])
                        thickness = 3 if is_shooter else 2
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

                        # Label above bbox
                        team_label = f"Team {tid}" if tid is not None else "?"
                        label = f"{team_label} ({conf:.0%})"
                        if is_shooter:
                            label = f"SHOOTER | {label}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(
                            vis, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        )

                        person_entries.append({
                            "bbox": [x1, y1, x2, y2],
                            "team_id": tid,
                            "confidence": round(conf, 3),
                            "is_shooter": is_shooter,
                            "track_id": track_id,
                            "extracted_lab": extracted_lab,
                            "extracted_hex": extracted_hex,
                            "crop_path": f"crops/{stem}_player_{person_idx}.jpg",
                            "ground_truth_team": None,  # To be filled by annotator
                        })
                        person_idx += 1

            except Exception as det_e:
                logger.warning(f"[DEBUG FRAME] Person detection overlay failed: {det_e}")

            # ----------------------------------------------------------
            # 2. Draw shooter foot marker
            # ----------------------------------------------------------
            if sp:
                px, py = int(sp["x"]), int(sp["y"])
                cv2.circle(vis, (px, py), 12, SHOOTER_COLOR_BGR, -1)
                cv2.circle(vis, (px, py), 15, (255, 255, 255), 2)
                cv2.putText(
                    vis, f"FOOT ({px},{py})", (px + 18, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, SHOOTER_COLOR_BGR, 2,
                )

            # ----------------------------------------------------------
            # 3. Draw ball position
            # ----------------------------------------------------------
            try:
                ball_history = self.detector.state.ball_pos
                if ball_history:
                    ball_center = ball_history[-1][0]
                    bx, by = int(ball_center[0]), int(ball_center[1])
                    cv2.circle(vis, (bx, by), 14, (0, 165, 255), 2)
                    cv2.circle(vis, (bx, by), 4, (0, 165, 255), -1)
                    cv2.putText(
                        vis, f"BALL ({bx},{by})", (bx + 16, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
                    )
            except Exception:
                pass

            # ----------------------------------------------------------
            # 4. Draw court outline overlay
            # ----------------------------------------------------------
            localizer = self.detector.shot_localizer
            if localizer is not None:
                try:
                    polylines = localizer.get_court_outline_pixels()
                    for polyline in polylines:
                        if len(polyline) < 2:
                            continue
                        pts = np.array(polyline, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 0), thickness=2)
                    H_inv = np.linalg.inv(localizer.homography_matrix)
                    basket_pt = np.array([[[7.5, 1.575]]], dtype=np.float64)
                    basket_px = cv2.perspectiveTransform(basket_pt, H_inv)[0][0]
                    bx, by = int(basket_px[0]), int(basket_px[1])
                    cv2.circle(vis, (bx, by), 14, (0, 255, 255), 2)
                    cv2.putText(
                        vis, "BASKET", (bx + 16, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )
                except Exception as court_e:
                    logger.warning(f"[DEBUG FRAME] Could not draw court overlay: {court_e}")

            # ----------------------------------------------------------
            # 5. Draw team legend (top-right corner)
            # ----------------------------------------------------------
            hex0, hex1 = self.detector.get_team_colors_hex()
            legend_lines = [
                f"Shot #{shot_event.shot_id}  {'MADE' if shot_event.is_made else 'MISS'}  {shot_event.shot_type or '?'}",
                f"Assigned team_id: {shot_event.team_id}  conf: {shot_event.team_confidence:.0%}",
                "",
                "--- Team Legend ---",
                f"Team 0 (cluster 0): jersey {hex0}",
                f"Team 1 (cluster 1): jersey {hex1}",
            ]
            lx, ly = vis.shape[1] - 420, 20
            line_h = 24
            # Semi-transparent background
            overlay = vis.copy()
            cv2.rectangle(
                overlay,
                (lx - 10, ly - 10),
                (vis.shape[1] - 10, ly + line_h * len(legend_lines) + 10),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

            for i, line in enumerate(legend_lines):
                y = ly + i * line_h + 16
                cv2.putText(
                    vis, line, (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
            # Color swatches next to team labels
            swatch_y0 = ly + 4 * line_h + 6
            cv2.rectangle(vis, (lx - 6, swatch_y0), (lx - 1, swatch_y0 + 14), TEAM_COLORS_BGR[0], -1)
            cv2.rectangle(vis, (lx - 6, swatch_y0 + line_h), (lx - 1, swatch_y0 + line_h + 14), TEAM_COLORS_BGR[1], -1)

            cv2.imwrite(str(img_path), vis)

            # ----------------------------------------------------------
            # 6. Write JSON sidecar (extended for evaluation framework)
            # ----------------------------------------------------------
            # Compute inter-team Delta-E for quality assessment
            inter_team_delta_e = None
            team0_lab = None
            team1_lab = None
            if team_det.is_configured:
                team0_lab = team_det.team0_color.tolist()
                team1_lab = team_det.team1_color.tolist()
                inter_team_delta_e = round(
                    team_det._color_distance(team_det.team0_color, team_det.team1_color), 2
                )

            meta = {
                "shot_id": shot_event.shot_id,
                "sequence_id": self._sequence_id,
                "is_made": shot_event.is_made,
                "shooter_position_px": sp,
                "court_position_m": list(shot_event.court_position) if shot_event.court_position else None,
                "zone": shot_event.zone,
                "shot_type": shot_event.shot_type,
                "team_id": shot_event.team_id,
                "team_confidence": round(shot_event.team_confidence, 3),
                "team_colors": {
                    "team_0": {"lab": team0_lab, "hex": hex0},
                    "team_1": {"lab": team1_lab, "hex": hex1},
                },
                "inter_team_delta_e": inter_team_delta_e,
                "persons_detected": person_entries,
                "calibration_mode": self.calibration_mode,
                "frame_dims_pipeline": [self.frame_width, self.frame_height],
                "frame_dims_actual": [img.shape[1], img.shape[0]],
                "note": (
                    "Use this frame with: "
                    f"python static_shot_localization.py --image {img_path}"
                ),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"[DEBUG FRAME] Saved shot frame → {img_path}")

        except Exception as e:
            logger.warning(f"[DEBUG FRAME] Could not save frame: {e}")

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self._running

    @property
    def is_calibrated(self) -> bool:
        """Check if detector is calibrated"""
        return self.detector.shot_localizer is not None
