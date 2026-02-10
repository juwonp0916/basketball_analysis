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
import time
from typing import Optional, Callable, List, Tuple, Any

from schema import (
    GameStats, TotalShots, Percentages, Shot, Point,
    FrameSyncPayload, ShotPayload
)

logger = logging.getLogger(__name__)


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
    SKIP_THRESHOLD = 10

    def __init__(
        self,
        frame_buffer: Any,  # SharedFrameBuffer
        broadcaster: Callable,
        calibration_points: Optional[List[List[float]]] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        frame_rate: float = 30.0
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

        # Initialize detector
        self.detector = StreamingShotDetector(
            calibration_points=calibration_points,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_rate=frame_rate
        )

        # Pipeline state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sequence_id = 0

        # Stats tracking
        self._two_pt_made = 0
        self._two_pt_total = 0
        self._three_pt_made = 0
        self._three_pt_total = 0
        self._shot_id_counter = 0

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
        mode: str = "6-point"
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
            logger.info(f"Calibration updated: {mode} ({len(points)} points), {dimensions}")
        return success

    def set_team_colors(self, team0_color: str, team1_color: str) -> bool:
        """
        Set team colors for jersey-based team detection.

        Args:
            team0_color: Color name for team 0 (e.g., 'red', 'blue')
            team1_color: Color name for team 1

        Returns:
            True if colors were set successfully
        """
        success = self.detector.set_team_colors(team0_color, team1_color)
        if success:
            logger.info(f"Team colors set: team0={team0_color}, team1={team1_color}")
        return success

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
                    await self.broadcaster(shot_msg.model_dump())

                    logger.info(
                        f"Shot detected: id={shot_event.shot_id}, "
                        f"made={shot_event.is_made}, type={shot_event.shot_type}"
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
        court_x = 0.0
        court_y = 0.0
        if shot_event.court_position and shot_event.court_position[0] is not None:
            court_x = shot_event.court_position[0]
            court_y = shot_event.court_position[1]
        elif shot_event.shooter_position:
            # Normalize pixel position to court coordinates
            court_x = shot_event.shooter_position['x'] / self.frame_width * 15.0
            court_y = shot_event.shooter_position['y'] / self.frame_height * 14.0

        # Convert court meters → frontend chart coordinates
        # X: [0, 15]m → [0, 50] (left to right)
        coord_x = court_x / 15.0 * 50.0
        # Y: [0, 14]m → [47, 0] (baseline at top, half-court at bottom)
        # Flip Y-axis: baseline (Y=0) should map to top of chart
        coord_y = (14.0 - court_y) / 14.0 * 47.0

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

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self._running

    @property
    def is_calibrated(self) -> bool:
        """Check if detector is calibrated"""
        return self.detector.shot_localizer is not None
