import asyncio
import collections
import logging
import json
import time
import random
from typing import List, Dict, Deque, Any, Optional, Tuple
import sys
import os

# Add score_detection to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'score_detection'))
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from schema import GameStats, TotalShots, Percentages, Shot, Point, FrameSyncPayload, ShotPayload

# Alias for backward compatibility
ShotDetectedPayload = ShotPayload

logger = logging.getLogger(__name__)


class SharedFrameBuffer:
    """
    Thread-safe buffer to share frames between WebRTC (Producer) and AI Model (Consumer).
    """

    def __init__(self, maxsize: int = 600, history_maxlen: int = 120):
        self._q: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=maxsize)
        self._history: Deque[VideoFrame] = collections.deque(maxlen=history_maxlen)

        self._total_frames_received: int = 0
        self._last_frame_ts: Optional[float] = None

    async def put_frame(self, frame: VideoFrame) -> None:
        """
        Enqueue a frame (FIFO). This applies backpressure if queue is full,
        so frames are not dropped by this buffer.
        """
        await self._q.put(frame)
        self._history.append(frame)
        self._total_frames_received += 1
        self._last_frame_ts = time.time()

    async def get_next(self) -> VideoFrame:
        """Dequeue the next frame in FIFO order (consumes it)."""
        frame = await self._q.get()
        self._q.task_done()
        return frame

    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics for debugging"""
        return {
            "queue_size": self._q.qsize(),
            "queue_maxsize": self._q.maxsize,
            "history_len": len(self._history),
            "total_frames_received": self._total_frames_received,
            "last_frame_ts": self._last_frame_ts
        }


class TrackConsumer:
    def __init__(self, track, buffer: SharedFrameBuffer):
        self.track = track
        self.buffer = buffer
        self.task = None

    async def run(self):
        """Continuous loop to drain the WebRTC track."""
        try:
            while True:
                frame = await self.track.recv()
                await self.buffer.put_frame(frame)
        except Exception:
            pass

    def start(self):
        self.task = asyncio.create_task(self.run())

    def stop(self):
        if self.task:
            self.task.cancel()


class ConnectionManager:
    def __init__(self):
        self.pcs = set()
        self.consumers = set()
        self.frame_buffer = SharedFrameBuffer(maxsize=60)  # Store ~2 seconds for analysis
        self.active_data_channel = None

        # Dummy broadcast loop (for frontend UI verification)
        self._dummy_task: Optional[asyncio.Task] = None
        self._dummy_seq: int = 0

        # Shot detection pipeline
        self.shot_pipeline: Optional[Any] = None  # ShotProcessingPipeline
        self.calibration_points: Optional[List[List[float]]] = None
        self.calibration_dimensions: Optional[Tuple[int, int]] = None
        self._is_calibrated: bool = False
        self._calibration_localizer: Optional[Any] = None  # ShotLocalizer for diagnostics

    async def handle_offer(self, params):
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel opened: {channel.label}")
            self.active_data_channel = channel

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                logger.info("Video track received, starting consumer")
                consumer = TrackConsumer(track, self.frame_buffer)
                self.consumers.add(consumer)
                consumer.start()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup(pc)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }

    async def broadcast(self, data: dict):
        """Called by the Model Runner to send stats to frontend."""
        if self.active_data_channel and self.active_data_channel.readyState == "open":
            try:
                self.active_data_channel.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to send data: {e}")

    def start_dummy_broadcast(self, interval_sec: float = 0.5) -> None:
        """Start background dummy broadcaster (idempotent)."""
        if self._dummy_task and not self._dummy_task.done():
            return
        self._dummy_task = asyncio.create_task(self._dummy_broadcast_loop(interval_sec))
        logger.info("Dummy broadcast loop started")

    async def stop_dummy_broadcast(self) -> None:
        if not self._dummy_task:
            return
        self._dummy_task.cancel()
        try:
            await self._dummy_task
        except (asyncio.CancelledError, Exception):
            pass  # Expected when cancelling the task
        self._dummy_task = None

    async def _dummy_broadcast_loop(self, interval_sec: float) -> None:
        shot_id = 0
        made_cnt = 0
        total_cnt = 0
        two_total = 0
        three_total = 0
        two_made = 0
        three_made = 0

        while True:
            try:
                # Wait for next frame
                frame: VideoFrame = await self.frame_buffer.get_next()

                # ========== TODO: HANDLE PROCESSING ==========

                frame.to_ndarray()

                # ========================================

                await asyncio.sleep(interval_sec)
                self._dummy_seq += 1
                now_ms = int(time.time() * 1000)
                video_ts_ms = now_ms

                current_stats = GameStats(
                    totalShots=TotalShots(made=made_cnt, total=total_cnt),
                    percentages=Percentages(
                        fieldGoal=(made_cnt / total_cnt * 100.0) if total_cnt else 0.0,
                        twoPoint=(two_made / two_total * 100.0) if two_total else 0.0,
                        threePoint=(three_made / three_total * 100.0) if three_total else 0.0,
                    ),
                )

                frame_msg = FrameSyncPayload(
                    sequence_id=self._dummy_seq,
                    video_timestamp_ms=video_ts_ms,
                    current_stats=current_stats,
                )

                sent_frame = await self.broadcast(frame_msg.model_dump())

                if self._dummy_seq % 5 == 0:
                    shot_id += 1
                    is_three = (shot_id % 2 == 0)
                    made = (shot_id % 3 != 0)  # 2 made, 1 miss repeating

                    total_cnt += 1
                    if made:
                        made_cnt += 1

                    if is_three:
                        three_total += 1
                        if made:
                            three_made += 1
                        shot_type: Shot.__annotations__["type"] = "3pt"
                    else:
                        two_total += 1
                        if made:
                            two_made += 1
                        shot_type = "2pt"

                    event = Shot(
                        id=shot_id,
                        timestamp_ms=video_ts_ms,
                        type=shot_type,
                        result="made" if made else "missed",
                        location=random.choice(
                            ["Top of Key", "Left Wing", "Right Wing", "In the Paint", "Corner"]
                        ),
                        coord=Point(x=random.uniform(0, 50), y=random.uniform(0, 47)),
                    )

                    updated_stats = GameStats(
                        totalShots=TotalShots(made=made_cnt, total=total_cnt),
                        percentages=Percentages(
                            fieldGoal=(made_cnt / total_cnt * 100.0) if total_cnt else 0.0,
                            twoPoint=(two_made / two_total * 100.0) if two_total else 0.0,
                            threePoint=(three_made / three_total * 100.0) if three_total else 0.0,
                        ),
                    )

                    shot_msg = ShotDetectedPayload(
                        video_timestamp_ms=video_ts_ms,
                        event_data=event,
                        updated_stats=updated_stats,
                    )

                    sent_shot = await self.broadcast(shot_msg.model_dump())
                else:
                    sent_shot = False

                if self._dummy_seq % 10 == 0:
                    logger.info(
                        "dummy(seq=%s) sent_frame=%s sent_shot=%s dc=%s totals=%s/%s",
                        self._dummy_seq,
                        sent_frame,
                        sent_shot,
                        getattr(self.active_data_channel, "readyState", None),
                        made_cnt,
                        total_cnt,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Dummy broadcaster loop crashed; continuing")
                await asyncio.sleep(0.5)

    async def cleanup(self, pc):
        """Gracefully cleanup all resources for a connection and reset state."""
        logger.info("Starting connection cleanup...")
        
        # 1. Stop shot detection pipeline
        if self.shot_pipeline:
            await self.shot_pipeline.stop()
            self.shot_pipeline = None
            logger.info("Shot pipeline stopped")
        
        # 2. Stop dummy broadcast
        await self.stop_dummy_broadcast()
        logger.info("Dummy broadcast stopped")
        
        # 3. Stop all track consumers
        for c in list(self.consumers):
            c.stop()
        self.consumers.clear()
        logger.info("Track consumers stopped")
        
        # 4. Create fresh frame buffer (clears all queued frames)
        self.frame_buffer = SharedFrameBuffer(maxsize=60)
        logger.info("Frame buffer reset")
        
        # 5. Reset calibration state
        self.calibration_points = None
        self.calibration_dimensions = None
        self._is_calibrated = False
        logger.info("Calibration state reset")
        
        # 6. Clear data channel reference
        self.active_data_channel = None
        
        # 7. Reset sequence counter
        self._dummy_seq = 0
        
        # 8. Close peer connection
        self.pcs.discard(pc)
        try:
            await pc.close()
        except Exception as e:
            logger.warning(f"Error closing peer connection: {e}")
        
        logger.info("Connection cleanup complete - all state reset for next session")

    async def reset_session_state(self) -> None:
        """Reset all accumulated data without closing the connection."""
        # Reset pipeline stats
        if self.shot_pipeline:
            self.shot_pipeline.full_reset()
        
        # Reset dummy sequence
        self._dummy_seq = 0
        
        # Clear frame buffer
        self.frame_buffer = SharedFrameBuffer(maxsize=60)
        
        logger.info("Session state reset (connection maintained)")

    # ============ Calibration & Detection Methods ============

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been set"""
        return self._is_calibrated

    @property
    def is_detecting(self) -> bool:
        """Check if real shot detection is running"""
        return self.shot_pipeline is not None and self.shot_pipeline.is_running

    def set_calibration(
        self,
        points: List[List[float]],
        dimensions: Tuple[int, int],
        mode: str = "4-point"
    ) -> bool:
        """
        Set court calibration (4-point or 6-point).

        Args:
            points: List of 4 or 6 calibration points [[x1,y1], [x2,y2], ...]
            dimensions: (width, height) of the calibration frame
            mode: "4-point" (paint box only) or "6-point" (full baseline)

        Returns:
            True if calibration was successful
        """
        # Validate mode
        if mode not in ["4-point", "4-point", "6-point", "6-point-3pt"]:
            logger.error(f"Invalid calibration mode '{mode}'")
            return False

        # Validate point count based on mode
        expected_points = 4 if mode in ("4-point", "4-point") else 6
        if len(points) != expected_points:
            logger.error(f"{mode} calibration requires {expected_points} points, got {len(points)}")
            return False

        for i, point in enumerate(points):
            if len(point) != 2:
                logger.error(f"Point {i} must have 2 coordinates, got {len(point)}")
                return False

        self.calibration_points = points
        self.calibration_dimensions = dimensions
        self.calibration_mode = mode
        self._is_calibrated = True

        # Create a localizer immediately so diagnostics work before detection starts
        try:
            from localization import ShotLocalizer
            self._calibration_localizer = ShotLocalizer(
                calibration_points=points,
                image_dimensions=dimensions,
                calibration_mode=mode
            )
        except Exception as e:
            logger.warning(f"Could not create calibration localizer: {e}")
            self._calibration_localizer = None

        # Update pipeline if running
        if self.shot_pipeline:
            self.shot_pipeline.set_calibration(points, dimensions, mode)

        logger.info(f"Calibration set: {mode} ({len(points)} points), dimensions {dimensions}")
        return True

    def get_calibration_diagnostics(self) -> Dict[str, Any]:
        """
        Return per-point reprojection errors and court outline pixels for the current calibration.
        Prefers the pipeline's localizer if detection is running; falls back to the
        standalone localizer created at calibration time.
        """
        try:
            localizer = None
            if self.shot_pipeline and self.shot_pipeline.detector.shot_localizer:
                localizer = self.shot_pipeline.detector.shot_localizer
            elif self._calibration_localizer:
                localizer = self._calibration_localizer

            if localizer:
                diag = localizer.get_calibration_diagnostics()
                outline = localizer.get_court_outline_pixels()
                return {'diagnostics': diag, 'court_outline_pixels': outline}
        except Exception as e:
            logger.warning(f"Could not get calibration diagnostics: {e}")
        return {'diagnostics': None, 'court_outline_pixels': None}

    def get_calibration(self) -> Dict[str, Any]:
        """Get current calibration state"""
        return {
            "is_calibrated": self._is_calibrated,
            "points": self.calibration_points,
            "dimensions": self.calibration_dimensions
        }

    async def start_shot_detection(self) -> bool:
        """
        Start real shot detection (requires calibration).

        This stops the dummy broadcast and starts the real detection pipeline.

        Returns:
            True if detection started successfully
        """
        if not self._is_calibrated:
            logger.warning("Cannot start detection without calibration")
            return False

        if self.shot_pipeline and self.shot_pipeline.is_running:
            logger.warning("Shot detection already running")
            return True

        # Stop dummy broadcast
        await self.stop_dummy_broadcast()

        # Import and create pipeline
        try:
            from shot_processing_pipeline import ShotProcessingPipeline

            self.shot_pipeline = ShotProcessingPipeline(
                frame_buffer=self.frame_buffer,
                broadcaster=self.broadcast,
                calibration_points=self.calibration_points,
                frame_width=self.calibration_dimensions[0] if self.calibration_dimensions else 1280,
                frame_height=self.calibration_dimensions[1] if self.calibration_dimensions else 720,
                calibration_mode=getattr(self, 'calibration_mode', '4-point')
            )

            await self.shot_pipeline.start()
            logger.info("Shot detection started")
            return True

        except Exception as e:
            logger.exception(f"Failed to start shot detection: {e}")
            # Restart dummy on failure
            self.start_dummy_broadcast()
            return False

    async def stop_shot_detection(self) -> None:
        """
        Stop shot detection and revert to dummy broadcast.
        """
        if self.shot_pipeline:
            await self.shot_pipeline.stop()
            self.shot_pipeline = None
            logger.info("Shot detection stopped")

        # Restart dummy broadcast
        self.start_dummy_broadcast()

    def reset_stats(self) -> None:
        """Reset accumulated statistics"""
        if self.shot_pipeline:
            self.shot_pipeline.reset_stats()
        self._dummy_seq = 0
        logger.info("Statistics reset")

    def get_current_stats(self) -> Optional[Dict[str, Any]]:
        """Get current detection statistics"""
        if self.shot_pipeline:
            stats = self.shot_pipeline.get_current_stats()
            return stats.model_dump()
        return None
