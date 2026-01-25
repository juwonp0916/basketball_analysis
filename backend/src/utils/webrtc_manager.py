import asyncio
import collections
import logging
import json
import time
import random
from typing import List, Dict, Deque, Any, Optional
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from schema import GameStats, TotalShots, Percentages, Shot, Point, FrameSyncPayload, ShotDetectedPayload

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
                self.buffer.add_frame(frame)
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
        self.frame_buffer = SharedFrameBuffer(maxlen=60)  # Store ~2 seconds for analysis
        self.active_data_channel = None

        # Dummy broadcast loop (for frontend UI verification)
        self._dummy_task: Optional[asyncio.Task] = None
        self._dummy_seq: int = 0

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
        except Exception:
            pass
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
        self.pcs.discard(pc)
        for c in list(self.consumers):
            c.stop()
        await pc.close()
