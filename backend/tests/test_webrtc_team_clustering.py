"""
WebRTC Team Clustering Test — records WebRTC frames during a live analysis session
and runs the exact same team clustering algorithm that test_team_clustering.py uses
on local videos.

Usage:
  1. Start the backend with WEBRTC_TEAM_TEST=1 :
       WEBRTC_TEAM_TEST=1 uvicorn main:app --reload --host 0.0.0.0 --port 8000
  2. On the frontend, start analysis as normal (calibrate → start).
  3. When you click "Stop Analysis", the recorder finalizes and saves two files:
       backend/output_videos/webrtc_raw.mp4       — raw WebRTC frames (no overlay)
       backend/output_videos/webrtc_teams.mp4      — team-labeled overlay (like test_team_clustering.py)
  4. Compare webrtc_teams.mp4 with the output of test_team_clustering.py on the
     same video to see if WebRTC frame quality causes color differences.

Architecture:
  - WebRTCFrameRecorder is an independent class that receives BGR frames via
    record_frame().  It does NOT consume from the SharedFrameBuffer — the pipeline
    feeds it copies after to_ndarray(), so it has zero impact on the detection path.
  - The recorder runs team clustering in its own thread so it never blocks the
    async detection loop.
  - Enabled only when WEBRTC_TEAM_TEST=1 env var is set.

To detach completely: simply unset the env var (or remove the two integration
points marked with "# WEBRTC_TEAM_TEST" in shot_processing_pipeline.py).
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Ensure score_detection is importable
_SCORE_DIR = Path(__file__).parent.parent / "score_detection"
sys.path.insert(0, str(_SCORE_DIR))


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


class WebRTCFrameRecorder:
    """
    Records WebRTC frames and runs team clustering offline.

    Thread-safe: record_frame() can be called from the async pipeline;
    the heavy work (YOLO + clustering + video write) runs in a background thread.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self._output_dir = output_dir or (Path(__file__).parent.parent / "output_videos")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._frames: list = []  # list of (BGR ndarray, wall_clock_time)
        self._lock = threading.Lock()
        self._finalized = False
        self._last_logged_dims: Optional[Tuple[int, int]] = None  # track resolution changes

        # Load YOLO config (same as test_team_clustering.py)
        config_path = _SCORE_DIR / "config.yaml"
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        logger.info(
            f"[WebRTCFrameRecorder] Initialized — frames will be saved to {self._output_dir}"
        )

    def record_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Enqueue a BGR frame for recording.  This is lightweight — just appends
        to a list.  Call from the pipeline's processing loop.
        """
        if self._finalized:
            return
        with self._lock:
            # Log resolution changes as they happen
            h, w = bgr_frame.shape[:2]
            dims = (w, h)
            if dims != self._last_logged_dims:
                logger.info(
                    f"[WebRTCFrameRecorder] Frame #{len(self._frames)}: "
                    f"resolution changed to {w}x{h}"
                    + (f" (was {self._last_logged_dims[0]}x{self._last_logged_dims[1]})"
                       if self._last_logged_dims else " (first frame)")
                )
                self._last_logged_dims = dims
            # Store a copy so the caller can mutate/release the original
            self._frames.append((bgr_frame.copy(), time.time()))

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def finalize(self) -> None:
        """
        Process all recorded frames and write output videos.
        This runs synchronously (call from the stop-detection path)
        or in a background thread.
        """
        if self._finalized:
            return
        self._finalized = True

        with self._lock:
            frames = list(self._frames)
            self._frames.clear()

        if not frames:
            logger.warning("[WebRTCFrameRecorder] No frames recorded, nothing to finalize")
            return

        logger.info(
            f"[WebRTCFrameRecorder] Finalizing {len(frames)} frames..."
        )

        # Run in a background thread so we don't block the event loop
        t = threading.Thread(target=self._process_frames, args=(frames,), daemon=True)
        t.start()
        # Don't join — let it finish in the background

    def _process_frames(self, frames: list) -> None:
        """Heavy lifting: write raw video + run team clustering + write labeled video."""
        try:
            from team_detector import StreamingTeamDetector
            from ultralytics import YOLO

            # Unpack (frame, timestamp) tuples
            raw_frames = [f for f, _t in frames]
            timestamps = [t for _f, t in frames]

            # ------- 0. Determine canonical resolution -------
            # WebRTC frames can change resolution mid-stream (browser adaptive
            # bitrate / renegotiation).  Pick the most common (width, height)
            # as the canonical size and resize outliers to match.
            dim_counts: dict = {}
            for f in raw_frames:
                h, w = f.shape[:2]
                key = (w, h)
                dim_counts[key] = dim_counts.get(key, 0) + 1

            canonical_wh = max(dim_counts, key=dim_counts.get)
            width, height = canonical_wh

            if len(dim_counts) > 1:
                logger.warning(
                    f"[WebRTCFrameRecorder] Detected {len(dim_counts)} distinct frame sizes "
                    f"across {len(raw_frames)} frames: {dim_counts}.  "
                    f"Using canonical {width}x{height} — resizing outliers."
                )
            else:
                logger.info(
                    f"[WebRTCFrameRecorder] All {len(raw_frames)} frames are {width}x{height}"
                )

            # Compute effective fps from wall-clock timestamps.
            # With frame skipping the pipeline may process fewer frames than
            # the source 30fps, so writing at 30fps would make playback too fast.
            if len(timestamps) >= 2:
                duration = timestamps[-1] - timestamps[0]
                fps = (len(timestamps) - 1) / duration if duration > 0 else 30.0
                # Clamp to reasonable range
                fps = max(1.0, min(fps, 60.0))
            else:
                fps = 30.0
            logger.info(f"[WebRTCFrameRecorder] Effective recording fps: {fps:.1f}")

            def ensure_size(img: np.ndarray) -> np.ndarray:
                """Resize to canonical dims if needed (no-op for matching frames)."""
                h, w = img.shape[:2]
                if w != width or h != height:
                    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                return img

            # ------- 1. Write raw (unmodified) WebRTC video -------
            raw_path = self._output_dir / "webrtc_raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            raw_writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
            for f in raw_frames:
                raw_writer.write(ensure_size(f))
            raw_writer.release()
            logger.info(f"[WebRTCFrameRecorder] Raw video saved: {raw_path}")

            # ------- 2. Team clustering (same as test_team_clustering.py) -------
            weights_path = _SCORE_DIR / self._config["weights_path"]
            model = YOLO(str(weights_path), verbose=False)
            class_names = self._config["classes"]
            device = self._config.get("device", "cpu")

            inf_w = max(640, int(width * 0.5) // 32 * 32)
            inf_h = max(384, int(height * 0.5) // 32 * 32)
            inference_dims = (inf_w, inf_h)

            team_detector = StreamingTeamDetector()

            teams_path = self._output_dir / "webrtc_teams.mp4"
            teams_writer = cv2.VideoWriter(str(teams_path), fourcc, fps, (width, height))

            for frame_idx, frame in enumerate(raw_frames):
                frame = ensure_size(frame)
                vis = frame.copy()

                if frame_idx % 30 == 0:
                    logger.info(f"[WebRTCFrameRecorder] Processing frame {frame_idx}/{len(raw_frames)}...")

                if not team_detector.is_configured:
                    team_detector.auto_calibrate(frame, model, class_names, inference_dims, device)
                    cv2.putText(vis, "Calibrating Teams...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                else:
                    if frame_idx % 60 == 0:
                        team_detector.recheck(frame, model, class_names, inference_dims, device)

                    hex0, hex1 = team_detector.get_team_colors_hex()
                    color0, color1 = hex_to_bgr(hex0), hex_to_bgr(hex1)

                    det_frame = cv2.resize(frame, inference_dims)
                    results = model(
                        det_frame, stream=True, verbose=False,
                        imgsz=inf_w, device=device, conf=0.3, max_det=30
                    )

                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            if cls >= len(class_names) or class_names[cls] != 'person':
                                continue
                            if float(box.conf[0]) < 0.3:
                                continue

                            x1 = int(box.xyxy[0][0] * width / inf_w)
                            y1 = int(box.xyxy[0][1] * height / inf_h)
                            x2 = int(box.xyxy[0][2] * width / inf_w)
                            y2 = int(box.xyxy[0][3] * height / inf_h)
                            bbox = (x1, y1, x2, y2)

                            team_id, confidence = team_detector.classify_from_bbox(frame, bbox)

                            if team_id is not None:
                                box_color = color0 if team_id == 0 else color1
                                label = f"Team {team_id} ({confidence:.2f})"
                            else:
                                box_color = (128, 128, 128)
                                label = "Unknown"

                            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(vis, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Legend
                    cv2.rectangle(vis, (30, 30), (70, 70), color0, -1)
                    cv2.putText(vis, f"Team 0: {hex0}", (80, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color0, 2)

                    cv2.rectangle(vis, (30, 90), (70, 130), color1, -1)
                    cv2.putText(vis, f"Team 1: {hex1}", (80, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color1, 2)

                # Stamp frame source info
                cv2.putText(vis, f"WebRTC Frame #{frame_idx}", (30, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                teams_writer.write(vis)

            teams_writer.release()

            # Log final colors
            if team_detector.is_configured:
                hex0, hex1 = team_detector.get_team_colors_hex()
                logger.info(
                    f"[WebRTCFrameRecorder] Final team colors: Team0={hex0}, Team1={hex1}"
                )
            else:
                logger.warning("[WebRTCFrameRecorder] Team detector never calibrated")

            logger.info(
                f"[WebRTCFrameRecorder] Team-labeled video saved: {teams_path}"
            )

        except Exception:
            logger.exception("[WebRTCFrameRecorder] Error during frame processing")


# ---------------------------------------------------------------------------
# Integration helper — used by shot_processing_pipeline.py
# ---------------------------------------------------------------------------

_recorder: Optional[WebRTCFrameRecorder] = None


def is_webrtc_test_enabled() -> bool:
    """Check if the WEBRTC_TEAM_TEST env var is set."""
    return os.environ.get("WEBRTC_TEAM_TEST", "0") == "1"


def get_recorder() -> Optional[WebRTCFrameRecorder]:
    """Get or create the singleton recorder (only if enabled)."""
    global _recorder
    if not is_webrtc_test_enabled():
        return None
    if _recorder is None:
        _recorder = WebRTCFrameRecorder()
    return _recorder


def finalize_recorder() -> None:
    """Finalize and reset the singleton recorder."""
    global _recorder
    if _recorder is not None:
        _recorder.finalize()
        _recorder = None
