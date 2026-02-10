import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.utils.webrtc_manager import ConnectionManager
from schema import CalibrationRequest, CalibrationResponse, DetectionStatusResponse, TeamColorsRequest, TeamColorsResponse

VIDEO_DIR = Path(__file__).parent / "score_detection"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

app = FastAPI()
manager = ConnectionManager()
_broadcast_task: Optional[asyncio.Task] = None


class Offer(BaseModel):
    sdp: str
    type: str


@app.on_event("startup")
async def on_startup() -> None:
    # Start dummy broadcaster so frontend should immediately receive messages once DC opens
    manager.start_dummy_broadcast(interval_sec=0.5)
    logger.info("FastAPI startup complete")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _broadcast_task
    if _broadcast_task:
        _broadcast_task.cancel()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/debug/frame_stats")
async def debug_frame_stats() -> Dict[str, Any]:
    return manager.frame_buffer.stats()


@app.get("/video/{filename}")
async def serve_video(filename: str):
    """Serve .mp4 video files from the score_detection directory."""
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are allowed")
    file_path = VIDEO_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.post("/offer")
async def offer(offer: Offer) -> Dict[str, str]:
    # This is called by the frontend's sendOffer(...)
    return await manager.handle_offer(offer.model_dump())


# ============ Calibration Endpoints ============

@app.post("/calibration", response_model=CalibrationResponse)
async def set_calibration(request: CalibrationRequest) -> CalibrationResponse:
    """
    Set court calibration for shot localization.

    Supports two modes:
    - "4-point": Paint box only (4 corners of penalty box)
      1. Baseline Left Penalty Box
      2. Baseline Right Penalty Box
      3. Free Throw Line Left
      4. Free Throw Line Right

    - "6-point": Full baseline (all 6 points)
      1. Baseline Left Sideline
      2. Baseline Left Penalty Box
      3. Baseline Right Penalty Box
      4. Baseline Right Sideline
      5. Free Throw Line Left
      6. Free Throw Line Right
    """
    # Validate mode
    if request.mode not in ["4-point", "6-point"]:
        return CalibrationResponse(
            success=False,
            is_calibrated=manager.is_calibrated,
            error=f"Invalid calibration mode '{request.mode}'. Must be '4-point' or '6-point'"
        )

    # Validate point count based on mode
    expected_points = 4 if request.mode == "4-point" else 6
    if len(request.points) != expected_points:
        return CalibrationResponse(
            success=False,
            is_calibrated=manager.is_calibrated,
            error=f"{request.mode} calibration requires {expected_points} points, got {len(request.points)}"
        )

    # Validate each point has 2 coordinates
    for i, point in enumerate(request.points):
        if len(point) != 2:
            return CalibrationResponse(
                success=False,
                is_calibrated=manager.is_calibrated,
                error=f"Point {i+1} must have 2 coordinates (x, y)"
            )

    # Set calibration with mode
    success = manager.set_calibration(
        points=request.points,
        dimensions=(request.image_width, request.image_height),
        mode=request.mode
    )

    return CalibrationResponse(
        success=success,
        is_calibrated=manager.is_calibrated,
        points=request.points if success else None,
        error=None if success else "Failed to set calibration"
    )


@app.get("/calibration", response_model=CalibrationResponse)
async def get_calibration() -> CalibrationResponse:
    """Get current calibration state"""
    cal = manager.get_calibration()
    return CalibrationResponse(
        success=True,
        is_calibrated=cal["is_calibrated"],
        points=cal["points"]
    )


# ============ Team Colors Endpoints ============

@app.post("/team-colors", response_model=TeamColorsResponse)
async def set_team_colors(request: TeamColorsRequest) -> TeamColorsResponse:
    """
    Set team jersey colors for team-based shot tracking.

    Available colors: red, blue, green, yellow, orange, purple, pink, cyan,
    white, black, gray, brown, navy, maroon, lime, teal, gold
    """
    if request.team0_color == request.team1_color:
        return TeamColorsResponse(
            success=False,
            error="Team colors must be different"
        )

    success = manager.set_team_colors(request.team0_color, request.team1_color)

    if success:
        return TeamColorsResponse(
            success=True,
            team0_color=request.team0_color,
            team1_color=request.team1_color
        )
    else:
        return TeamColorsResponse(
            success=False,
            error="Invalid color name. Available: red, blue, green, yellow, orange, purple, pink, cyan, white, black, gray, brown, navy, maroon, lime, teal, gold"
        )


# ============ Detection Control Endpoints ============

@app.post("/detection/start", response_model=DetectionStatusResponse)
async def start_detection() -> DetectionStatusResponse:
    """
    Start real shot detection.

    Requires calibration to be set first. This will stop the dummy broadcast
    and start processing frames with the YOLO-based shot detector.
    """
    if not manager.is_calibrated:
        raise HTTPException(
            status_code=400,
            detail="Calibration required before starting detection. Call POST /calibration first."
        )

    success = await manager.start_shot_detection()

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to start shot detection"
        )

    return DetectionStatusResponse(
        status="started",
        is_detecting=manager.is_detecting,
        is_calibrated=manager.is_calibrated
    )


@app.post("/detection/stop", response_model=DetectionStatusResponse)
async def stop_detection() -> DetectionStatusResponse:
    """
    Stop shot detection and revert to dummy broadcast.
    """
    await manager.stop_shot_detection()

    return DetectionStatusResponse(
        status="stopped",
        is_detecting=manager.is_detecting,
        is_calibrated=manager.is_calibrated
    )


@app.post("/detection/reset", response_model=DetectionStatusResponse)
async def reset_stats() -> DetectionStatusResponse:
    """
    Reset accumulated shot statistics.

    This clears all shot counts and resets the detector state.
    Does not affect calibration.
    """
    manager.reset_stats()

    return DetectionStatusResponse(
        status="reset",
        is_detecting=manager.is_detecting,
        is_calibrated=manager.is_calibrated
    )


@app.get("/detection/status", response_model=DetectionStatusResponse)
async def get_detection_status() -> DetectionStatusResponse:
    """Get current detection status"""
    return DetectionStatusResponse(
        status="detecting" if manager.is_detecting else "idle",
        is_detecting=manager.is_detecting,
        is_calibrated=manager.is_calibrated
    )


@app.get("/detection/stats")
async def get_detection_stats() -> Dict[str, Any]:
    """Get current shot detection statistics"""
    stats = manager.get_current_stats()
    if stats is None:
        return {
            "is_detecting": False,
            "stats": None
        }
    return {
        "is_detecting": True,
        "stats": stats
    }


# Allow your frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
