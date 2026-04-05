from typing import Literal, Union, Optional, List
from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class TotalShots(BaseModel):
    made: int
    total: int


class Percentages(BaseModel):
    fieldGoal: float
    twoPoint: float
    threePoint: float


class GameStats(BaseModel):
    totalShots: TotalShots
    percentages: Percentages


class Shot(BaseModel):
    id: int
    timestamp_ms: int
    type: Literal["2pt", "3pt"]
    result: Literal["made", "missed"]
    location: str
    coord: Point
    team_id: Optional[int] = None  # 0 or 1, None if not detected
    team_confidence: float = 0.0


class FrameSyncPayload(BaseModel):
    type: Literal["frame_sync"] = "frame_sync"
    sequence_id: int
    video_timestamp_ms: int
    current_stats: GameStats


class ShotPayload(BaseModel):
    type: Literal["shot_event"] = "shot_event"
    video_timestamp_ms: int
    event_data: Shot
    updated_stats: GameStats


# Alias for backward compatibility
ShotDetectedPayload = ShotPayload

# Union type for type hinting elsewhere
WebRTCMessage = Union[FrameSyncPayload, ShotPayload]


# Calibration schemas
class CalibrationRequest(BaseModel):
    """Request schema for setting court calibration (4-point or 6-point)"""
    points: List[List[float]]  # 4 or 6 points [[x1,y1], [x2,y2], ...]
    image_width: int
    image_height: int
    mode: str = "4-point"


class CalibrationResponse(BaseModel):
    """Response schema for calibration operations"""
    success: bool
    is_calibrated: bool = False
    error: Optional[str] = None
    points: Optional[List[List[float]]] = None
    # Per-point reprojection errors in meters (same order as input points)
    point_errors: Optional[List[float]] = None
    # Avg reprojection error in meters
    avg_error: Optional[float] = None
    # Court outline projected to pixel space for overlay rendering.
    # List of polylines, each polyline is a list of [x, y] pixel coords.
    court_outline_pixels: Optional[List[List[List[float]]]] = None


class DetectionStatusResponse(BaseModel):
    """Response schema for detection status"""
    status: str
    is_detecting: bool = False
    is_calibrated: bool = False


class TeamColorsRequest(BaseModel):
    """Request schema for setting team colors"""
    team0_color: str  # Color name e.g., "red", "blue"
    team1_color: str


class TeamColorsResponse(BaseModel):
    """Response schema for team colors operations"""
    success: bool
    team0_color: Optional[str] = None
    team1_color: Optional[str] = None
    error: Optional[str] = None
