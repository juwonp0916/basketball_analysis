from typing import Literal, Union
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


# Union type for type hinting elsewhere
WebRTCMessage = Union[FrameSyncPayload, ShotPayload]
