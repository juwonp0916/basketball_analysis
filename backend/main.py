import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.webrtc_manager import ConnectionManager

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


@app.post("/offer")
async def offer(offer: Offer) -> Dict[str, str]:
    # This is called by the frontend's sendOffer(...)
    return await manager.handle_offer(offer.model_dump())


# Allow your frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
