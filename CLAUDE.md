# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Basketball individual player analytics system with real-time shot detection and localization, built for moving cameras with court calibration support. Uses YOLO-based shot detection, WebRTC streaming, and homography-based court localization.

## Commands

### Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run dev server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # Dev server (localhost:5173)
npm run build      # Production build
npm run check      # TypeScript type checking
npm run preview    # Preview production build
```

## Architecture

The system has three main layers:

**Frontend (SvelteKit + TypeScript)**
`/live/v4/+page.svelte` is the primary UI. It captures camera streams, handles court calibration via click-based point marking, and displays shot charts and statistics. Communicates with the backend over WebRTC (video + data channel).

**Backend (FastAPI + aiortc)**
`backend/main.py` exposes REST endpoints for calibration, team colors, and detection control. `backend/src/utils/webrtc_manager.py` manages WebRTC peer connections, a `SharedFrameBuffer` async queue, and the detection pipeline lifecycle.

**Detection Pipeline (Python)**
`score_detection/shot_processing_pipeline.py` consumes frames from the buffer, runs YOLO inference in a thread pool with frame-skipping when the queue exceeds 10 frames, then broadcasts results over the WebRTC data channel. The pipeline calls:
- `StreamingShotDetector` — YOLO-based detection with temporal voting across frames
- `ShotLocalizer` — homography transform from pixel coords to FIBA court coords
- `StreamingTeamDetector` — jersey HSV color matching

## Key Data Flows

**Shot detection:** Camera → WebRTC track → `SharedFrameBuffer` → `ShotProcessingPipeline` → YOLO inference → `ShotLocalizer` → broadcast `ShotPayload` via data channel → frontend `ShotChart`

**Calibration:** User clicks points on video overlay → POST `/calibration` → homography matrix stored in `ConnectionManager` → used by `ShotLocalizer` per-connection

**Message types on data channel:**
- `FrameSyncPayload` — sent every frame
- `ShotPayload` — sent on shot detection (includes `coord: {x, y}`, `type`, `result`, `team_id`)

## Coordinate Systems

**Critical:** Frontend SVG court has the basket at the TOP (y=5.25 in viewBox 0–47). The yDomain is `[47, 0]` (inverted).

Backend FIBA coordinates: origin at baseline center, X: 0–15m (left→right), Y: 0–14m (baseline→halfcourt). Basket at `(7.5, 1.575)`.

Conversion in `shot_processing_pipeline.py`:
```python
coord_y = (14.0 - fiba_y) / 14.0 * 47.0  # Inverts Y so baseline → top of chart
```
This was a known bug fix — shots near the basket (low FIBA Y) should appear near the top.

## Calibration Modes

Defined in `score_detection/constants.py`:
- `"4-point"` — paint box corners only (minimal, for partially visible courts)
- `"6-point"` — baseline corners + free throw line (standard)
- `"4-point-court"` — full court corners
- `"6-point-3pt"` — paint box + 3PT arc elbows

The homography uses OpenCV (`cv2.findHomography`) computed from these FIBA reference points.

## YOLO Model

- Config: `backend/score_detection/config.yaml`
- Weights: `backend/score_detection/weights/new_weight.pt`
- Classes: `["ball", "made", "person", "rim", "shoot"]` (5 classes)
- Default device: CPU; change to `"gpu"` in config.yaml for GPU inference

## Frontend Shot Chart

`ShotChart.svelte` renders a LayerChart-based court. Key parameters:
- ViewBox: 50×47 (normalized FIBA-proportional)
- Basket position: `(25, 5.25)` — near top
- Shot coordinates come from backend as `{x, y}` in the 0–50/0–47 space
- Green = made, red = missed
