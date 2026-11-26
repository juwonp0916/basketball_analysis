# Setup and usage

[Project overview](README.md) · [System architecture](docs/ARCHITECTURE.md)

The live application consists of a FastAPI backend and a SvelteKit frontend. Run both services locally, then open the dashboard at `/live/v4`.

## Prerequisites

- A Python environment compatible with [backend/requirements.txt](backend/requirements.txt). The dependency ranges are broad; this repository does not provide a locked Python environment.
- Node.js **22.12 or later in the 22.x series**, or another version satisfying the engines in [frontend/package-lock.json](frontend/package-lock.json).
- A camera and a browser supporting WebRTC and camera capture.
- A stationary view of the basket and court markings, with two visually distinguishable jersey colors for team attribution.

## 1. Clone the repository

```bash
git clone https://github.com/juwonp0916/basketball_analysis.git
cd basketball_analysis
```

## 2. Install the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv/Scripts/Activate.ps1`.

Before launching, check [score_detection/config.yaml](backend/score_detection/config.yaml):

| Setting | Current value | Configuration |
| --- | --- | --- |
| `weights_path` | `weights/260413_best.pt` | Resolved relative to `backend/score_detection/`. This checkpoint is tracked in the repository. |
| `classes` | `ball`, `made`, `person`, `rim`, `shoot` | Must match the loaded checkpoint's class order. |
| `device` | `mps` | The checked-in setting targets Apple Metal. Set it to `cpu` when MPS is unavailable; CUDA use requires a compatible PyTorch installation and device setting. |

The `input` entry is a legacy local video path; the live pipeline receives frames over WebRTC. The custom basketball checkpoint is required for the documented class mapping.

Start the backend from `backend/`:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Check [backend health](http://127.0.0.1:8000/health) and [interactive API documentation](http://127.0.0.1:8000/docs).

## 3. Install and start the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Open **[the analysis dashboard](http://127.0.0.1:5173/live/v4)**. The `/` route still contains the SvelteKit starter page.

The backend address is configured in [frontend/src/lib/index.ts](frontend/src/lib/index.ts). Backend CORS currently allows `http://localhost:5173` and `http://127.0.0.1:5173`; changing ports or hosts requires updating these settings in [backend/main.py](backend/main.py).

## 4. Run a camera session

1. Start the camera and grant browser access.
2. Select **Start Analysis** to enter calibration.
3. Mark the four points in the order shown by the overlay: baseline left paint corner, baseline right paint corner, free-throw line left corner, free-throw line right corner.
4. Inspect the projected court lines, drag the points to refine alignment, and select **Confirm**.
5. The browser establishes WebRTC and requests detection when the analytics channel opens. Team-color initialization runs as frames arrive.
6. Use the team views, shot filters, and event log to inspect the session. Select **Stop** to close the session and release the video source.

Keep the camera fixed after calibration. A changed view requires recalibration.

### Recorded-video simulation

The simulation control loads `http://127.0.0.1:8000/video/match1.mp4`. Supply your own clip at `backend/sample_videos/match1.mp4`; that file is not part of the tracked application assets. The frontend uses the video element's `captureStream()` method, so simulation also depends on browser support for that API.

## Current integration caveats

- **Streaming localization:** `StreamingShotDetector._setup_localization()` passes `court_spec` to `ShotLocalizer`, whose current constructor does not accept that argument. The exception is caught and the streaming localizer remains unset. Calibration previews use a separate initialization path and can still appear; a successful preview therefore does not establish that live shot localization is working. This interface mismatch needs resolving before relying on live court positions.
- **Session state:** analytics are held in memory. The dashboard retains up to 200 shots and 50 log entries; there is no persistent game-history store.
- **Deployment:** the current application shares backend calibration and detection state and is intended for one analysis session. Remote deployment requires additional browser-origin, HTTPS, and WebRTC connectivity configuration.

These instructions describe the checked-in code. They do not represent a verified fresh-environment installation or a completed camera-session test.

## Useful endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health |
| POST | `/offer` | WebRTC SDP signaling |
| GET / POST | `/calibration` | Read or submit court calibration |
| POST | `/detection/start` | Start detection after calibration |
| POST | `/detection/stop` | Stop the detection pipeline |
| POST | `/detection/reset` | Reset detector statistics |
| GET | `/detection/status` | Calibration and detection state |
| GET | `/detection/stats` | Current statistics |
| GET | `/debug/frame_stats` | Inspect the frame buffer |

## Frontend development commands

Run from `frontend/`:

```bash
npm run check
npm run build
```

See the [offline tool guide](backend/tools/README.md) for localization utilities, coordinate checks, and their input requirements. Model-dependent scripts and tests have their own media and checkpoint requirements; inspect their inputs before running them.
