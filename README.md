# Real-Time Basketball Shot Analytics

**From a single camera feed to a live shot chart, scoring events, and team statistics.**

A computer vision project built for the **HKUST Computer Science Final Year Project, 2025–2026 · Team GCH6**. The system combines basketball detection models, court geometry, jersey-color clustering, and browser-based video streaming to analyze shooting activity during a session.

[System architecture](docs/ARCHITECTURE.md) · [Setup and usage](INSTRUCTION.md) · [Backend](backend/) · [Frontend](frontend/)

## Project overview

The project explores how basketball teams can access in-session shot analytics using a standard camera and a local analysis machine. A browser captures video, a Python backend detects and interprets shooting events, and a live dashboard presents where shots were taken, whether they went in, and which team attempted them.

The application is designed around a **stationary camera, a FIBA half-court, and two teams with distinguishable jerseys**. Its focus is team-level shooting analysis.

## What we built

| Component | Implementation |
| --- | --- |
| **Live video analysis** | Browser camera capture and WebRTC transport to a FastAPI backend, with analytics returned through a WebRTC data channel. A recorded-video simulation mode supports development. |
| **Shot detection** | Custom basketball YOLO model, temporal voting for shooting poses, shooter-position estimation, and spatial/temporal deduplication of repeated detections. |
| **Made/missed classification** | Ball and rim tracking with score-region and trajectory checks to classify shot attempts. |
| **Court calibration and localization** | Interactive court-point selection, OpenCV homography, and projection of the shooter's estimated ground-contact position into FIBA court coordinates. |
| **Automatic team differentiation** | Jersey-region extraction, dominant-color estimation in CIELAB space, and unsupervised clustering into two teams, with outlier filtering and confidence scores. |
| **Live analytics dashboard** | Half-court shot charts, team comparison, field-goal and 2PT/3PT percentages, shot filters, and an event log linked to chart markers. |

## How it works

1. **Capture and calibrate.** Start the camera, mark four paint corners, and fine-tune the projected court overlay.
2. **Stream frames.** The browser sends video through WebRTC; the backend feeds decoded frames into an asynchronous processing pipeline.
3. **Interpret shots.** Detection and temporal logic identify attempts and outcomes. Court geometry supplies shot location and type, while jersey colors supply team attribution.
4. **Update the dashboard.** Structured shot events and statistics return over the `analytics` data channel and update the charts and event log.

See the [architecture guide](docs/ARCHITECTURE.md) for the modules behind each stage and the distinction between the report's design and the current code.

## Engineering highlights

- **Streaming pipeline:** video ingestion, model inference, and dashboard updates are separated through an asynchronous frame buffer and inference in a worker thread.
- **Temporal reasoning:** detections are evaluated across frames, with cooldowns and deduplication to avoid counting the same shooting action repeatedly.
- **Geometric localization:** a homography maps image coordinates onto the court plane; a projected court overlay helps the operator inspect calibration.
- **Unsupervised team assignment:** team colors are learned from visible players across frames, with periodic rechecks as the session continues.
- **Integrated interface:** calibration, video controls, team views, shot charts, and event inspection are available in one SvelteKit dashboard.

## Technology stack

| Layer | Technologies |
| --- | --- |
| Detection and image processing | Python, Ultralytics YOLO, PyTorch, OpenCV, NumPy |
| Team-color clustering | scikit-learn, CIELAB color features |
| API and streaming | FastAPI, Uvicorn, aiortc, WebRTC, Pydantic |
| Dashboard | SvelteKit, Svelte 5, TypeScript, Tailwind CSS, LayerChart |

## Explore the repository

```text
basketball_analysis/
├── backend/
│   ├── main.py                         # REST API and WebRTC signaling
│   ├── schema.py                       # Calibration and analytics payloads
│   ├── score_detection/                # Streaming detection and court geometry
│   │   ├── shot_processing_pipeline.py # Frame processing and event delivery
│   │   ├── streaming_shot_detector.py  # Shot detection and outcome logic
│   │   ├── localization.py             # Homography and court projection
│   │   ├── team_detector.py            # Jersey-color clustering
│   │   └── config.yaml                 # Model path and inference device
│   ├── src/utils/webrtc_manager.py     # Connections and frame ingestion
│   ├── scripts/                        # Offline processing utilities
│   ├── tests/                          # Team-detection checks
│   └── tools/                          # Offline utilities and diagnostics
│       ├── localization/               # Image/video localization and keypoints
│       ├── diagnostics/                # Court and coordinate checks
│       └── visualization/              # Clustering visualization experiment
├── frontend/src/
│   ├── routes/live/v4/+page.svelte     # Main analysis dashboard
│   └── lib/                            # Charts, calibration, WebRTC utilities
├── docs/ARCHITECTURE.md                # Technical walkthrough
└── INSTRUCTION.md                     # Local setup and usage
```

The tree highlights the live application. See the [tool guide](backend/tools/README.md) for offline localization commands and diagnostics. Supporting media and calibration data remain in their existing locations.

## Run locally

Follow [Setup and usage](INSTRUCTION.md) to install dependencies, select an inference device, and launch both services. The dashboard is at **[localhost:5173/live/v4](http://localhost:5173/live/v4)**; the root route is a SvelteKit starter page.

This repository is a research prototype. The setup guide records current integration caveats, including a localization constructor mismatch in the streaming path.

## Project context

Developed by **Team GCH6** at **The Hong Kong University of Science and Technology**, advised by **Prof. CHAN Sheung-Han Gary**.

The project builds on [HoopIQ — Automated Basketball Highlight Generation](https://github.com/candylyt/automated-basketball-highlight-generation), extending the earlier work toward live video ingestion, streaming analytics, interactive calibration, and team-level shot visualization.

The project scope follows the final report, *Advancements in Basketball Game Analysis Using Computer Vision: Real-time Basketball Analysis System*. Implementation details in this README and the linked guides have been checked against the source in this repository.
