# Basketball Analysis System - Setup and Running Instructions

This is a basketball shot detection and analysis system with real-time video processing capabilities. The system consists of a Python FastAPI backend and a SvelteKit frontend.

## System Overview

- **Backend**: FastAPI server with WebRTC support for video streaming and YOLO-based shot detection
- **Frontend**: SvelteKit application with real-time data visualization
- **Communication**: WebRTC for video + REST API for control

## Prerequisites

### Backend Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Frontend Requirements
- Node.js 16+ and npm
- Modern web browser with WebRTC support

## Installation

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Note**: Some dependencies like `detectron2` may require additional setup. If you encounter issues:
- For PyTorch: Visit https://pytorch.org to get the correct installation command for your system
- For detectron2: Follow instructions at https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### 2. Frontend Setup

Navigate to the frontend directory:
```bash
cd ../frontend
```

Install npm dependencies:
```bash
npm install
```

## Running the Application

### Step 1: Start the Backend Server

From the `backend` directory with your virtual environment activated:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Options:
- `--reload`: Auto-reload on code changes (development mode)
- `--host 0.0.0.0`: Allow external connections
- `--port 8000`: Run on port 8000 (default)

The backend will be available at:
- Local: http://localhost:8000
- Health check: http://localhost:8000/health

### Step 2: Start the Frontend Development Server

From the `frontend` directory:

```bash
npm run dev
```

The frontend will be available at:
- Local: http://localhost:5173
- Network: Check terminal output for network URL

## Using the System

### Workflow

1. **Access the Application**
   - Open your browser to http://localhost:5173

2. **Set Up Court Calibration**
   - Click on the calibration interface
   - Mark 6 points on the basketball court in clockwise order:
     1. Baseline Left Sideline
     2. Baseline Left Penalty Box
     3. Baseline Right Penalty Box
     4. Baseline Right Sideline
     5. Free Throw Line Left
     6. Free Throw Line Right
   - Submit calibration via POST /calibration endpoint

3. **Start Shot Detection**
   - Once calibrated, start detection via POST /detection/start
   - The system will begin processing video frames
   - Shot statistics will be broadcast in real-time

4. **Monitor Statistics**
   - View real-time shot data on the dashboard
   - Check shot locations, success rates, and other metrics

### API Endpoints

#### Health & Debug
- `GET /health` - Check if backend is running
- `GET /debug/frame_stats` - View frame buffer statistics

#### Calibration
- `POST /calibration` - Set 6-point court calibration
- `GET /calibration` - Get current calibration state

#### Detection Control
- `POST /detection/start` - Start shot detection (requires calibration)
- `POST /detection/stop` - Stop detection and revert to dummy broadcast
- `POST /detection/reset` - Reset shot statistics
- `GET /detection/status` - Get current detection status
- `GET /detection/stats` - Get current shot statistics

#### WebRTC
- `POST /offer` - Handle WebRTC offer from frontend

## Troubleshooting

### Backend Issues

**Import errors or missing modules:**
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Use a different port
uvicorn main:app --reload --port 8001
# Update CORS settings in main.py if needed
```

**YOLO model not found:**
- The system uses ultralytics YOLO models
- Models are downloaded automatically on first run
- Ensure you have internet connection for initial setup

### Frontend Issues

**Dependencies not installed:**
```bash
npm install
```

**Port 5173 already in use:**
```bash
# Vite will automatically try the next available port
# Or specify a different port:
npm run dev -- --port 5174
```

**CORS errors:**
- Check that backend CORS settings allow your frontend URL
- Default allowed: http://localhost:5173, http://127.0.0.1:5173
- Update in backend/main.py if using different ports

### WebRTC Issues

**Video not showing:**
- Check browser console for errors
- Ensure camera permissions are granted
- Verify WebRTC is supported in your browser
- Check that /offer endpoint is responding correctly

## Development Commands

### Backend
```bash
# Run with auto-reload (development)
uvicorn main:app --reload

# Run in production mode
uvicorn main:app --host 0.0.0.0 --port 8000

# Run tests
pytest

# Format code
black .
```

### Frontend
```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run check

# Continuous type checking
npm run check:watch
```

## Project Structure

```
basketball_analysis/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── schema.py               # Pydantic schemas
│   ├── requirements.txt        # Python dependencies
│   ├── src/
│   │   └── utils/
│   │       └── webrtc_manager.py  # WebRTC connection handling
│   └── score_detection/        # Shot detection modules
│       ├── shot_processing_pipeline.py
│       └── streaming_shot_detector.py
├── frontend/
│   ├── package.json            # Node dependencies
│   ├── src/
│   │   ├── routes/            # SvelteKit pages
│   │   │   └── live/v2/       # Live detection interface
│   │   └── lib/               # Shared components
│   └── vite.config.js         # Vite configuration
└── INSTRUCTION.md             # This file
```

## Current Branch

You are currently on the `shot_localization` branch. This branch includes:
- 6-point court calibration
- Shot localization features
- Enhanced statistics tracking
- Updated UI components

## Additional Resources

- FastAPI docs: https://fastapi.tiangolo.com
- SvelteKit docs: https://kit.svelte.dev
- Ultralytics YOLO: https://docs.ultralytics.com
- WebRTC: https://webrtc.org

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review console/terminal output for error messages
3. Ensure all dependencies are correctly installed
4. Verify correct ports and network configuration
