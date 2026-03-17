# Basketball Individual Player Analytics with Moving Camera Support

## Overview
This project extends automated basketball highlight generation capabilities to support moving cameras and individual player tracking. Building upon the foundation of [HoopIQ](https://github.com/candylyt/automated-basketball-highlight-generation), this system adds:

1. **Moving Camera Support**: Robust analytics for videos with camera movement using homography and court calibration
2. **Individual Player Tracking**: Per-player statistics and action recognition with player re-identification

## Features

### Current Capabilities (from HoopIQ base)
- ✅ Real-time basketball game analysis
- ✅ Automatic highlight generation
- ✅ Team-level statistics extraction
- ✅ Fixed camera support

### New Extensions (FYP Additions)
- 🚧 **Moving Camera Analytics**: Court calibration and homography estimation for moving cameras
- 🚧 **Individual Player Tracking**: Multi-object tracking with player re-identification
- 🚧 **Per-Player Statistics**: Individual shooting, passing, and defensive metrics
- 🚧 **Action Recognition**: Individual player action classification (shoot, pass, dribble, etc.)
- 🚧 **Enhanced Highlights**: Individual player highlight compilation

## Project Structure

```
basketball_analysis/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── data/           # Data processing and annotation tools
│   ├── models/         # ML models (court detection, tracking, etc.)
│   ├── tracking/       # Multi-object tracking and re-identification
│   ├── analytics/      # Statistics and highlight generation
│   ├── calibration/    # Court calibration and homography
│   └── utils/          # Utility functions
├── scripts/            # Training and processing scripts
├── notebooks/          # Jupyter notebooks for analysis
├── tests/             # Unit and integration tests
├── docs/              # Documentation
├── data/              # Dataset storage
├── models/            # Trained model storage
├── results/           # Experiment results
└── frontend/          # Web interface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/basketball-individual-analytics.git
cd basketball-individual-analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Quick Start

### Data Collection and Annotation
```bash
# Set up data collection pipeline
python scripts/setup_data_pipeline.py

# Process raw videos
python scripts/process_videos.py --input data/raw_videos --output data/processed
```

### Model Training
```bash
# Train court detection model
python scripts/train_models.py --model court_detection

# Train player tracking model
python scripts/train_models.py --model player_tracking
```

### Running Analysis
```bash
# Process a single video
python scripts/process_video.py --input video.mp4 --output results/

# Batch process videos
python scripts/batch_process.py --input_dir data/raw_videos
```


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Base project: [HoopIQ - Automated Basketball Highlight Generation](https://github.com/candylyt/automated-basketball-highlight-generation)
- HKUST Final Year Project supervision and support
- Open source computer vision and machine learning communities
