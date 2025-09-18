#!/usr/bin/env python3
"""
Data Collection Pipeline Setup Script
Sets up the data collection and annotation infrastructure for the basketball analytics project
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.io_utils import IOUtils, DatasetManager
from data.annotation_tools.annotation_manager import AnnotationManager

class DataPipelineSetup:
    """Setup class for data collection pipeline"""

    def __init__(self, config_path: str):
        self.config = IOUtils.load_config(config_path)
        self.data_root = Path(self.config['dataset']['raw_videos']['fixed_camera']).parent.parent
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_root / 'data_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_directory_structure(self):
        """Create the complete directory structure for data collection"""
        self.logger.info("Creating directory structure...")

        directories = [
            # Raw video directories
            "data/raw_videos/fixed_camera",
            "data/raw_videos/moving_camera/broadcast",
            "data/raw_videos/moving_camera/handheld",
            "data/raw_videos/moving_camera/drone",

            # Annotation directories
            "data/annotations/player_boxes",
            "data/annotations/court_keypoints",
            "data/annotations/actions",
            "data/annotations/player_ids",

            # Processed data directories
            "data/processed/frames",
            "data/processed/tracks",
            "data/processed/statistics",

            # Metadata directory
            "data/metadata",

            # Temporary processing directories
            "data/temp/extraction",
            "data/temp/preprocessing",

            # Quality control directories
            "data/quality_control/failed",
            "data/quality_control/review",
        ]

        for directory in directories:
            dir_path = self.data_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def setup_annotation_system(self):
        """Setup annotation management system"""
        self.logger.info("Setting up annotation system...")

        annotation_root = self.data_root / "data" / "annotations"
        annotation_manager = AnnotationManager(str(annotation_root))

        # Create annotation guidelines
        guidelines = {
            "player_annotation_guidelines": {
                "bbox_guidelines": [
                    "Include full player body when visible",
                    "Tight bounding box around player",
                    "Include jersey number region",
                    "Handle partial occlusions appropriately"
                ],
                "team_assignment": [
                    "Assign team based on jersey color",
                    "Use team_id 0 and 1",
                    "Maintain consistency throughout video"
                ],
                "jersey_number_annotation": [
                    "Read jersey number clearly",
                    "Use string format (e.g., '23', '7')",
                    "Mark as null if not visible"
                ]
            },
            "court_annotation_guidelines": {
                "keypoint_annotation": [
                    "Mark exact intersection points",
                    "Use consistent naming convention",
                    "Mark visibility flag accurately"
                ],
                "frequency": "Annotate court keypoints every 30 frames"
            },
            "action_annotation_guidelines": {
                "action_types": [
                    "shoot: Player attempting shot",
                    "pass: Player passing ball",
                    "dribble: Player dribbling ball",
                    "defend: Player in defensive stance",
                    "rebound: Player collecting rebound"
                ],
                "temporal_boundaries": [
                    "Mark exact start and end frames",
                    "Avoid overlapping actions for same player",
                    "Include action context in attributes"
                ]
            },
            "quality_standards": {
                "minimum_requirements": [
                    "All visible players must be annotated",
                    "Court keypoints annotated when visible",
                    "Actions annotated for key players",
                    "Consistent player IDs maintained"
                ],
                "review_process": [
                    "Peer review for complex scenes",
                    "Automated validation checks",
                    "Final approval by senior annotator"
                ]
            }
        }

        guidelines_file = annotation_root / "annotation_guidelines.json"
        with open(guidelines_file, 'w') as f:
            json.dump(guidelines, f, indent=2)

        self.logger.info(f"Created annotation guidelines: {guidelines_file}")

    def create_data_collection_templates(self):
        """Create templates for data collection"""
        self.logger.info("Creating data collection templates...")

        templates_dir = self.data_root / "data" / "templates"
        templates_dir.mkdir(exist_ok=True)

        # Video metadata template
        video_metadata_template = {
            "video_id": "unique_video_identifier",
            "filename": "video_filename.mp4",
            "source": "broadcast/handheld/drone/fixed",
            "game_info": {
                "date": "YYYY-MM-DD",
                "teams": ["Team A", "Team B"],
                "venue": "Venue Name",
                "game_type": "professional/college/amateur",
                "duration": "game_duration_minutes"
            },
            "technical_specs": {
                "resolution": [1920, 1080],
                "fps": 30,
                "codec": "h264",
                "file_size_mb": 0
            },
            "camera_info": {
                "camera_type": "broadcast/handheld/fixed/drone",
                "movement_type": "static/panning/following/aerial",
                "position": "court_side/elevated/center/corner",
                "quality_score": 5  # 1-5 scale
            },
            "annotation_status": {
                "assigned_annotator": "annotator_name",
                "start_date": "YYYY-MM-DD",
                "completion_status": "pending/in_progress/completed/reviewed",
                "quality_checked": false
            }
        }

        with open(templates_dir / "video_metadata_template.json", 'w') as f:
            json.dump(video_metadata_template, f, indent=2)

        # Data collection checklist
        collection_checklist = {
            "pre_collection": [
                "Identify data sources",
                "Obtain necessary permissions",
                "Setup collection environment",
                "Prepare storage infrastructure"
            ],
            "during_collection": [
                "Verify video quality",
                "Check technical specifications",
                "Document source information",
                "Organize files properly"
            ],
            "post_collection": [
                "Validate file integrity",
                "Create metadata entries",
                "Backup raw files",
                "Schedule for annotation"
            ],
            "quality_criteria": {
                "minimum_resolution": "720p",
                "minimum_duration": "60 seconds",
                "maximum_duration": "3600 seconds",
                "required_elements": [
                    "Visible court lines",
                    "Clear player visibility",
                    "Consistent lighting",
                    "Minimal motion blur"
                ]
            }
        }

        with open(templates_dir / "data_collection_checklist.json", 'w') as f:
            json.dump(collection_checklist, f, indent=2)

        self.logger.info(f"Created templates in: {templates_dir}")

    def setup_quality_control(self):
        """Setup quality control processes"""
        self.logger.info("Setting up quality control processes...")

        qc_dir = self.data_root / "data" / "quality_control"
        qc_dir.mkdir(exist_ok=True)

        # Quality control configuration
        qc_config = {
            "validation_checks": {
                "video_format": {
                    "allowed_formats": [".mp4", ".avi", ".mov"],
                    "min_resolution": [720, 1280],
                    "min_fps": 20,
                    "max_file_size_gb": 5
                },
                "content_validation": {
                    "basketball_court_detection": true,
                    "player_visibility_check": true,
                    "lighting_quality_check": true,
                    "motion_blur_detection": true
                },
                "annotation_validation": {
                    "bbox_consistency": true,
                    "temporal_consistency": true,
                    "label_validation": true,
                    "completeness_check": true
                }
            },
            "review_process": {
                "automatic_validation": true,
                "peer_review_threshold": 0.9,
                "senior_review_required": true,
                "final_approval_needed": true
            },
            "rejection_criteria": [
                "Poor video quality",
                "Insufficient player visibility",
                "Incorrect annotation format",
                "Missing required elements"
            ]
        }

        with open(qc_dir / "quality_control_config.json", 'w') as f:
            json.dump(qc_config, f, indent=2)

        self.logger.info(f"Created quality control config: {qc_dir}")

    def create_collection_scripts(self):
        """Create utility scripts for data collection"""
        scripts_dir = self.data_root / "scripts" / "data_collection"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Video validation script
        validation_script = '''#!/usr/bin/env python3
"""Video validation script for data collection"""

import cv2
import argparse
from pathlib import Path
import json

def validate_video(video_path):
    """Validate video file"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False, "Cannot open video file"

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    # Validation checks
    if width < 1280 or height < 720:
        return False, f"Resolution too low: {width}x{height}"

    if fps < 20:
        return False, f"Frame rate too low: {fps}"

    if duration < 60:
        return False, f"Video too short: {duration} seconds"

    if duration > 3600:
        return False, f"Video too long: {duration} seconds"

    return True, "Video passed validation"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate basketball video")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()

    valid, message = validate_video(args.video_path)
    print(f"{'PASS' if valid else 'FAIL'}: {message}")
'''

        with open(scripts_dir / "validate_video.py", 'w') as f:
            f.write(validation_script)

        self.logger.info(f"Created collection scripts in: {scripts_dir}")

    def generate_collection_report(self):
        """Generate data collection setup report"""
        self.logger.info("Generating setup report...")

        report = {
            "setup_date": "2024-01-01",  # Will be set dynamically
            "project_structure": "Created complete directory structure",
            "annotation_system": "Configured annotation management system",
            "quality_control": "Setup validation and review processes",
            "collection_targets": {
                "total_videos": self.config['targets']['total_videos'],
                "fixed_camera": self.config['targets']['fixed_camera_videos'],
                "moving_camera": self.config['targets']['moving_camera_videos'],
                "total_hours": self.config['targets']['total_hours']
            },
            "next_steps": [
                "Begin data collection from identified sources",
                "Setup annotation team and training",
                "Start with pilot annotation project",
                "Implement automated quality checks",
                "Monitor collection progress regularly"
            ]
        }

        report_file = self.data_root / "data_pipeline_setup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Setup complete! Report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Setup basketball analytics data pipeline")
    parser.add_argument("--config", default="config/data_configs/dataset_config.yaml",
                       help="Path to dataset configuration file")
    parser.add_argument("--force", action="store_true",
                       help="Force recreation of existing directories")

    args = parser.parse_args()

    try:
        setup = DataPipelineSetup(args.config)

        # Execute setup steps
        setup.create_directory_structure()
        setup.setup_annotation_system()
        setup.create_data_collection_templates()
        setup.setup_quality_control()
        setup.create_collection_scripts()
        setup.generate_collection_report()

        print("\\n✅ Data pipeline setup completed successfully!")
        print("Next steps:")
        print("1. Review annotation guidelines")
        print("2. Begin data collection")
        print("3. Setup annotation team")
        print("4. Start pilot annotation project")

    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()