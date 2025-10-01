"""Annotation management for basketball videos"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

@dataclass
class PlayerBox:
    """Player bounding box annotation"""
    player_id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    team_id: int
    jersey_number: Optional[str] = None
    action_label: Optional[str] = None

@dataclass
class CourtKeypoint:
    """Court keypoint annotation"""
    keypoint_id: str
    position: List[float]  # [x, y]
    visibility: bool
    confidence: float

@dataclass
class ActionSegment:
    """Action annotation segment"""
    action_id: str
    player_id: str
    action_type: str
    start_frame: int
    end_frame: int
    confidence: float
    attributes: Dict[str, Any]

@dataclass
class FrameAnnotation:
    """Complete frame annotation"""
    frame_id: int
    timestamp: float
    video_id: str
    player_boxes: List[PlayerBox]
    court_keypoints: List[CourtKeypoint]
    actions: List[ActionSegment]
    metadata: Dict[str, Any]

class AnnotationManager:
    """Manager for basketball video annotations"""

    def __init__(self, annotation_root: str):
        self.annotation_root = Path(annotation_root)
        self.annotation_root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.annotation_root / "player_boxes").mkdir(exist_ok=True)
        (self.annotation_root / "court_keypoints").mkdir(exist_ok=True)
        (self.annotation_root / "actions").mkdir(exist_ok=True)
        (self.annotation_root / "player_ids").mkdir(exist_ok=True)
        (self.annotation_root / "quality_control").mkdir(exist_ok=True)

        self.annotation_schema = self._load_annotation_schema()

    def _load_annotation_schema(self) -> Dict[str, Any]:
        """Load annotation schema configuration"""
        return {
            "player_actions": [
                "shoot", "pass", "dribble", "defend", "rebound",
                "steal", "block", "assist", "turnover", "foul"
            ],
            "court_keypoints": [
                "center_circle", "free_throw_circle_left", "free_throw_circle_right",
                "three_point_arc_left_1", "three_point_arc_left_2", "three_point_arc_left_3",
                "three_point_arc_right_1", "three_point_arc_right_2", "three_point_arc_right_3",
                "baseline_left_corner", "baseline_right_corner",
                "sideline_left_corner", "sideline_right_corner", "halfcourt_line_center"
            ],
            "team_ids": [0, 1],
            "jersey_number_range": [0, 99]
        }

    def create_video_annotation_project(self, video_path: str, project_name: str) -> str:
        """Create new annotation project for a video"""
        project_id = str(uuid.uuid4())
        project_dir = self.annotation_root / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Project metadata
        project_metadata = {
            "project_id": project_id,
            "project_name": project_name,
            "video_path": video_path,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "annotation_progress": {
                "frames_annotated": 0,
                "total_frames": 0,
                "completion_percentage": 0.0
            },
            "annotators": [],
            "quality_control": {
                "reviewed": False,
                "approved": False,
                "issues": []
            }
        }

        # Save project metadata
        with open(project_dir / "metadata.json", 'w') as f:
            json.dump(project_metadata, f, indent=2)

        return project_id

    def add_frame_annotation(self, project_id: str, frame_annotation: FrameAnnotation) -> None:
        """Add frame annotation to project"""
        project_dir = self.annotation_root / "projects" / project_id
        annotations_dir = project_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        # Save frame annotation
        annotation_file = annotations_dir / f"frame_{frame_annotation.frame_id:06d}.json"
        with open(annotation_file, 'w') as f:
            json.dump(asdict(frame_annotation), f, indent=2)

        # Update project progress
        self._update_project_progress(project_id)

    def load_frame_annotation(self, project_id: str, frame_id: int) -> Optional[FrameAnnotation]:
        """Load frame annotation from project"""
        project_dir = self.annotation_root / "projects" / project_id
        annotation_file = project_dir / "annotations" / f"frame_{frame_id:06d}.json"

        if not annotation_file.exists():
            return None

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Convert dict back to dataclass
        return FrameAnnotation(
            frame_id=data['frame_id'],
            timestamp=data['timestamp'],
            video_id=data['video_id'],
            player_boxes=[PlayerBox(**box) for box in data['player_boxes']],
            court_keypoints=[CourtKeypoint(**kp) for kp in data['court_keypoints']],
            actions=[ActionSegment(**action) for action in data['actions']],
            metadata=data['metadata']
        )

    def export_annotations(self, project_id: str, format: str = 'json') -> str:
        """Export all annotations from project"""
        project_dir = self.annotation_root / "projects" / project_id
        annotations_dir = project_dir / "annotations"
        export_dir = project_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        if format == 'json':
            return self._export_json(project_id, export_dir)
        elif format == 'coco':
            return self._export_coco(project_id, export_dir)
        elif format == 'yolo':
            return self._export_yolo(project_id, export_dir)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, project_id: str, export_dir: Path) -> str:
        """Export annotations in JSON format"""
        project_dir = self.annotation_root / "projects" / project_id
        annotations_dir = project_dir / "annotations"

        all_annotations = []
        for annotation_file in sorted(annotations_dir.glob("frame_*.json")):
            with open(annotation_file, 'r') as f:
                all_annotations.append(json.load(f))

        export_file = export_dir / f"{project_id}_annotations.json"
        with open(export_file, 'w') as f:
            json.dump(all_annotations, f, indent=2)

        return str(export_file)

    def _export_coco(self, project_id: str, export_dir: Path) -> str:
        """Export annotations in COCO format"""
        # COCO format implementation
        coco_data = {
            "info": {
                "description": "Basketball Player Detection Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Basketball Analytics Team"
            },
            "categories": [
                {"id": 1, "name": "player", "supercategory": "person"}
            ],
            "images": [],
            "annotations": []
        }

        project_dir = self.annotation_root / "projects" / project_id
        annotations_dir = project_dir / "annotations"

        annotation_id = 1
        for annotation_file in sorted(annotations_dir.glob("frame_*.json")):
            with open(annotation_file, 'r') as f:
                frame_data = json.load(f)

            # Add image info
            image_info = {
                "id": frame_data['frame_id'],
                "file_name": f"frame_{frame_data['frame_id']:06d}.jpg",
                "width": frame_data['metadata'].get('width', 1920),
                "height": frame_data['metadata'].get('height', 1080)
            }
            coco_data["images"].append(image_info)

            # Add annotations
            for player_box in frame_data['player_boxes']:
                x1, y1, x2, y2 = player_box['bbox']
                width = x2 - x1
                height = y2 - y1

                annotation = {
                    "id": annotation_id,
                    "image_id": frame_data['frame_id'],
                    "category_id": 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "player_id": player_box['player_id'],
                    "team_id": player_box['team_id'],
                    "jersey_number": player_box.get('jersey_number')
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

        export_file = export_dir / f"{project_id}_coco.json"
        with open(export_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        return str(export_file)

    def _update_project_progress(self, project_id: str) -> None:
        """Update project annotation progress"""
        project_dir = self.annotation_root / "projects" / project_id
        metadata_file = project_dir / "metadata.json"

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Count annotated frames
        annotations_dir = project_dir / "annotations"
        if annotations_dir.exists():
            annotated_frames = len(list(annotations_dir.glob("frame_*.json")))
        else:
            annotated_frames = 0

        # Update progress
        total_frames = metadata['annotation_progress']['total_frames']
        if total_frames > 0:
            completion_percentage = (annotated_frames / total_frames) * 100
        else:
            completion_percentage = 0.0

        metadata['annotation_progress'].update({
            "frames_annotated": annotated_frames,
            "completion_percentage": completion_percentage,
            "last_updated": datetime.now().isoformat()
        })

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def validate_annotations(self, project_id: str) -> Dict[str, Any]:
        """Validate annotations in project"""
        project_dir = self.annotation_root / "projects" / project_id
        annotations_dir = project_dir / "annotations"

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_frames": 0,
                "total_players": 0,
                "total_keypoints": 0,
                "total_actions": 0
            }
        }

        for annotation_file in sorted(annotations_dir.glob("frame_*.json")):
            try:
                frame_annotation = self.load_frame_annotation(project_id,
                    int(annotation_file.stem.split('_')[1]))

                validation_results["statistics"]["total_frames"] += 1
                validation_results["statistics"]["total_players"] += len(frame_annotation.player_boxes)
                validation_results["statistics"]["total_keypoints"] += len(frame_annotation.court_keypoints)
                validation_results["statistics"]["total_actions"] += len(frame_annotation.actions)

                # Validate frame annotation
                frame_errors = self._validate_frame_annotation(frame_annotation)
                validation_results["errors"].extend(frame_errors)

            except Exception as e:
                validation_results["errors"].append(f"Error loading {annotation_file}: {str(e)}")
                validation_results["valid"] = False

        return validation_results

    def _validate_frame_annotation(self, frame_annotation: FrameAnnotation) -> List[str]:
        """Validate individual frame annotation"""
        errors = []

        # Validate player boxes
        for player_box in frame_annotation.player_boxes:
            if player_box.team_id not in self.annotation_schema["team_ids"]:
                errors.append(f"Invalid team_id: {player_box.team_id}")

            if player_box.jersey_number:
                try:
                    jersey_num = int(player_box.jersey_number)
                    if not (0 <= jersey_num <= 99):
                        errors.append(f"Invalid jersey number: {player_box.jersey_number}")
                except ValueError:
                    errors.append(f"Non-numeric jersey number: {player_box.jersey_number}")

            # Validate bbox coordinates
            x1, y1, x2, y2 = player_box.bbox
            if x1 >= x2 or y1 >= y2:
                errors.append(f"Invalid bbox coordinates: {player_box.bbox}")

        # Validate court keypoints
        for keypoint in frame_annotation.court_keypoints:
            if keypoint.keypoint_id not in self.annotation_schema["court_keypoints"]:
                errors.append(f"Invalid keypoint_id: {keypoint.keypoint_id}")

        # Validate actions
        for action in frame_annotation.actions:
            if action.action_type not in self.annotation_schema["player_actions"]:
                errors.append(f"Invalid action_type: {action.action_type}")

            if action.start_frame >= action.end_frame:
                errors.append(f"Invalid action timing: start={action.start_frame}, end={action.end_frame}")

        return errors