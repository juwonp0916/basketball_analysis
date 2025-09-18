"""Input/Output utility functions"""

import json
import yaml
import pickle
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import logging

class IOUtils:
    """Utility class for file I/O operations"""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @staticmethod
    def load_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from video file"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        metadata = {
            'filename': video_path.name,
            'path': str(video_path),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }

        cap.release()
        return metadata

    @staticmethod
    def save_annotations(annotations: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save annotations to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(annotations, f, indent=2, default=str)
        elif output_path.suffix == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(annotations, f)
        else:
            raise ValueError(f"Unsupported annotation format: {output_path.suffix}")

    @staticmethod
    def load_annotations(annotation_path: Union[str, Path]) -> Dict[str, Any]:
        """Load annotations from file"""
        annotation_path = Path(annotation_path)

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        if annotation_path.suffix == '.json':
            with open(annotation_path, 'r') as f:
                return json.load(f)
        elif annotation_path.suffix == '.pkl':
            with open(annotation_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_path.suffix}")

class VideoReader:
    """Video reading utility with frame extraction"""

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.metadata = IOUtils.load_video_metadata(video_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def read_frame(self, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """Read a specific frame or next frame"""
        if frame_idx is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.cap.read()
        return frame if ret else None

    def read_frames(self, start_idx: int = 0, end_idx: Optional[int] = None,
                   step: int = 1) -> List[np.ndarray]:
        """Read multiple frames"""
        if end_idx is None:
            end_idx = self.metadata['frame_count']

        frames = []
        for frame_idx in range(start_idx, end_idx, step):
            frame = self.read_frame(frame_idx)
            if frame is not None:
                frames.append(frame)

        return frames

    def extract_frames(self, output_dir: Union[str, Path],
                      start_idx: int = 0, end_idx: Optional[int] = None,
                      step: int = 1, format: str = 'jpg') -> List[str]:
        """Extract frames to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if end_idx is None:
            end_idx = self.metadata['frame_count']

        frame_paths = []
        for frame_idx in range(start_idx, end_idx, step):
            frame = self.read_frame(frame_idx)
            if frame is not None:
                frame_path = output_dir / f"frame_{frame_idx:06d}.{format}"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))

        return frame_paths

    def release(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()

class DatasetManager:
    """Dataset management utility"""

    def __init__(self, data_config_path: Union[str, Path]):
        self.config = IOUtils.load_config(data_config_path)
        self.data_root = Path(self.config['dataset']['raw_videos']['fixed_camera']).parent.parent

    def list_videos(self, category: str = 'all') -> List[Dict[str, Any]]:
        """List all videos in dataset"""
        videos = []

        if category in ['all', 'fixed']:
            fixed_dir = self.data_root / 'raw_videos' / 'fixed_camera'
            if fixed_dir.exists():
                for video_path in fixed_dir.glob('*.mp4'):
                    metadata = IOUtils.load_video_metadata(video_path)
                    metadata['category'] = 'fixed_camera'
                    videos.append(metadata)

        if category in ['all', 'moving']:
            for subcategory in ['broadcast', 'handheld', 'drone']:
                moving_dir = self.data_root / 'raw_videos' / 'moving_camera' / subcategory
                if moving_dir.exists():
                    for video_path in moving_dir.glob('*.mp4'):
                        metadata = IOUtils.load_video_metadata(video_path)
                        metadata['category'] = f'moving_camera_{subcategory}'
                        videos.append(metadata)

        return videos

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        videos = self.list_videos()

        stats = {
            'total_videos': len(videos),
            'total_duration': sum(v['duration'] for v in videos),
            'categories': {},
            'resolution_distribution': {},
            'fps_distribution': {}
        }

        for video in videos:
            category = video['category']
            stats['categories'][category] = stats['categories'].get(category, 0) + 1

            resolution = f"{video['width']}x{video['height']}"
            stats['resolution_distribution'][resolution] = stats['resolution_distribution'].get(resolution, 0) + 1

            fps = int(video['fps'])
            stats['fps_distribution'][fps] = stats['fps_distribution'].get(fps, 0) + 1

        return stats