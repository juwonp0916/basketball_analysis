"""Video preprocessing for basketball analytics"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class VideoProcessor:
    """Video preprocessing and frame extraction"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_video(self, video_path: str, output_dir: str,
                     extract_frames: bool = True,
                     frame_interval: int = 1) -> Dict[str, Any]:
        """Process single video file"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        metadata = self._extract_video_metadata(cap, video_path)

        # Validate video quality
        quality_check = self._validate_video_quality(metadata)
        if not quality_check['valid']:
            self.logger.warning(f"Video quality issues: {quality_check['issues']}")

        # Extract frames if requested
        if extract_frames:
            frames_dir = output_dir / 'frames'
            frames_dir.mkdir(exist_ok=True)

            frame_paths = self._extract_frames(cap, frames_dir,
                                             video_path.stem,
                                             frame_interval)
            metadata['extracted_frames'] = len(frame_paths)
            metadata['frame_paths'] = frame_paths

        cap.release()

        # Save metadata
        metadata_file = output_dir / f"{video_path.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata

    def batch_process_videos(self, video_dir: str, output_dir: str,
                           max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process multiple videos in parallel"""
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for video_file in video_files:
                video_output_dir = Path(output_dir) / video_file.stem
                future = executor.submit(self.process_video, str(video_file),
                                       str(video_output_dir))
                futures.append((video_file, future))

            # Collect results
            for video_file, future in tqdm(futures, desc="Processing videos"):
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Processed: {video_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {video_file.name}: {str(e)}")

        return results

    def _extract_video_metadata(self, cap: cv2.VideoCapture,
                               video_path: Path) -> Dict[str, Any]:
        """Extract comprehensive video metadata"""
        metadata = {
            'filename': video_path.name,
            'file_path': str(video_path),
            'file_size_mb': video_path.stat().st_size / (1024 * 1024),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'duration_seconds': 0,
            'aspect_ratio': 0,
            'quality_metrics': {}
        }

        # Calculate derived properties
        if metadata['fps'] > 0:
            metadata['duration_seconds'] = metadata['frame_count'] / metadata['fps']

        if metadata['height'] > 0:
            metadata['aspect_ratio'] = metadata['width'] / metadata['height']

        # Analyze video quality
        metadata['quality_metrics'] = self._analyze_video_quality(cap)

        return metadata

    def _analyze_video_quality(self, cap: cv2.VideoCapture) -> Dict[str, Any]:
        """Analyze video quality metrics"""
        quality_metrics = {
            'brightness_mean': 0,
            'brightness_std': 0,
            'contrast_mean': 0,
            'sharpness_mean': 0,
            'motion_blur_score': 0,
            'noise_level': 0
        }

        # Sample frames for quality analysis
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)

        brightness_values = []
        contrast_values = []
        sharpness_values = []
        motion_blur_scores = []

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Brightness (mean intensity)
                brightness = np.mean(gray)
                brightness_values.append(brightness)

                # Contrast (standard deviation of intensity)
                contrast = np.std(gray)
                contrast_values.append(contrast)

                # Sharpness (Laplacian variance)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_values.append(sharpness)

                # Motion blur detection (gradient magnitude)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                motion_blur = np.mean(np.sqrt(sobelx**2 + sobely**2))
                motion_blur_scores.append(motion_blur)

        # Calculate statistics
        if brightness_values:
            quality_metrics['brightness_mean'] = np.mean(brightness_values)
            quality_metrics['brightness_std'] = np.std(brightness_values)
            quality_metrics['contrast_mean'] = np.mean(contrast_values)
            quality_metrics['sharpness_mean'] = np.mean(sharpness_values)
            quality_metrics['motion_blur_score'] = np.mean(motion_blur_scores)

        return quality_metrics

    def _validate_video_quality(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video quality against requirements"""
        quality_config = self.config.get('quality', {})

        validation_result = {
            'valid': True,
            'issues': []
        }

        # Check resolution
        min_resolution = quality_config.get('min_resolution', [720, 1280])
        if (metadata['height'] < min_resolution[0] or
            metadata['width'] < min_resolution[1]):
            validation_result['valid'] = False
            validation_result['issues'].append(
                f"Resolution too low: {metadata['width']}x{metadata['height']}")

        # Check frame rate
        min_fps = quality_config.get('min_fps', 24)
        if metadata['fps'] < min_fps:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Frame rate too low: {metadata['fps']}")

        # Check duration
        min_duration = quality_config.get('min_video_length', 60)
        max_duration = quality_config.get('max_video_length', 3600)

        if metadata['duration_seconds'] < min_duration:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Video too short: {metadata['duration_seconds']}s")

        if metadata['duration_seconds'] > max_duration:
            validation_result['issues'].append(f"Video too long: {metadata['duration_seconds']}s")

        # Check quality metrics
        quality_metrics = metadata.get('quality_metrics', {})

        # Check brightness (should be reasonable)
        brightness = quality_metrics.get('brightness_mean', 0)
        if brightness < 50 or brightness > 200:
            validation_result['issues'].append(f"Poor brightness: {brightness}")

        # Check sharpness (higher is better)
        sharpness = quality_metrics.get('sharpness_mean', 0)
        if sharpness < 100:
            validation_result['issues'].append(f"Low sharpness: {sharpness}")

        return validation_result

    def _extract_frames(self, cap: cv2.VideoCapture, output_dir: Path,
                       video_name: str, interval: int = 1) -> List[str]:
        """Extract frames from video"""
        frame_paths = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename

                # Save frame
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))

        return frame_paths

    def create_training_splits(self, processed_data_dir: str,
                             split_ratios: Dict[str, float] = None) -> Dict[str, List[str]]:
        """Create training/validation/test splits"""
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

        processed_data_dir = Path(processed_data_dir)

        # Collect all video metadata files
        metadata_files = list(processed_data_dir.glob('*/*_metadata.json'))

        # Shuffle and split
        np.random.shuffle(metadata_files)

        n_total = len(metadata_files)
        n_train = int(n_total * split_ratios['train'])
        n_val = int(n_total * split_ratios['val'])

        splits = {
            'train': metadata_files[:n_train],
            'val': metadata_files[n_train:n_train + n_val],
            'test': metadata_files[n_train + n_val:]
        }

        # Save split information
        splits_info = {}
        for split_name, files in splits.items():
            splits_info[split_name] = [str(f) for f in files]

            # Create split-specific metadata
            split_metadata = {
                'split': split_name,
                'video_count': len(files),
                'videos': []
            }

            for metadata_file in files:
                with open(metadata_file, 'r') as f:
                    video_metadata = json.load(f)
                split_metadata['videos'].append(video_metadata)

            # Save split metadata
            split_file = processed_data_dir / f'{split_name}_metadata.json'
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2, default=str)

        # Save overall splits information
        splits_file = processed_data_dir / 'dataset_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits_info, f, indent=2)

        self.logger.info(f"Created dataset splits: {dict((k, len(v)) for k, v in splits.items())}")

        return splits_info

class DatasetOrganizer:
    """Organize dataset structure for training"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.logger = logging.getLogger(__name__)

    def organize_for_training(self, processed_data_dir: str) -> None:
        """Organize processed data for training"""
        processed_data_dir = Path(processed_data_dir)

        # Create training structure
        training_structure = {
            'frames': ['train', 'val', 'test'],
            'annotations': ['player_boxes', 'court_keypoints', 'actions', 'player_ids']
        }

        for main_dir, subdirs in training_structure.items():
            for subdir in subdirs:
                target_dir = self.data_root / 'processed' / main_dir / subdir
                target_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset splits
        splits_file = processed_data_dir / 'dataset_splits.json'
        with open(splits_file, 'r') as f:
            splits = json.load(f)

        # Organize frames and annotations by split
        for split_name, metadata_files in splits.items():
            split_frames_dir = self.data_root / 'processed' / 'frames' / split_name

            for metadata_file in metadata_files:
                with open(metadata_file, 'r') as f:
                    video_metadata = json.load(f)

                # Move frames to split directory
                if 'frame_paths' in video_metadata:
                    for frame_path in video_metadata['frame_paths']:
                        frame_path = Path(frame_path)
                        if frame_path.exists():
                            target_path = split_frames_dir / frame_path.name
                            if not target_path.exists():
                                frame_path.rename(target_path)

        self.logger.info(f"Organized dataset for training in: {self.data_root / 'processed'}")

    def create_dataset_statistics(self) -> Dict[str, Any]:
        """Create comprehensive dataset statistics"""
        stats = {
            'total_videos': 0,
            'total_frames': 0,
            'total_duration_hours': 0,
            'splits': {},
            'quality_distribution': {},
            'resolution_distribution': {},
            'fps_distribution': {}
        }

        # Analyze each split
        for split in ['train', 'val', 'test']:
            split_metadata_file = self.data_root / f'{split}_metadata.json'

            if split_metadata_file.exists():
                with open(split_metadata_file, 'r') as f:
                    split_data = json.load(f)

                split_stats = {
                    'video_count': len(split_data['videos']),
                    'total_frames': sum(v.get('extracted_frames', 0) for v in split_data['videos']),
                    'total_duration': sum(v.get('duration_seconds', 0) for v in split_data['videos']),
                    'avg_fps': np.mean([v.get('fps', 0) for v in split_data['videos']])
                }

                stats['splits'][split] = split_stats
                stats['total_videos'] += split_stats['video_count']
                stats['total_frames'] += split_stats['total_frames']
                stats['total_duration_hours'] += split_stats['total_duration'] / 3600

        # Save statistics
        stats_file = self.data_root / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats