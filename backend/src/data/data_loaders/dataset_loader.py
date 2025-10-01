"""Dataset loader for basketball analytics"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BasketballDataset(Dataset):
    """Basketball dataset for training and evaluation"""

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 task: str = 'detection',
                 transform: Optional[A.Compose] = None,
                 load_annotations: bool = True):
        """
        Initialize basketball dataset

        Args:
            data_root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            task: Task type ('detection', 'tracking', 'court_detection', 'action_recognition')
            transform: Albumentations transform pipeline
            load_annotations: Whether to load annotations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.task = task
        self.transform = transform
        self.load_annotations = load_annotations

        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_file = self.data_root / 'metadata' / f'{self.split}_metadata.json'

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples"""
        samples = []

        # Load frame-level samples for detection/court detection
        if self.task in ['detection', 'court_detection']:
            frames_dir = self.data_root / 'processed' / 'frames' / self.split

            for frame_file in sorted(frames_dir.glob('*.jpg')):
                sample = {
                    'frame_path': str(frame_file),
                    'frame_id': frame_file.stem,
                    'video_id': frame_file.stem.split('_')[0]
                }

                # Add annotation path if available
                if self.load_annotations:
                    annotation_file = self._get_annotation_path(frame_file)
                    if annotation_file.exists():
                        sample['annotation_path'] = str(annotation_file)

                samples.append(sample)

        # Load sequence-level samples for tracking/action recognition
        elif self.task in ['tracking', 'action_recognition']:
            sequences_file = self.data_root / 'metadata' / f'{self.split}_sequences.json'

            if sequences_file.exists():
                with open(sequences_file, 'r') as f:
                    sequences = json.load(f)

                for seq in sequences:
                    samples.append(seq)

        return samples

    def _get_annotation_path(self, frame_path: Path) -> Path:
        """Get annotation path for frame"""
        frame_id = frame_path.stem
        video_id = frame_id.split('_')[0]

        if self.task == 'detection':
            return self.data_root / 'annotations' / 'player_boxes' / f'{frame_id}.json'
        elif self.task == 'court_detection':
            return self.data_root / 'annotations' / 'court_keypoints' / f'{frame_id}.json'
        else:
            return Path('')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item"""
        sample = self.samples[idx]

        if self.task in ['detection', 'court_detection']:
            return self._get_frame_sample(sample)
        elif self.task in ['tracking', 'action_recognition']:
            return self._get_sequence_sample(sample)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _get_frame_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get frame-level sample"""
        # Load image
        image = cv2.imread(sample['frame_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = {
            'image': image,
            'frame_id': sample['frame_id'],
            'video_id': sample['video_id']
        }

        # Load annotations if available
        if 'annotation_path' in sample:
            annotations = self._load_frame_annotations(sample['annotation_path'])
            result.update(annotations)

        # Apply transforms
        if self.transform:
            if self.task == 'detection' and 'bboxes' in result:
                # Handle detection transforms
                transformed = self.transform(
                    image=image,
                    bboxes=result['bboxes'],
                    class_labels=result['class_labels']
                )
                result['image'] = transformed['image']
                result['bboxes'] = transformed['bboxes']
            elif self.task == 'court_detection' and 'keypoints' in result:
                # Handle keypoint transforms
                transformed = self.transform(
                    image=image,
                    keypoints=result['keypoints']
                )
                result['image'] = transformed['image']
                result['keypoints'] = transformed['keypoints']
            else:
                # Basic image transform
                transformed = self.transform(image=image)
                result['image'] = transformed['image']

        return result

    def _get_sequence_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get sequence-level sample"""
        sequence_dir = Path(sample['sequence_path'])

        # Load sequence frames
        frames = []
        frame_files = sorted(sequence_dir.glob('*.jpg'))

        for frame_file in frame_files:
            image = cv2.imread(str(frame_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)

        result = {
            'frames': np.array(frames),
            'sequence_id': sample['sequence_id'],
            'video_id': sample['video_id'],
            'start_frame': sample['start_frame'],
            'end_frame': sample['end_frame']
        }

        # Load sequence annotations
        if 'annotation_path' in sample:
            annotations = self._load_sequence_annotations(sample['annotation_path'])
            result.update(annotations)

        return result

    def _load_frame_annotations(self, annotation_path: str) -> Dict[str, Any]:
        """Load frame-level annotations"""
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        if self.task == 'detection':
            # Convert player boxes to format expected by transforms
            bboxes = []
            class_labels = []
            player_ids = []
            team_ids = []

            for player_box in annotations.get('player_boxes', []):
                bbox = player_box['bbox']  # [x1, y1, x2, y2]
                bboxes.append(bbox)
                class_labels.append(0)  # All players are class 0
                player_ids.append(player_box['player_id'])
                team_ids.append(player_box['team_id'])

            return {
                'bboxes': bboxes,
                'class_labels': class_labels,
                'player_ids': player_ids,
                'team_ids': team_ids
            }

        elif self.task == 'court_detection':
            # Convert court keypoints
            keypoints = []
            keypoint_labels = []

            for kp in annotations.get('court_keypoints', []):
                if kp['visibility']:
                    keypoints.append(kp['position'])  # [x, y]
                    keypoint_labels.append(kp['keypoint_id'])

            return {
                'keypoints': keypoints,
                'keypoint_labels': keypoint_labels
            }

        return {}

    def _load_sequence_annotations(self, annotation_path: str) -> Dict[str, Any]:
        """Load sequence-level annotations"""
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        if self.task == 'tracking':
            # Load tracking annotations
            return {
                'tracks': annotations.get('tracks', []),
                'track_ids': annotations.get('track_ids', [])
            }

        elif self.task == 'action_recognition':
            # Load action annotations
            return {
                'actions': annotations.get('actions', []),
                'action_labels': annotations.get('action_labels', [])
            }

        return {}

class DatasetBuilder:
    """Builder class for creating basketball datasets"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_transforms(self, split: str, task: str) -> A.Compose:
        """Build augmentation pipeline"""
        transforms = []

        if split == 'train':
            # Training augmentations
            transforms.extend([
                A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            ])
        else:
            # Validation/test transforms
            transforms.append(A.Resize(height=640, width=640))

        # Common transforms
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Task-specific transform parameters
        if task == 'detection':
            return A.Compose(transforms, bbox_params=A.BboxParams(
                format='pascal_voc', label_fields=['class_labels']))
        elif task == 'court_detection':
            return A.Compose(transforms, keypoint_params=A.KeypointParams(
                format='xy', remove_invisible=False))
        else:
            return A.Compose(transforms)

    def build_dataset(self, split: str, task: str) -> BasketballDataset:
        """Build dataset for given split and task"""
        transform = self.build_transforms(split, task)

        return BasketballDataset(
            data_root=self.config['data_root'],
            split=split,
            task=task,
            transform=transform,
            load_annotations=True
        )

    def build_dataloader(self, dataset: BasketballDataset,
                        batch_size: int = 16, shuffle: bool = True,
                        num_workers: int = 4) -> DataLoader:
        """Build dataloader for dataset"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn if dataset.task in ['tracking', 'action_recognition'] else None
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for sequence data"""
        # Handle variable-length sequences
        return {
            'sequences': [item['frames'] for item in batch],
            'sequence_ids': [item['sequence_id'] for item in batch],
            'video_ids': [item['video_id'] for item in batch]
        }