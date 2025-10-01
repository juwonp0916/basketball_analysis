"""Basketball court detection and keypoint extraction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torchvision.transforms as transforms
from torchvision.models import resnet50
import logging

class CourtKeypointNet(nn.Module):
    """Neural network for basketball court keypoint detection"""

    def __init__(self, num_keypoints: int = 14, backbone: str = 'resnet50'):
        super(CourtKeypointNet, self).__init__()
        self.num_keypoints = num_keypoints

        # Backbone network
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            # Remove final layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature Pyramid Network layers
        self.fpn_conv1 = nn.Conv2d(backbone_channels, 256, 1)
        self.fpn_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        # Keypoint detection head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1)
        )

        # Visibility prediction head
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_keypoints),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Extract features
        features = self.backbone(x)

        # FPN processing
        fpn_features = self.fpn_conv1(features)
        fpn_features = self.fpn_conv2(fpn_features)

        # Keypoint heatmaps
        heatmaps = self.keypoint_head(fpn_features)

        # Visibility predictions
        visibility = self.visibility_head(fpn_features)

        return {
            'heatmaps': heatmaps,
            'visibility': visibility
        }

class CourtDetector:
    """Court detection and keypoint extraction system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = CourtKeypointNet(
            num_keypoints=config['keypoints']['num_keypoints'],
            backbone=config['architecture']['backbone']
        ).to(self.device)

        # Initialize preprocessing
        self.transform = self._build_transform()

        # Court template for geometric constraints
        self.court_template = self._load_court_template()

    def _build_transform(self) -> transforms.Compose:
        """Build image preprocessing transform"""
        input_size = self.config['input']['image_size']
        mean = self.config['input']['mean']
        std = self.config['input']['std']

        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def _load_court_template(self) -> Dict[str, Any]:
        """Load basketball court template with standard keypoints"""
        # NBA court dimensions (in feet)
        court_length = 94
        court_width = 50

        # Standard court keypoints in real-world coordinates
        template = {
            'dimensions': {'length': court_length, 'width': court_width},
            'keypoints': {
                'center_circle': [court_length/2, court_width/2],
                'free_throw_circle_left': [19, court_width/2],
                'free_throw_circle_right': [75, court_width/2],
                'three_point_arc_left_1': [14, 3],
                'three_point_arc_left_2': [14, court_width/2],
                'three_point_arc_left_3': [14, 47],
                'three_point_arc_right_1': [80, 3],
                'three_point_arc_right_2': [80, court_width/2],
                'three_point_arc_right_3': [80, 47],
                'baseline_left_corner': [0, 0],
                'baseline_right_corner': [court_length, 0],
                'sideline_left_corner': [0, court_width],
                'sideline_right_corner': [court_length, court_width],
                'halfcourt_line_center': [court_length/2, 0]
            }
        }

        return template

    def detect_court(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect court keypoints in image"""
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Post-process outputs
        keypoints = self._extract_keypoints(outputs, image.shape[:2])

        # Apply geometric constraints
        keypoints = self._apply_geometric_constraints(keypoints)

        # Estimate court boundaries
        court_boundaries = self._estimate_court_boundaries(keypoints, image.shape[:2])

        return {
            'keypoints': keypoints,
            'court_boundaries': court_boundaries,
            'confidence_score': self._calculate_detection_confidence(keypoints)
        }

    def _extract_keypoints(self, outputs: Dict[str, torch.Tensor],
                          image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Extract keypoints from model outputs"""
        heatmaps = outputs['heatmaps'].squeeze(0).cpu().numpy()
        visibility = outputs['visibility'].squeeze(0).cpu().numpy()

        keypoints = []
        keypoint_names = self.config['keypoints']['names']

        for i, keypoint_name in enumerate(keypoint_names):
            heatmap = heatmaps[i]
            vis_score = visibility[i]

            # Find peak in heatmap
            peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = heatmap[peak_idx]

            # Convert to image coordinates
            heatmap_h, heatmap_w = heatmap.shape
            image_h, image_w = image_shape

            x = (peak_idx[1] / heatmap_w) * image_w
            y = (peak_idx[0] / heatmap_h) * image_h

            keypoint = {
                'name': keypoint_name,
                'position': [float(x), float(y)],
                'confidence': float(confidence),
                'visibility': float(vis_score),
                'visible': vis_score > 0.5 and confidence > 0.3
            }

            keypoints.append(keypoint)

        return keypoints

    def _apply_geometric_constraints(self, keypoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply geometric constraints based on court template"""
        # Filter out low-confidence keypoints
        valid_keypoints = [kp for kp in keypoints if kp['visible']]

        if len(valid_keypoints) < 4:
            self.logger.warning("Insufficient keypoints for geometric constraints")
            return keypoints

        # Apply court geometry constraints
        # For example, ensure center circle is at court center relative to baselines
        constrained_keypoints = []

        for keypoint in keypoints:
            # Apply position refinement based on court template
            refined_position = self._refine_keypoint_position(keypoint, valid_keypoints)
            keypoint['position'] = refined_position
            constrained_keypoints.append(keypoint)

        return constrained_keypoints

    def _refine_keypoint_position(self, keypoint: Dict[str, Any],
                                 reference_keypoints: List[Dict[str, Any]]) -> List[float]:
        """Refine keypoint position using geometric constraints"""
        # Simple refinement - can be enhanced with more sophisticated geometry
        return keypoint['position']

    def _estimate_court_boundaries(self, keypoints: List[Dict[str, Any]],
                                 image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Estimate court boundaries from keypoints"""
        visible_keypoints = [kp for kp in keypoints if kp['visible']]

        if len(visible_keypoints) < 4:
            return {'corners': [], 'valid': False}

        # Extract positions
        positions = np.array([kp['position'] for kp in visible_keypoints])

        # Estimate court corners using convex hull
        from scipy.spatial import ConvexHull
        if len(positions) >= 4:
            hull = ConvexHull(positions)
            corner_indices = hull.vertices
            corners = positions[corner_indices].tolist()
        else:
            corners = positions.tolist()

        return {
            'corners': corners,
            'valid': len(corners) >= 4,
            'area': self._calculate_polygon_area(corners) if len(corners) >= 3 else 0
        }

    def _calculate_polygon_area(self, corners: List[List[float]]) -> float:
        """Calculate area of polygon defined by corners"""
        if len(corners) < 3:
            return 0

        # Shoelace formula
        x = [corner[0] for corner in corners]
        y = [corner[1] for corner in corners]

        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        return area

    def _calculate_detection_confidence(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate overall detection confidence"""
        visible_keypoints = [kp for kp in keypoints if kp['visible']]

        if not visible_keypoints:
            return 0.0

        # Average confidence weighted by visibility
        total_confidence = sum(kp['confidence'] * kp['visibility'] for kp in visible_keypoints)
        total_weight = sum(kp['visibility'] for kp in visible_keypoints)

        return total_confidence / total_weight if total_weight > 0 else 0.0

    def load_model(self, model_path: str) -> None:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.logger.info(f"Loaded court detection model from {model_path}")

    def save_model(self, model_path: str, epoch: int, optimizer_state: Dict = None) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }

        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, model_path)
        self.logger.info(f"Saved court detection model to {model_path}")

class TraditionalCourtDetector:
    """Traditional computer vision approach for court detection"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect court lines using traditional CV methods"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Line detection using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=50, maxLineGap=10)

        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_lines.append({
                    'start': [int(x1), int(y1)],
                    'end': [int(x2), int(y2)],
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })

        return detected_lines

    def detect_circles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect circular features (center circle, free throw circles)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Circle detection using Hough transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=200)

        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append({
                    'center': [int(x), int(y)],
                    'radius': int(r)
                })

        return detected_circles

    def combine_detections(self, lines: List[Dict[str, Any]],
                          circles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine line and circle detections into court structure"""
        court_structure = {
            'lines': lines,
            'circles': circles,
            'keypoints': self._extract_keypoints_from_lines(lines),
            'confidence': self._calculate_traditional_confidence(lines, circles)
        }

        return court_structure

    def _extract_keypoints_from_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract keypoints from line intersections"""
        keypoints = []

        # Find line intersections
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                intersection = self._line_intersection(line1, line2)
                if intersection:
                    keypoints.append({
                        'position': intersection,
                        'type': 'line_intersection',
                        'confidence': 0.8
                    })

        return keypoints

    def _line_intersection(self, line1: Dict[str, Any],
                          line2: Dict[str, Any]) -> Optional[List[float]]:
        """Calculate intersection point of two lines"""
        x1, y1 = line1['start']
        x2, y2 = line1['end']
        x3, y3 = line2['start']
        x4, y4 = line2['end']

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None  # Lines are parallel

        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection within both line segments
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return [float(x), float(y)]

        return None

    def _calculate_traditional_confidence(self, lines: List[Dict[str, Any]],
                                        circles: List[Dict[str, Any]]) -> float:
        """Calculate confidence for traditional detection"""
        # Simple heuristic based on number of detected features
        line_score = min(len(lines) / 10.0, 1.0)  # Expect ~10 major lines
        circle_score = min(len(circles) / 3.0, 1.0)  # Expect ~3 circles

        return (line_score + circle_score) / 2.0