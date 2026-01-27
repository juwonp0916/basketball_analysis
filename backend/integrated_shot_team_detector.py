"""
Integrated Shot and Team Detection System
Combines shot detection with jersey team classification for team-based shot counting.
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from threading import Thread, Event, Lock
import time
from typing import Callable, Optional, Tuple, List, Dict
from ultralytics import YOLO
import yaml

# Import from existing modules
from jersey_team_classifier_enhanced import EnhancedJerseyTeamClassifier
from utils import detect_score, in_score_region


class TeamAwareScoreCounter:
    """Enhanced score counter that tracks statistics per team with quarter breakdown."""

    def __init__(self, quarters: List[float] = None):
        """
        Initialize team-aware score counter.

        Args:
            quarters: List of quarter end times in seconds [Q1_end, Q2_end, Q3_end, Q4_end]
        """
        self.quarters = quarters or []
        self.current_quarter = 0

        # Team statistics: {team_id: {'makes': [Q1, Q2, Q3, Q4], 'attempts': [...]}}
        self.team_stats = {
            0: {'makes': [0] * (len(quarters) + 1), 'attempts': [0] * (len(quarters) + 1)},
            1: {'makes': [0] * (len(quarters) + 1), 'attempts': [0] * (len(quarters) + 1)}
        }

        # Team colors (RGB tuples)
        self.team_colors = {0: None, 1: None}

        # Detailed shot log for analysis
        self.shot_log = []

    def _get_quarter(self, timestamp: float) -> int:
        """Determine which quarter based on timestamp."""
        for i, q_end in enumerate(self.quarters):
            if timestamp <= q_end:
                return i
        return len(self.quarters)  # After all quarters

    def record_shot(self, timestamp: float, team: int, made: bool,
                    location: Optional[Tuple[float, float]] = None,
                    confidence: float = 0.0):
        """
        Record a shot attempt.

        Args:
            timestamp: Time in seconds when shot occurred
            team: Team ID (0 or 1)
            made: True if shot was made, False if missed
            location: Optional (x, y) normalized coordinates of shooter
            confidence: Team classification confidence
        """
        quarter = self._get_quarter(timestamp)

        # Update statistics
        if made:
            self.team_stats[team]['makes'][quarter] += 1
        self.team_stats[team]['attempts'][quarter] += 1

        # Log shot details
        self.shot_log.append({
            'timestamp': timestamp,
            'team': team,
            'made': made,
            'quarter': quarter,
            'location': location,
            'confidence': confidence
        })

    def get_stats(self) -> Dict:
        """Get complete statistics."""
        stats = {
            'team_0': self.team_stats[0].copy(),
            'team_1': self.team_stats[1].copy(),
            'shot_log': self.shot_log.copy()
        }

        # Calculate field goal percentages
        for team_id in [0, 1]:
            fg_pct = []
            for q in range(len(self.team_stats[team_id]['makes'])):
                attempts = self.team_stats[team_id]['attempts'][q]
                makes = self.team_stats[team_id]['makes'][q]
                fg_pct.append(makes / attempts * 100 if attempts > 0 else 0.0)
            stats[f'team_{team_id}']['fg_percentage'] = fg_pct

        return stats

    def set_team_color(self, team_id: int, rgb_color: Tuple[int, int, int]):
        """Set the RGB color for a team."""
        self.team_colors[team_id] = rgb_color

    def _rgb_to_color_name(self, rgb):
        """Convert RGB to approximate color name."""
        if rgb is None:
            return "Unknown"

        r, g, b = rgb

        # Calculate brightness and saturation
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        brightness = (max_val + min_val) / 2

        # Very dark colors
        if max_val < 80:
            if b > r and b > g and b - max(r,g) > 10:
                return "Dark Blue/Navy"
            elif r < g < b:
                return "Dark Purple/Navy"
            else:
                return "Black/Dark"

        # Check for purple/violet (blue bias with some red)
        if b > g and r > g and abs(b - r) < 40:
            return "Purple/Violet"

        # Check for brown/orange (red > green > blue pattern)
        if r > g > b and r - b > 30:
            if brightness < 100:
                return "Brown/Dark Orange"
            else:
                return "Orange/Brown"

        # Standard color classification
        if r > 150 and g < 100 and b < 100:
            return "Red"
        elif r < 100 and g < 100 and b > 150:
            return "Blue"
        elif r < 100 and g > 150 and b < 100:
            return "Green"
        elif r > 150 and g > 150 and b < 100:
            return "Yellow"
        elif r > 150 and g < 150 and b < 100:
            return "Orange"
        elif r < 100 and g > 100 and b > 100:
            return "Cyan"
        elif r > 150 and g > 150 and b > 150:
            return "White"
        elif abs(r-g) < 20 and abs(g-b) < 20:
            if brightness < 80:
                return "Dark Gray"
            elif brightness > 180:
                return "Light Gray"
            else:
                return "Gray"
        else:
            return f"RGB({r},{g},{b})"

    def print_report(self):
        """Print formatted statistics report."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("GAME STATISTICS - TEAM BREAKDOWN")
        print("="*60)

        for team_id in [0, 1]:
            color_name = self._rgb_to_color_name(self.team_colors[team_id])
            print(f"\nTEAM {team_id} ({color_name}):")
            print("-" * 40)

            team_data = stats[f'team_{team_id}']
            total_makes = sum(team_data['makes'])
            total_attempts = sum(team_data['attempts'])
            total_fg = total_makes / total_attempts * 100 if total_attempts > 0 else 0.0

            if total_attempts > 0:
                # Per quarter breakdown
                for q in range(len(team_data['makes'])):
                    quarter_name = f"Q{q+1}" if q < 4 else "OT"
                    makes = team_data['makes'][q]
                    attempts = team_data['attempts'][q]
                    fg_pct = team_data['fg_percentage'][q]

                    if attempts > 0:
                        print(f"  {quarter_name}: {makes}/{attempts} ({fg_pct:.1f}%)")

                print(f"  TOTAL: {total_makes}/{total_attempts} ({total_fg:.1f}%)")
            else:
                print(f"  No shots detected")

        print("\n" + "="*60)


class IntegratedShotTeamDetector:
    """
    Integrated shot and team detection system.
    Combines YOLO-based shot detection with jersey team classification.
    """

    def __init__(self,
                 video_path: str,
                 config_path: str = 'config.yaml',
                 on_shot_detected: Optional[Callable] = None,
                 on_complete: Optional[Callable] = None,
                 show_video: bool = False,
                 save_video: bool = False,
                 output_path: Optional[str] = None,
                 reinit_interval: int = 300,
                 team0_color: Optional[str] = None,
                 team1_color: Optional[str] = None):
        """
        Initialize integrated detector.

        Args:
            video_path: Path to video file
            config_path: Path to configuration YAML
            on_shot_detected: Callback(timestamp, team, made, location, confidence)
            on_complete: Callback when processing complete
            show_video: Whether to display video during processing
            save_video: Whether to save output video with visualizations
            output_path: Path for output video (default: video_path_integrated.mp4)
            reinit_interval: Frames between team color reinitialization
            team0_color: Optional manual team 0 color
            team1_color: Optional manual team 1 color
        """
        self.video_path = video_path
        self.on_shot_detected = on_shot_detected or self._default_shot_callback
        self.on_complete = on_complete or (lambda: None)
        self.show_video = show_video
        self.save_video = save_video
        self.output_path = output_path or video_path.replace('.mp4', '_integrated.mp4')

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load YOLO models
        print("Loading YOLO models...")
        self.ball_rim_model = YOLO(self.config['weights_path'])
        self.shoot_model = YOLO(self.config['weights_path_shoot'])

        # Initialize jersey classifier
        print("Initializing jersey team classifier...")
        self.jersey_classifier = EnhancedJerseyTeamClassifier(
            reinit_interval=reinit_interval,
            temporal_window=5,
            confidence_threshold=0.3,
            user_team0_color=team0_color,
            user_team1_color=team1_color
        )

        # Shot detection state - using original algorithm structure
        self.ball_positions = []  # List of tuples: ((x, y), frame_count, w, h, conf)
        self.rim_positions = []   # List of tuples: ((x, y), frame_count, w, h, conf)
        self.shoot_poses = deque(maxlen=60)

        self.last_shot_frame = 0
        self.shot_cooldown_frames = int(self.fps * 2.5)  # 2.5 seconds for miss
        self.shot_cooldown_made_frames = int(self.fps * 3)  # 3 seconds for make
        self.ball_entered = False
        self.last_point_in_region = None
        self.attempt_time = 0
        self.attempt_detection_interval = int(self.fps * 1.5)  # Increased from 0.3 to 1.5 seconds

        # Enhanced ball tracking for occlusion handling
        self.ball_velocity = np.array([0.0, 0.0])
        self.frames_since_ball_detection = 0
        self.max_occlusion_frames = int(self.fps * 0.5)  # Allow 0.5 seconds of occlusion
        self.predicted_ball_position = None
        self.original_ball_size = None  # (avg_w, avg_h) of tracked ball
        self.original_ball_conf = None  # avg confidence of tracked ball
        self.max_position_jump = 200  # Maximum pixels ball can move between frames

        # Player tracking for team classification
        self.player_tracks = {}
        self.next_track_id = 0

        # Threading
        self.stop_event = Event()
        self.processing_thread = None

        # Statistics
        self.score_counter = TeamAwareScoreCounter(
            quarters=[600, 1200, 1800, 2400]  # 10min quarters
        )

        # Video writer for output
        self.video_writer = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            print(f"Output video will be saved to: {self.output_path}")

        print("Initialization complete!")

    def _default_shot_callback(self, timestamp, team, made, location, confidence):
        """Default callback that prints shot info."""
        team_str = f"Team {team}" if team is not None else "Unknown Team"
        result = "MADE" if made else "MISSED"
        conf_str = f"(confidence: {confidence:.2f})" if team is not None else ""
        print(f"[{timestamp:.2f}s] {team_str} {result} shot {conf_str}")

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _get_scoring_region(self, rim_pos):
        """Define scoring region around rim - matches original algorithm."""
        if rim_pos is None:
            return None

        x, y, w, h = rim_pos
        cx, cy = x + w/2, y + h/2

        # Scoring region: vertical zone around rim (original dimensions)
        # Extended bottom boundary to capture ball falling through hoop
        region = {
            'x_min': cx - w * 2,
            'x_max': cx + w * 2,
            'y_min': cy - h * 5.5,  # Increased from 3.5 to 5.5 to match original
            'y_max': cy + h * 2.5   # Extended from 0.9 to 2.5 to track ball falling through
        }
        return region

    def _ball_in_region(self, ball_pos, region):
        """Check if ball is in scoring region."""
        if ball_pos is None or region is None:
            return False

        x, y, w, h = ball_pos
        cx, cy = x + w/2, y + h/2

        return (region['x_min'] <= cx <= region['x_max'] and
                region['y_min'] <= cy <= region['y_max'])

    def _predict_ball_position(self):
        """Predict ball position based on velocity and gravity."""
        if not self.ball_positions:
            return None

        last_pos = np.array(self.ball_positions[-1][0], dtype=float)
        gravity = 0.5  # Pixels per frame squared
        vx, vy = self.ball_velocity
        pred = last_pos.copy()

        # Simulate trajectory with gravity
        for _ in range(self.frames_since_ball_detection):
            vy += gravity
            pred[0] += vx
            pred[1] += vy

        return (int(pred[0]), int(pred[1]))

    def _update_ball_profile(self):
        """Update the profile of the tracked ball based on recent stable detections."""
        if len(self.ball_positions) >= 5:
            recent = self.ball_positions[-5:]
            avg_w = np.mean([p[2] for p in recent])
            avg_h = np.mean([p[3] for p in recent])
            avg_conf = np.mean([p[4] for p in recent])
            self.original_ball_size = (avg_w, avg_h)
            self.original_ball_conf = avg_conf

    def _score_ball_detection(self, center, w, h, conf, target_pos):
        """
        Score a ball detection based on multiple factors to find the best match.
        Returns a score where higher is better.
        """
        score = 0.0

        # Factor 1: Distance to predicted/last position (lower distance is better)
        if target_pos:
            dist = np.sqrt((center[0] - target_pos[0])**2 + (center[1] - target_pos[1])**2)
            # Normalize distance score (max 100 points, decreases with distance)
            dist_score = max(0, 100 - dist * 0.5)
            score += dist_score * 0.4  # 40% weight

        # Factor 2: Confidence score (higher is better)
        conf_score = conf * 100
        score += conf_score * 0.3  # 30% weight

        # Factor 3: Size similarity to original ball (closer is better)
        if self.original_ball_size:
            orig_w, orig_h = self.original_ball_size
            size_diff = abs(w - orig_w) + abs(h - orig_h)
            avg_size = (orig_w + orig_h) / 2
            if avg_size > 0:
                size_similarity = max(0, 100 - (size_diff / avg_size) * 50)
                score += size_similarity * 0.3  # 30% weight
        else:
            # If no original ball profile, just use confidence
            score += conf * 30

        return score

    def _select_best_ball(self, ball_detections, frame_count):
        """
        Select the best ball detection from multiple candidates using multi-factor scoring.
        Returns the best detection as ((x, y), frame_count, w, h, conf) or None.
        """
        if not ball_detections:
            # No detections - use prediction if available
            self.frames_since_ball_detection += 1
            if self.ball_positions and self.frames_since_ball_detection <= self.max_occlusion_frames:
                self.predicted_ball_position = self._predict_ball_position()
            else:
                self.predicted_ball_position = None
            return None

        # Determine target position for scoring
        target_pos = None
        if self.predicted_ball_position:
            target_pos = self.predicted_ball_position
        elif self.ball_positions:
            target_pos = self.ball_positions[-1][0]

        # Score all detections
        best_detection = None
        best_score = -1

        for center, w, h, conf in ball_detections:
            # Check for sudden position jumps (likely different object)
            if self.ball_positions and len(self.ball_positions) > 0:
                last_center = self.ball_positions[-1][0]
                frames_since_last = frame_count - self.ball_positions[-1][1]
                dist = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)

                # Reject if ball jumped too far too quickly
                if dist > self.max_position_jump and frames_since_last < 3:
                    continue  # Skip this detection

            # Score this detection
            score = self._score_ball_detection(center, w, h, conf, target_pos)

            if score > best_score:
                best_score = score
                best_detection = (center, frame_count, w, h, conf)

        # Update tracking state
        if best_detection:
            self.frames_since_ball_detection = 0
            self.predicted_ball_position = None

            # Update velocity
            if len(self.ball_positions) >= 1:
                prev_pos = self.ball_positions[-1]
                dt = frame_count - prev_pos[1]
                if dt > 0:
                    dx = best_detection[0][0] - prev_pos[0][0]
                    dy = best_detection[0][1] - prev_pos[0][1]
                    self.ball_velocity = np.array([dx/dt, dy/dt])

            # Update ball profile periodically
            if len(self.ball_positions) >= 5:
                self._update_ball_profile()

        return best_detection

    def _clean_positions(self, frame_count):
        """Clean old ball and rim positions to prevent memory buildup."""
        # Remove ball positions older than 90 frames
        if len(self.ball_positions) > 0:
            while len(self.ball_positions) > 0 and frame_count - self.ball_positions[0][1] > 90:
                self.ball_positions.pop(0)

        # Remove old rim positions (keep last 40)
        if len(self.rim_positions) > 40:
            self.rim_positions.pop(0)

    def _detect_shot(self, current_frame, frame, ball_detected, rim_detected, attempt_cooldown):
        """
        Detect if a shot occurred and determine the shooting team.
        Uses the original algorithm with linear interpolation for accurate detection.

        Returns:
            tuple: (shot_made, team_id, confidence, location) or None
        """
        # Only execute if hoop and ball positions are known
        if len(self.rim_positions) == 0 or len(self.ball_positions) == 0:
            return None

        # Check if we have recent rim detection and current ball detection
        # Note: We're more lenient than original to avoid missing shots
        if ball_detected and attempt_cooldown == 0:
            # Check if ball is in scoring region using original in_score_region function
            in_region = in_score_region(self.ball_positions, self.rim_positions)

            if in_region:
                # Ball is in scoring region
                if self.ball_entered:
                    self.attempt_time += 1
                else:
                    self.ball_entered = True
                    self.attempt_time = 1
                    current_time = current_frame / self.fps
                    print(f"[DEBUG {current_time:.2f}s] Ball entered scoring region")

                # Use linear interpolation to detect scoring (original algorithm)
                if not self.last_point_in_region:
                    self.last_point_in_region = self.ball_positions[-1]
                    scored = False
                else:
                    current_time = current_frame / self.fps
                    # Debug only around shot 2 (34-36s)
                    debug_mode = 33 < current_time < 37
                    scored = detect_score(self.ball_positions, self.rim_positions, self.last_point_in_region, debug=debug_mode)
                    if scored:
                        print(f"[DEBUG {current_time:.2f}s] detect_score returned TRUE - shot made!")

                if scored:
                    # Shot made!
                    team, confidence, location = self._identify_shooter(frame)

                    # Reset state
                    self.last_point_in_region = None
                    self.ball_entered = False
                    self.attempt_time = 0
                    self.last_shot_frame = current_frame

                    return (True, team, confidence, location, self.shot_cooldown_made_frames)
                else:
                    # Still tracking the ball in region
                    self.last_point_in_region = self.ball_positions[-1]

            else:
                # Ball is NOT in scoring region
                if self.ball_entered:
                    self.attempt_time += 1

                # Check if this is a missed attempt
                if self.attempt_time >= self.attempt_detection_interval:
                    # Shot missed (exited region without scoring)
                    current_time = current_frame / self.fps
                    print(f"[DEBUG {current_time:.2f}s] Ball exited region after {self.attempt_time} frames - shot missed")
                    print(f"[DEBUG {current_time:.2f}s] Total ball positions tracked: {len(self.ball_positions)}")

                    team, confidence, location = self._identify_shooter(frame)

                    # Reset state
                    self.attempt_time = 0
                    self.ball_entered = False
                    self.last_point_in_region = None
                    self.last_shot_frame = current_frame

                    return (False, team, confidence, location, self.shot_cooldown_frames)

        return None

    def _identify_shooter(self, frame):
        """
        Identify which team took the shot using jersey classification.

        Returns:
            tuple: (team_id, confidence, normalized_location)
        """
        # Detect all players in frame using shoot model
        results = self.shoot_model(frame, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0, None

        # Extract person bounding boxes
        person_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if self.config.get('classes_shoot', [])[cls] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append([x1, y1, x2, y2])

        if len(person_boxes) == 0:
            return None, 0.0, None

        # Classify teams for all detected players
        team_assignments = []
        for bbox in person_boxes:
            team, conf = self.jersey_classifier.classify_player(bbox, frame)
            if team is not None:
                team_assignments.append({
                    'bbox': bbox,
                    'team': team,
                    'confidence': conf
                })

        if len(team_assignments) == 0:
            return None, 0.0, None

        # Find shooter based on shoot pose detection or closest to recent shoot poses
        shooter = self._find_most_likely_shooter(team_assignments, frame)

        if shooter is None:
            # Fallback: use player with highest confidence
            shooter = max(team_assignments, key=lambda x: x['confidence'])

        # Calculate normalized location
        bbox = shooter['bbox']
        norm_x = (bbox[0] + bbox[2]) / 2 / self.width
        norm_y = (bbox[1] + bbox[3]) / 2 / self.height

        return shooter['team'], shooter['confidence'], (norm_x, norm_y)

    def _find_most_likely_shooter(self, team_assignments, frame):
        """
        Find the most likely shooter from detected players.
        Uses shoot pose detection if available.
        """
        # Detect shoot poses
        results = self.shoot_model(frame, verbose=False)

        shoot_poses = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if self.config.get('classes_shoot', [])[cls] == 'shoot':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                shoot_poses.append([x1, y1, x2, y2])

        if len(shoot_poses) == 0:
            return None

        # Find player closest to shoot pose
        min_distance = float('inf')
        closest_player = None

        for pose in shoot_poses:
            pose_cx = (pose[0] + pose[2]) / 2
            pose_cy = (pose[1] + pose[3]) / 2

            for player in team_assignments:
                bbox = player['bbox']
                player_cx = (bbox[0] + bbox[2]) / 2
                player_cy = (bbox[1] + bbox[3]) / 2

                distance = np.sqrt((pose_cx - player_cx)**2 + (pose_cy - player_cy)**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_player = player

        return closest_player

    def process_video(self):
        """Main video processing loop."""
        frame_count = 0
        attempt_cooldown = 0

        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}, FPS: {self.fps}")

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = frame_count / self.fps

            # Detect ball and rim
            ball_rim_results = self.ball_rim_model(frame, verbose=False)

            ball_detections = []  # Collect all ball detections for scoring
            rim_detected = False

            if len(ball_rim_results) > 0:
                for box in ball_rim_results[0].boxes:
                    cls = int(box.cls[0])
                    class_name = self.config.get('classes', [])[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    if class_name == 'ball' and conf > 0.4:  # Lowered to 0.4 to catch more candidates
                        ball_detections.append((center, w, h, conf))
                    elif class_name == 'rim' and conf > 0.5 and not rim_detected:
                        rim_detected = True
                        self.rim_positions.append((center, frame_count, w, h, conf))

            # Select best ball detection using multi-factor scoring
            best_ball = self._select_best_ball(ball_detections, frame_count)
            ball_detected = best_ball is not None

            if best_ball:
                self.ball_positions.append(best_ball)

            # Clean ball and rim positions (remove old positions)
            self._clean_positions(frame_count)

            # Detect shots using original algorithm
            shot_result = self._detect_shot(frame_count, frame, ball_detected, rim_detected, attempt_cooldown)

            if shot_result is not None:
                made, team, confidence, location, new_cooldown = shot_result

                # Record shot
                if team is not None:
                    self.score_counter.record_shot(
                        timestamp=current_time,
                        team=team,
                        made=made,
                        location=location,
                        confidence=confidence
                    )

                # Trigger callback
                self.on_shot_detected(current_time, team, made, location, confidence)

                # Set cooldown
                attempt_cooldown = new_cooldown

            # Decrement cooldown
            if attempt_cooldown > 0:
                attempt_cooldown -= 1

            # Detect and classify players for visualization
            player_teams = None
            if self.show_video or self.save_video:
                results = self.shoot_model(frame, verbose=False)
                player_boxes = []
                if len(results) > 0:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        if self.config.get('classes_shoot', [])[cls] == 'person':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            player_boxes.append([x1, y1, x2, y2])

                # Classify teams
                if len(player_boxes) > 0:
                    player_teams = []
                    for bbox in player_boxes:
                        team, conf = self.jersey_classifier.classify_player(bbox, frame)
                        if team is not None:
                            player_teams.append({
                                'bbox': bbox,
                                'team': team,
                                'confidence': conf
                            })

            # Prepare ball/rim position for visualization (convert from new to old format)
            ball_pos_vis = None
            rim_pos_vis = None
            if len(self.ball_positions) > 0:
                center, _, w, h, _ = self.ball_positions[-1]
                ball_pos_vis = [center[0] - w//2, center[1] - h//2, w, h]
            if len(self.rim_positions) > 0:
                center, _, w, h, _ = self.rim_positions[-1]
                rim_pos_vis = [center[0] - w//2, center[1] - h//2, w, h]

            # Create visualization
            if self.show_video or self.save_video:
                vis_frame = self._visualize_frame(frame, ball_pos_vis, rim_pos_vis, player_teams)

                # Display if requested
                if self.show_video:
                    cv2.imshow('Integrated Shot & Team Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Write to output video
                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(vis_frame)

            # Update jersey classifier (for color reinitialization)
            if frame_count % self.jersey_classifier.reinit_interval == 0:
                # Detect players for color initialization
                results = self.shoot_model(frame, verbose=False)
                player_boxes = []
                if len(results) > 0:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        if self.config.get('classes_shoot', [])[cls] == 'person':
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            player_boxes.append([x1, y1, x2, y2])

                if len(player_boxes) > 0:
                    self.jersey_classifier.initialize_team_colors(player_boxes, frame, frame_count)

                    # Store team colors in score counter
                    if hasattr(self.jersey_classifier, 'team_colors_rgb') and self.jersey_classifier.team_colors_rgb is not None:
                        for team_id in [0, 1]:
                            if team_id in self.jersey_classifier.team_colors_rgb:
                                rgb = self.jersey_classifier.team_colors_rgb[team_id]
                                # Convert numpy types to regular ints
                                rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                                self.score_counter.set_team_color(team_id, rgb_tuple)

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{self.total_frames} frames "
                      f"({frame_count/self.total_frames*100:.1f}%)")

        # Cleanup
        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Output video saved to: {self.output_path}")
        if self.show_video:
            cv2.destroyAllWindows()

        print("\nProcessing complete!")
        self.score_counter.print_report()
        self.on_complete()

    def _visualize_frame(self, frame, ball_pos, rim_pos, player_teams=None):
        """Draw visualizations on frame."""
        vis = frame.copy()

        # Draw players with team colors
        if player_teams is not None:
            for player_info in player_teams:
                bbox = player_info['bbox']
                team = player_info['team']
                conf = player_info['confidence']

                # Team colors: Team 0 = Red, Team 1 = Cyan
                color = (0, 0, 255) if team == 0 else (255, 255, 0)
                thickness = int(2 + conf * 2)  # Thickness based on confidence

                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

                # Team label
                label = f"T{team} ({conf:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis, (x1, y1-label_size[1]-5), (x1+label_size[0], y1), color, -1)
                cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)

        # Draw ball
        if ball_pos is not None:
            x, y, w, h = ball_pos
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, 'BALL', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
        elif self.predicted_ball_position is not None:
            # Draw predicted ball position during occlusion
            px, py = self.predicted_ball_position
            cv2.circle(vis, (px, py), 10, (0, 165, 255), 2)  # Orange circle for prediction
            cv2.putText(vis, 'PREDICTED', (px, py-15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (0, 165, 255), 1)

        # Draw ball trajectory (last 20 positions)
        if len(self.ball_positions) > 1:
            points = [p[0] for p in self.ball_positions[-20:]]
            for i in range(1, len(points)):
                # Fade trail based on age
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(vis, points[i-1], points[i], (100, 100, 100), thickness)

        # Draw rim
        if rim_pos is not None:
            x, y, w, h = rim_pos
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis, 'RIM', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 0, 0), 2)

            # Draw scoring region
            region = self._get_scoring_region(rim_pos)
            if region:
                cv2.rectangle(vis,
                            (int(region['x_min']), int(region['y_min'])),
                            (int(region['x_max']), int(region['y_max'])),
                            (0, 255, 255), 1)

        # Draw statistics overlay
        stats = self.score_counter.get_stats()
        y_offset = 30
        cv2.putText(vis, f"Team 0: {sum(stats['team_0']['makes'])}/{sum(stats['team_0']['attempts'])}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Team 1: {sum(stats['team_1']['makes'])}/{sum(stats['team_1']['attempts'])}",
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw occlusion status
        if self.frames_since_ball_detection > 0:
            cv2.putText(vis, f"Occlusion: {self.frames_since_ball_detection} frames",
                       (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        return vis

    def start(self):
        """Start processing in background thread."""
        self.processing_thread = Thread(target=self.process_video)
        self.processing_thread.start()

    def stop(self):
        """Stop processing."""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()

    def get_statistics(self):
        """Get current statistics."""
        return self.score_counter.get_stats()


def main():
    """Example usage."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Integrated Shot & Team Detection')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    parser.add_argument('--save', action='store_true', help='Save annotated video output')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--json', help='Output JSON file path (optional)')
    parser.add_argument('--team0', help='Team 0 color (optional)')
    parser.add_argument('--team1', help='Team 1 color (optional)')
    parser.add_argument('--reinit', type=int, default=300,
                       help='Frames between color reinitialization')

    args = parser.parse_args()

    # Default JSON output path
    json_output = args.json
    if json_output is None:
        json_output = Path(args.video).stem + '_shot_stats.json'

    detector = IntegratedShotTeamDetector(
        video_path=args.video,
        config_path=args.config,
        show_video=args.show,
        save_video=args.save,
        output_path=args.output,
        reinit_interval=args.reinit,
        team0_color=args.team0,
        team1_color=args.team1
    )

    try:
        detector.process_video()

        # Save statistics to JSON
        stats = detector.get_statistics()

        # Add team color names to output
        output_data = {
            'video_file': args.video,
            'team_0': {
                'color': detector.score_counter._rgb_to_color_name(
                    detector.score_counter.team_colors[0]
                ),
                'rgb': detector.score_counter.team_colors[0],
                'makes': stats['team_0']['makes'],
                'attempts': stats['team_0']['attempts'],
                'fg_percentage': stats['team_0']['fg_percentage']
            },
            'team_1': {
                'color': detector.score_counter._rgb_to_color_name(
                    detector.score_counter.team_colors[1]
                ),
                'rgb': detector.score_counter.team_colors[1],
                'makes': stats['team_1']['makes'],
                'attempts': stats['team_1']['attempts'],
                'fg_percentage': stats['team_1']['fg_percentage']
            },
            'shot_log': stats['shot_log']
        }

        with open(json_output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nStatistics saved to: {json_output}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        detector.stop()


if __name__ == '__main__':
    main()
