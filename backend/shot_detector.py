import numpy as np
from ultralytics import YOLO
from utils import in_score_region

class ShotDetector:
    """
    Handles shot detection using a YOLO model for ball, rim, person, and
    shoot-pose tracking. Detects when a shot attempt occurs and provides
    tracking data for downstream score detection and team identification.
    """
    def __init__(self, config):
        self.config = config

        # Load YOLO model for ball/rim/person/shoot detection
        print("Loading shot detection YOLO model...")
        self.model = YOLO(self.config['weights_path_shoot'])

        # Shot detection state
        self.ball_positions = []  # List of tuples: ((x, y), frame_count, w, h, conf)
        self.rim_positions = []   # List of tuples: ((x, y), frame_count, w, h, conf)
        self.person_boxes = []    # List of [x1, y1, x2, y2] from latest frame

        self.last_shot_frame = 0
        self.ball_entered = False
        self.last_point_in_region = None
        self.attempt_time = 0

        # Default FPS-dependent parameters (will be updated via set_fps)
        self.fps = 30.0
        self._update_fps_params()

        # Enhanced ball tracking for occlusion handling
        self.ball_velocity = np.array([0.0, 0.0])
        self.frames_since_ball_detection = 0
        self.predicted_ball_position = None
        self.original_ball_size = None  # (avg_w, avg_h) of tracked ball
        self.original_ball_conf = None  # avg confidence of tracked ball
        self.max_position_jump = 200  # Maximum pixels ball can move between frames

    def set_fps(self, fps):
        """Update FPS and dependent parameters."""
        self.fps = fps
        self._update_fps_params()

    def _update_fps_params(self):
        self.shot_cooldown_frames = int(self.fps * 2.5)  # 2.5 seconds between shot attempts
        self.attempt_detection_interval = int(self.fps * 1.5)  # Increased from 0.3 to 1.5 seconds
        self.max_occlusion_frames = int(self.fps * 0.5)  # Allow 0.5 seconds of occlusion

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

    def process_frame(self, frame, frame_count, device='cpu'):
        """
        Process a single frame to detect ball, rim, person, and shoot pose.
        Updates internal state (ball_positions, rim_positions, person_boxes).

        Returns:
            tuple: (ball_detected, rim_detected, shoot_detected) boolean flags
        """
        results = self.model(frame, verbose=False, device=device)

        ball_detections = []
        rim_detected = False
        shoot_detected = False
        self.person_boxes = []

        if len(results) > 0:
            names = self.model.names

            for box in results[0].boxes:
                cls = int(box.cls[0])
                class_name = names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])

                center = (int(x1 + w / 2), int(y1 + h / 2))

                if class_name == 'ball' and conf > 0.4:
                    ball_detections.append((center, w, h, conf))
                elif class_name in ['rim', 'hoop'] and conf > 0.5 and not rim_detected:
                    rim_detected = True
                    self.rim_positions.append((center, frame_count, w, h, conf))
                elif class_name == 'person' and conf > 0.5:
                    self.person_boxes.append([x1, y1, x2, y2])
                elif class_name == 'shoot' and conf > 0.5:
                    shoot_detected = True

        # Select best ball detection using multi-factor scoring
        best_ball = self._select_best_ball(ball_detections, frame_count)
        ball_detected = best_ball is not None

        if best_ball:
            self.ball_positions.append(best_ball)

        # Clean ball and rim positions (remove old positions)
        self._clean_positions(frame_count)

        return ball_detected, rim_detected, shoot_detected

    def detect_shot(self, frame_count, ball_detected, attempt_cooldown, shoot_detected=False):
        """
        Detect if a shot attempt occurred based on ball trajectory near the rim
        and shoot pose detection.

        Args:
            frame_count: Current frame number
            ball_detected: Whether ball was detected this frame
            attempt_cooldown: Current cooldown counter (0 = ready to detect)
            shoot_detected: Whether YOLO 'shoot' class was detected this frame

        Returns:
            dict with shot attempt data if a shot is detected, None otherwise.
        """
        # Only execute if hoop and ball positions are known
        if len(self.rim_positions) == 0 or len(self.ball_positions) == 0:
            return None

        if (ball_detected or shoot_detected) and attempt_cooldown == 0:
            # Check if ball is in scoring region
            in_region = in_score_region(self.ball_positions, self.rim_positions)

            if in_region:
                # Ball is in scoring region
                if not self.ball_entered:
                    self.ball_entered = True
                    self.attempt_time = 1
                    self.last_point_in_region = self.ball_positions[-1]
                    current_time = frame_count / self.fps
                    print(f"[DEBUG {current_time:.2f}s] Ball entered scoring region (shoot_detected={shoot_detected})")
                else:
                    self.attempt_time += 1
                    self.last_point_in_region = self.ball_positions[-1]

                # If shoot pose detected while ball is in scoring region, trigger immediately
                if shoot_detected:
                    current_time = frame_count / self.fps
                    print(f"[DEBUG {current_time:.2f}s] Shoot pose detected with ball in region - shot attempt triggered")

                    shot_data = self._build_shot_data(frame_count)
                    self._reset_attempt_state(frame_count)
                    return shot_data

            else:
                # Ball is NOT in scoring region
                if self.ball_entered:
                    self.attempt_time += 1

                # Shot attempt complete: ball entered region and has now exited
                if self.attempt_time >= self.attempt_detection_interval:
                    current_time = frame_count / self.fps
                    print(f"[DEBUG {current_time:.2f}s] Ball exited region after {self.attempt_time} frames - shot attempt detected")

                    shot_data = self._build_shot_data(frame_count)
                    self._reset_attempt_state(frame_count)
                    return shot_data

        return None

    def _build_shot_data(self, frame_count):
        """Build the shot attempt data dict returned by detect_shot."""
        return {
            'ball_positions': list(self.ball_positions),
            'rim_positions': list(self.rim_positions),
            'last_point_in_region': self.last_point_in_region,
            'frame_count': frame_count,
        }

    def _reset_attempt_state(self, frame_count):
        """Reset internal state after a shot attempt is detected."""
        self.attempt_time = 0
        self.ball_entered = False
        self.last_point_in_region = None
        self.last_shot_frame = frame_count
