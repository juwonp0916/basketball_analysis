# TODO: Review changes needed here
from PIL import Image
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import os, shutil
import yaml
from pathlib import Path
from datetime import timedelta, datetime
import time
from enum import Enum
import threading
from queue import Queue, Empty
from copy import copy, deepcopy

from utils import (
    clean_hoop_pos, 
    clean_ball_pos, 
    detect_score, 
    in_score_region,
    get_time_string
)

from score_counter import (
    ScoreCounter,
    MatchScoreCounter
)


from logger import (
    INFO,
    SOCKET,
    Logger
)

# get environment variables
# Use absolute path relative to this file's location
import os as _os
_config_dir = _os.path.dirname(__file__)
_config_path = _os.path.join(_config_dir, 'config.yaml')
env = yaml.load(open(_config_path, 'r'), Loader=yaml.SafeLoader)

# Resolve relative paths in config to be relative to config file location
if not _os.path.isabs(env['weights_path']):
    env['weights_path'] = _os.path.join(_config_dir, env['weights_path'])

print("Environment variables: ", env)

logger = Logger([
    INFO
])

class ShotDetector:
    def __init__(self,
                video_path,             # Video path for processing
                on_detect,              # on_detect(timestamp, success, team_id, shot_location, court_position) -> Callback to MatchHandler for when a shot is detected
                on_complete,            # on_complete(team_id) -> Callback to MatchHandler for when video finishes processing
                show_vid=False,         # Show CV2 window while processing, for debugging, will significantly slow down program
                video_id=None,          # 1 or 2
                calibration_points=None, # List of 4 or 6 (x,y) calibration points for shot localization
                enable_localization=False, # Enable shot localization with homography
                calibration_mode='6-point', # '4-point' (paint box) or '6-point' (full baseline)
                **kwargs):
        
        #TODO: initialize with team_id, updated based on switch timestamp
        self.video_path = video_path
        self.calibration_points = calibration_points
        self.calibration_mode = calibration_mode
        self.model = YOLO(env['weights_path'], verbose=False)
        self.class_names = env['classes']
        self.colors = [(0, 255, 0), (255, 255, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]
        self.colors_shoot = [(0, 255, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]
        self.on_detect = on_detect
        self.on_complete = on_complete
        self.show_vid = show_vid

        # for debug
        self.video_name = video_path.split("/")[-1].replace(".mp4", "")
        self.output_true_shot = f"true_shot_frames_{self.video_name}"
        self.output_all_shot = f"all_shot_frames_{self.video_name}"
        os.makedirs(self.output_true_shot, exist_ok=True)
        os.makedirs(self.output_all_shot, exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        logger.log(INFO, f"FPS: {self.frame_rate}")

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.shoot_pos = []  # array of dictionaries with full shot data (frame, timestamp, shoot_confidence, shooter_position)
        self.pending_shot_group = []  # accumulator for current shot group (for deduplication)
        self.last_group_processed_frame = -1  # track last processed shot group
        self.last_shot_detection_frame = -1  # track last frame where shoot class was detected
        self.recent_shots = []  # track recently detected shots for cross-method deduplication [(frame, timestamp, shooter_position), ...]
        self.frame_track = [] # array of tuples (det_frame, timestamp)
        self.num_frames_to_track = 2 * self.frame_rate # 2 seconds before
        self.frame_count = 0
        self.frame = None

        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))

        logger.log(INFO, f"Input Resolution: {self.width} X {self.height}")

        # Initialize shot localization
        self.enable_localization = enable_localization
        self.shot_localizer = None
        self.shots_data = []  # Accumulate shot data for JSON export

        # Debug frame saving
        self.save_debug_frames = kwargs.get('save_debug_frames', False)
        if self.save_debug_frames:
            self.debug_frame_dir = 'debug_frames'
            os.makedirs(self.debug_frame_dir, exist_ok=True)

        if enable_localization:
            if not calibration_points:
                raise ValueError("Calibration points required when enable_localization=True")

            # Validate calibration before proceeding
            self._validate_calibration_dimensions(
                calibration_points,
                self.width,
                self.height
            )

            from localization import ShotLocalizer

            # Resolve court image path relative to this file's location
            court_img_path = _os.path.join(_config_dir, 'court_img_halfcourt.png')

            self.shot_localizer = ShotLocalizer(
                calibration_points=calibration_points,
                image_dimensions=(self.width, self.height),
                court_img_path=court_img_path,
                enable_visualization=True,
                calibration_mode=self.calibration_mode
            )
            logger.log(INFO, f"✓ Shot localization enabled with {self.calibration_mode} calibration")

        self.makes = 0
        self.attempts = 0
        self.attempt_cooldown = 0
        self.attempt_time = 0


        self.video_id = video_id

        # For marking if the ball / rim / shoot have been detected in the current frame
        self.ball_detected = False
        self.rim_detected = False
        self.shoot_detected = False
        self.should_detect_shot = False
        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.last_point_in_region = None
        
        self.screen_shot_count = 0
        self.screenshot = env['screenshot']
        self.screen_shot_moment = False
        self.screen_shot_path = env['screenshot_path']
        self.save = env['save_video']
        Path(self.screen_shot_path).mkdir(parents=True, exist_ok=True)

        self.inference_width = ((self.width + 31) // 32) * 32
        self.inference_height = ((self.height + 31) // 32) * 32

        # if self.width > self.inference_width and self.height > self.inference_height:
        #     self.inference_width = self.width
        #     self.inference_height = self.height

        logger.log(INFO, f"Inference Resolution: {self.inference_width} X {self.inference_height}")

        self.attempt_cooldown = 0
        self.timestamp = None
        self.ball_entered = False
        self.rim_last_detected = -1

        self.MISS_ATTEMPT_COOLDOWN = int(self.frame_rate * 2.5)
        self.MADE_ATTEMPT_COOLDOWN = int(self.frame_rate * 3)
        self.ATTEMPT_DETECTION_INTERVAL = int(self.frame_rate * 0.3)

        # Deduplication constants
        from constants import DEDUPLICATION_FRAME_THRESHOLD, DEDUPLICATION_POSITION_THRESHOLD, DEDUPLICATION_SAFETY_COOLDOWN_SEC
        self.DEDUPLICATION_FRAME_THRESHOLD = DEDUPLICATION_FRAME_THRESHOLD
        self.DEDUPLICATION_POSITION_THRESHOLD = DEDUPLICATION_POSITION_THRESHOLD
        self.DEDUPLICATION_SAFETY_COOLDOWN = int(self.frame_rate * DEDUPLICATION_SAFETY_COOLDOWN_SEC)

        self.output_width = env['output_width']
        self.output_height = env['output_height']

        if self.save:
            os.makedirs(env['output_path'], exist_ok=True)
            output_name = env['output_path'] + '/' + video_path.split("/")[-1].split('.')[0] + str(datetime.now()) 
            output_name = output_name.replace(':','-').replace('.','-') + ".mp4"
            logger.log(INFO, f"Saving results to: {output_name}")
            self.out = cv2.VideoWriter(output_name,  cv2.VideoWriter_fourcc(*'mp4v'), self.frame_rate, (self.output_width, self.output_height))
        
        # Threading components
        self.detection_queue = Queue()
        # self.detection_thread = None
        self.detection_thread_active = True
        # self.detection_lock = threading.Lock()
        
        # Start detection worker thread
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        start_time = time.time()
        self.run()
        
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        logger.log(INFO, f"Total processing time: {minutes:02d}:{seconds:02d}")

    def _validate_calibration_dimensions(self, calibration_points, video_width, video_height):
        """
        Validate calibration points are within video bounds and well-spaced.
        Logs warnings for potential calibration issues.

        Args:
            calibration_points: Either comma-separated string or list of [x,y] points
            video_width: Video frame width
            video_height: Video frame height
        """
        # Parse points
        if isinstance(calibration_points, str):
            coords = list(map(float, calibration_points.split(',')))
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        else:
            points = calibration_points

        # Check each point
        issues_found = False
        for i, point in enumerate(points):
            x, y = point[0], point[1]

            # Check if point is outside video bounds
            if x < 0 or x > video_width or y < 0 or y > video_height:
                logger.log(INFO,
                    f"WARNING: Calibration point {i+1} ({x}, {y}) is OUTSIDE "
                    f"video bounds (0-{video_width}, 0-{video_height})")
                issues_found = True

            # Check if points are suspiciously close
            if i > 0:
                prev_x, prev_y = points[i-1][0], points[i-1][1]
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if dist < 10:
                    logger.log(INFO,
                        f"WARNING: Points {i} and {i+1} are very close ({dist:.1f}px) - "
                        f"calibration may be inaccurate")
                    issues_found = True

        if issues_found:
            logger.log(INFO, "⚠️  Calibration validation found issues. Consider recalibrating.")
            logger.log(INFO, f"   Video dimensions: {video_width}x{video_height}")
            logger.log(INFO, f"   Calibration points: {points}")
        else:
            logger.log(INFO, "✓ Calibration validation passed")

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                logger.log(INFO, "Processing complete")

                # Process any remaining pending shot group before finishing
                if len(self.pending_shot_group) > 0:
                    logger.log(INFO, f"Processing final shot group ({len(self.pending_shot_group)} detections)")
                    representative_shot = self._deduplicate_shot_group(self.pending_shot_group)

                    if representative_shot:
                        # Create detection task
                        detection_task = {
                            'shot_data': representative_shot,
                            'frame_track': deepcopy(self.frame_track),
                            'timestamp': representative_shot['timestamp'],
                            'is_scored': False,  # Assume miss for final shot
                            'video_id': self.video_id
                        }
                        self.detection_queue.put(detection_task)
                        self.attempts += 1
                        self.pending_shot_group = []

                # Wait for detection queue to be empty before stopping
                logger.log(INFO, "Waiting for shot detection queue to finish...")
                self.detection_queue.join()

                # Signal detection thread to stop
                self.detection_thread_active = False
                break

            self.timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            # resize to match - force 1280 and 720 for better model results
            det_frame = cv2.resize(self.frame, (self.inference_width, self.inference_height))

            results = self.model(det_frame, stream=True, verbose=False, imgsz=self.inference_width, device=env.get('device', 0))

            for r in results:

                #TODO: better way to get max conf boxes only
                boxes = sorted([(box.xyxy[0], box.conf, box.cls) for box in r.boxes], key=lambda x: -x[1])
                #sort and get only top prediction for ball / hoop

                # Reset detection variables
                self.ball_detected, self.rim_detected = False, False

                for box in boxes:
                    # Only one ball / rim should be detected per frame
                    if self.ball_detected and self.rim_detected:
                        break
                    
                    # Bounding box
                    x1, y1, x2, y2 = box[0]

                    # Scale back up to original dimensions
                    x1, y1, x2, y2 = int(x1 * self.width/self.inference_width), int(y1 * self.height/self.inference_height), int(x2 * self.width/self.inference_width), int(y2* self.height/self.inference_height)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box[1] * 100)) / 100

                    # Class Name
                    cls = int(box[2])
                    current_class = self.class_names[cls]
                    # print(cls, current_class)

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    if (conf > 0.7 and current_class == 'rim' and not self.rim_detected) or (conf > 0.7 and current_class == 'ball' and not self.ball_detected):

                        if self.show_vid or self.save or self.screenshot:
                            self.draw_bounding_box(current_class, conf, cls, x1, y1, x2, y2)
                        
                        if current_class == 'rim':
                            self.rim_detected = True
                            self.rim_last_detected = self.frame_count
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        elif current_class == 'ball':
                            self.ball_detected = True
                            self.ball_pos.append((center, self.frame_count, w, h, conf))

            # Run shoot detection model every 3 frames for performance
            self.shoot_detected = False
            if self.frame_count % 3 == 0:
                results_shoot = self.model(det_frame, stream=True, verbose=False, imgsz=self.inference_width, device=env.get('device', 0))

                for r in results_shoot:
                    boxes_shoot = sorted([(box.xyxy[0], box.conf, box.cls) for box in r.boxes], key=lambda x: -x[1])

                    for box in boxes_shoot:
                        conf = float(box[1])
                        cls = int(box[2])
                        current_class = self.class_names[cls]

                        # Detect "shoot" class with lower confidence threshold
                        if current_class == 'shoot' and conf > 0.3 and not self.shoot_detected:
                            self.shoot_detected = True

                            # Scale shoot box to original dimensions
                            x1_shoot = int(box[0][0] * self.width / self.inference_width)
                            y1_shoot = int(box[0][1] * self.height / self.inference_height)
                            x2_shoot = int(box[0][2] * self.width / self.inference_width)
                            y2_shoot = int(box[0][3] * self.height / self.inference_height)

                            # Find shooter position using helper method
                            shooter_position = self._find_shooter_position(
                                boxes_shoot,
                                (x1_shoot, y1_shoot, x2_shoot, y2_shoot)
                            )

                            # Store full detection data as dictionary
                            shot_data = {
                                'frame': self.frame_count,
                                'timestamp': self.timestamp,
                                'shoot_confidence': conf,
                                'shooter_position': shooter_position
                            }
                            self.shoot_pos.append(shot_data)
                            self.pending_shot_group.append(shot_data)
                            self.last_shot_detection_frame = self.frame_count  # Track last detection

                            logger.log(INFO, f"[SHOOT CLASS DETECT] Frame {self.frame_count} ({get_time_string(self.timestamp)}) conf: {conf:.2f}")
                            if shooter_position:
                                logger.log(INFO, f"  Shooter position: ({shooter_position['x']:.1f}, {shooter_position['y']:.1f})")
                            else:
                                logger.log(INFO, f"  NO shooter position found (no person with >70% IoU)")

                            # Save debug frame if enabled
                            if self.save_debug_frames:
                                self._save_debug_frame(box, det_frame, current_class, conf)

                            break  # Only need one shoot detection per frame

            # Store frame boxes info instead of frame
            if len(self.frame_track) >= self.num_frames_to_track:
                self.frame_track.pop(0)
            self.frame_track.append((det_frame,self.frame_count, self.timestamp, self.frame))

            self.clean_motion()
            self.score_detection()
            
            self.frame_count += 1

            if self.attempt_cooldown > 0:
                self.attempt_cooldown -= 1

            if self.show_vid or self.save or self.screenshot:
                self.draw_overlay()
                # self.draw_overlay()

                if self.show_vid:
                    cv2.imshow('Frame', self.frame)
                    # Close if 'q' is clicked
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                        break

                if self.screen_shot_moment:
                    cv2.imwrite(f"{self.screen_shot_path}/{self.screen_shot_count}.png", self.frame)
                    self.screen_shot_moment = False
                    self.screen_shot_count += 1

                if self.save:
                    self.out.write(cv2.resize(self.frame, (env['output_width'], env['output_height'])))
        
        # Wait for detection thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)

        # Save localization results (JSON and shot chart)
        if self.enable_localization:
            self.save_localization_results(self.video_path, self.calibration_points)

        self.on_complete()

        self.cap.release()
        
        if self.save:
            self.out.release()
        if self.show_vid:
            cv2.destroyAllWindows()

    def _save_debug_frame(self, box, frame, class_name, conf):
        """Save debug frame when shoot class is detected"""
        # Create a copy of the frame to draw on
        debug_frame = frame.copy()

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box[0])

        # Draw bounding box
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add label with confidence
        label = f"{class_name}: {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(debug_frame, (x1, y1 - 30), (x1 + text_w + 10, y1), (0, 255, 0), -1)
        cv2.putText(debug_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Add timestamp and frame info at top
        timestring = str(timedelta(milliseconds=self.timestamp)).split('.')[0]
        info_text = f"Frame: {self.frame_count} | Time: {timestring}"
        cv2.putText(debug_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save frame
        filename = f"shoot_frame_{self.frame_count:06d}_{timestring.replace(':', '-')}_conf{conf:.2f}.jpg"
        filepath = os.path.join(self.debug_frame_dir, filename)
        cv2.imwrite(filepath, debug_frame)
        logger.log(INFO, f"Debug frame saved: {filepath}")

    def _find_shooter_position(self, boxes_shoot, shoot_box_coords):
        """
        Find shooter position from person boxes based on overlap with shoot box

        Args:
            boxes_shoot: List of all boxes from shoot model (sorted by confidence)
            shoot_box_coords: Tuple (x1, y1, x2, y2) of shoot box in original dimensions

        Returns:
            dict: {'x': center_x, 'y': bottom_y} or None if no valid person found
        """
        x1_shoot, y1_shoot, x2_shoot, y2_shoot = shoot_box_coords
        shoot_area = (x2_shoot - x1_shoot) * (y2_shoot - y1_shoot)

        best_person = None
        best_overlap = 0.0

        # Iterate through all boxes to find person boxes
        for box_data in boxes_shoot:
            cls = int(box_data[2])
            class_name = self.class_names[cls]

            # Only consider person boxes
            if class_name != 'person':
                continue

            # Get person box coordinates (already in original dimensions)
            x1_person, y1_person, x2_person, y2_person = box_data[0]
            x1_person = int(x1_person * self.width / self.inference_width)
            y1_person = int(y1_person * self.height / self.inference_height)
            x2_person = int(x2_person * self.width / self.inference_width)
            y2_person = int(y2_person * self.height / self.inference_height)

            # Calculate intersection area
            x_left = max(x1_shoot, x1_person)
            y_top = max(y1_shoot, y1_person)
            x_right = min(x2_shoot, x2_person)
            y_bottom = min(y2_shoot, y2_person)

            if x_right < x_left or y_bottom < y_top:
                continue  # No overlap

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            overlap_ratio = intersection_area / shoot_area if shoot_area > 0 else 0

            # Check if this person has better overlap (threshold: 70%)
            if overlap_ratio >= 0.7 and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                # Calculate shooter position (center_x, bottom_y)
                center_x = (x1_person + x2_person) / 2
                bottom_y = y2_person
                best_person = {'x': center_x, 'y': bottom_y}

        return best_person

    # Function to draw bounding box for ball and rim
    def draw_bounding_box(self, current_class, conf, cls, x1, y1, x2, y2):
                        
        label = f"{current_class}: {conf}"
        color = self.colors[cls]

        self.frame = cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        self.frame = cv2.rectangle(self.frame, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        self.frame = cv2.putText(self.frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Function to draw overlay elements, mainly for debugging purposes
    def draw_overlay(self):
        if self.hoop_pos and self.rim_detected:
            # draw score-region
            x1 = self.hoop_pos[-1][0][0] - 2  * self.hoop_pos[-1][2]
            x2 = self.hoop_pos[-1][0][0] + 2  * self.hoop_pos[-1][2]
            y1 = self.hoop_pos[-1][0][1] - 3.5  * self.hoop_pos[-1][3]
            y2 = self.hoop_pos[-1][0][1] + 0.9 * self.hoop_pos[-1][3]

            pts = np.array([[x1, y1], [x2,y1], [x2, y2], [x1, y2]], np.int32)

            pts = pts.reshape((-1, 1, 2))

            self.frame = cv2.polylines(self.frame, [pts], True, (255, 0, 255), 3)

            #draw hoop-line
            hoop_y = self.hoop_pos[-1][0][1]
            x1 = self.hoop_pos[-1][0][0] - 0.5 * self.hoop_pos[-1][2]
            x2 = self.hoop_pos[-1][0][0] + 0.5 * self.hoop_pos[-1][2]

            pts = np.array([[x1, hoop_y], [x2, hoop_y]], np.int32)

            pts = pts.reshape((-1, 1, 2))

            self.frame = cv2.polylines(self.frame, [pts], True, (0, 255, 255), 2)

        #draw timestamp
        timestring = str(timedelta(milliseconds=self.timestamp)).split('.')[0]
        cv2.putText(self.frame, timestring, (int(self.width*0.9), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(self.frame, timestring, (int(self.width*0.9), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)

        #display ball trajectory
        for i in range(0, len(self.ball_pos)):
            color = (0, 0, 255) if i == len(self.ball_pos)-1 else (100, 100, 100, 0.5)
            thickness = 5 if i == len(self.ball_pos)-1 else 2
                
            cv2.circle(self.frame, self.ball_pos[i][0], 2, color, thickness)

        self.display_score()


    # Function to clean likely erroneous detections
    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        
        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

    # Function to handle scoring moment detection logic
    def score_detection(self):
        # STREAMING DEDUPLICATION: Process accumulated shot group when enough time has passed
        # Wait for DEDUPLICATION_FRAME_THRESHOLD frames AFTER the last detection to ensure group is complete
        frames_since_last_detection = self.frame_count - self.last_shot_detection_frame

        if (self.attempt_cooldown == 0 and
            len(self.pending_shot_group) > 0 and
            frames_since_last_detection > self.DEDUPLICATION_FRAME_THRESHOLD):
            # Deduplicate the accumulated detections
            representative_shot = self._deduplicate_shot_group(self.pending_shot_group)

            if representative_shot:
                # QUALITY FILTER: Reject shots without valid shooter position
                # These are likely false positives (e.g., random motion detected as "shoot" class)
                if not representative_shot.get('shooter_position'):
                    logger.log(INFO, f"[SHOOT CLASS] Rejecting shot at frame {representative_shot['frame']} "
                                   f"({get_time_string(representative_shot['timestamp'])}) - "
                                   f"NO valid shooter position (likely false positive)")
                    self.pending_shot_group = []
                    return

                # STALE DETECTION FILTER: Reject shots that are too old compared to current frame
                # This prevents processing delayed detections that happened before more recent shots
                frames_old = self.frame_count - representative_shot['frame']
                if frames_old > self.DEDUPLICATION_FRAME_THRESHOLD * 1.5:  # 67.5 frames ~2.25 seconds
                    logger.log(INFO, f"[SHOOT CLASS] Rejecting STALE shot at frame {representative_shot['frame']} "
                                   f"({get_time_string(representative_shot['timestamp'])}) - "
                                   f"{frames_old} frames old (current frame: {self.frame_count})")
                    self.pending_shot_group = []
                    return

                # CROSS-METHOD DEDUPLICATION: Check if this shot is too similar to recently detected shots
                # This prevents shoot class false positives from being added when the same shot
                # was already detected by the fallback (score region) method
                logger.log(INFO, f"[DEDUP CHECK] Checking shot at frame {representative_shot['frame']} against {len(self.recent_shots)} recent shot(s)")
                should_skip = self._is_duplicate_of_recent_shot(representative_shot)
                if should_skip:
                    logger.log(INFO, f"[SHOOT CLASS] Skipping duplicate shot at frame {representative_shot['frame']} "
                                   f"(too similar to recently detected shot)")
                    self.pending_shot_group = []
                    return
                logger.log(INFO, f"[SHOOT CLASS] Processing shot group at frame {representative_shot['frame']} "
                                f"(timestamp: {get_time_string(representative_shot['timestamp'])}, "
                                f"position: ({representative_shot['shooter_position']['x']:.1f}, {representative_shot['shooter_position']['y']:.1f}))")

                # Determine make/miss by checking score region
                is_scored = False

                # Check if ball is currently in score region (for immediate scoring)
                if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
                    if in_score_region(self.ball_pos, self.hoop_pos):
                        # Ball is in score region - check if it's going down (made shot)
                        if self.last_point_in_region:
                            is_scored = detect_score(self.ball_pos, self.hoop_pos, self.last_point_in_region)
                            self.last_point_in_region = self.ball_pos[-1]
                        else:
                            self.last_point_in_region = self.ball_pos[-1]

                # Create detection task with captured shooter position
                detection_task = {
                    'shot_data': representative_shot,
                    'frame_track': deepcopy(self.frame_track),
                    'timestamp': representative_shot['timestamp'],
                    'is_scored': is_scored,
                    'video_id': self.video_id
                }
                self.detection_queue.put(detection_task)

                # Update stats and apply cooldown
                if is_scored:
                    self.makes += 1
                    self.attempts += 1
                    logger.log(INFO, f"[{get_time_string(detection_task['timestamp'])}] {'Shot made'.ljust(13)}")
                    self.overlay_color = (0, 255, 0)
                    self.attempt_cooldown = self.MADE_ATTEMPT_COOLDOWN
                else:
                    self.attempts += 1
                    logger.log(INFO, f"[{get_time_string(detection_task['timestamp'])}] {'Attempt made'.ljust(13)}")
                    self.overlay_color = (0, 0, 255)
                    self.attempt_cooldown = self.MISS_ATTEMPT_COOLDOWN

                # Apply safety cooldown (minimum 1 second between shot groups)
                self.attempt_cooldown = max(
                    self.attempt_cooldown,
                    self.DEDUPLICATION_SAFETY_COOLDOWN
                )

                # Visual feedback
                self.fade_counter = self.fade_frames
                self.screen_shot_moment = True
                self.last_point_in_region = None

                # Track this shot for cross-method deduplication
                self._add_to_recent_shots(representative_shot)

                # Clear processed group
                self.pending_shot_group = []
                self.last_group_processed_frame = self.frame_count

        # FALLBACK: Keep old score region logic for cases where shoot class isn't detected
        elif len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:

            # Made: Enters hoop region, shortly after enters down region,
            # Attempt: Enters up region, then exits up region without entering hoop region

            if self.frame_count - self.rim_last_detected < self.frame_rate and self.ball_detected and self.attempt_cooldown == 0:
                if in_score_region(self.ball_pos, self.hoop_pos):
                    if self.ball_entered:
                        self.attempt_time += 1
                    else:
                        self.ball_entered = True
                        self.attempt_time = 1
                    

                    #Add linear interpolation
                    if not self.last_point_in_region:
                        self.last_point_in_region = self.ball_pos[-1]
                        scored = False
                    else: 
                        scored = detect_score(self.ball_pos, self.hoop_pos, self.last_point_in_region)

                    # self.attempts += 1
                    # self.frame = cv2.putText(self.frame, 'DOWN', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if scored:
                        # print(self.ball_pos[-1], self.last_point_in_region, self.hoop_pos[-1])
                        # time_string = get_time_string(self.timestamp)
                        self.makes += 1
                        self.attempts += 1

                        # Create detection task dictionary with current state
                        detection_task = {
                            'frame_track': deepcopy(self.frame_track),
                            'timestamp': self.timestamp,
                            'is_scored': True,
                            'video_id': self.video_id
                        }
                        self.detection_queue.put(detection_task)

                        # shot_location, shot_timestamp = self.shot_detection()

                        # if shot_location:
                        #     scaled_shot_location = (shot_location[0] / self.width, shot_location[1] / self.height)
                        # else:
                        #     scaled_shot_location = (None, None)

                        # if shot_timestamp:
                        #     logger.log(INFO, f"Shot detected at {shot_timestamp}  |  Location: {shot_location}")
                        # self.on_detect(self.timestamp, True, self.video_id, scaled_shot_location)
                        # move shoot detection here
                        # self.should_detect_shot = True

                        logger.log(INFO, f"[FALLBACK - MADE] [{get_time_string(detection_task['timestamp'])}] Shot made at frame {self.frame_count}")
                    # else:
                    #     self.detect_callback(max(0, self.timestamp-3000), self.timestamp+2000, False)
                    #     print("attempt made")
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames
                        self.attempt_cooldown = self.MADE_ATTEMPT_COOLDOWN
                        self.screen_shot_moment = True
                        self.last_point_in_region = None
                        self.ball_entered = False
                        self.attempt_time = 0

                        # CRITICAL: Clear pending_shot_group to prevent duplicate detections
                        # Fallback detection is authoritative (ball went through hoop)
                        if len(self.pending_shot_group) > 0:
                            logger.log(INFO, f"[FALLBACK] Clearing {len(self.pending_shot_group)} pending shoot class detection(s) - fallback is authoritative")
                            self.pending_shot_group = []

                        # Track fallback detection for cross-method deduplication
                        # Use ball position as proxy for shooter position
                        if len(self.ball_pos) > 0:
                            ball_x, ball_y = self.ball_pos[-1][0]
                            fallback_shot = {
                                'frame': self.frame_count,
                                'timestamp': self.timestamp,
                                'shooter_position': {'x': ball_x, 'y': ball_y}
                            }
                            self._add_to_recent_shots(fallback_shot)
                    
                    else:
                        self.last_point_in_region = self.ball_pos[-1]
                

                else:
                    if self.ball_entered:
                        self.attempt_time += 1

                    if self.attempt_time >= self.ATTEMPT_DETECTION_INTERVAL:
                        time_string = get_time_string(self.timestamp)
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames

                        # Create detection task dictionary with current state
                        detection_task = {
                            'frame_track': deepcopy(self.frame_track),
                            'timestamp': self.timestamp,
                            'is_scored': False,
                            'video_id': self.video_id
                        }
                        self.detection_queue.put(detection_task)

                        # on_detect(timestamp, success, video_id, shot_location)
                        # shot_location, shot_timestamp = self.shot_detection()
                        # if shot_location:
                        #     scaled_shot_location = (shot_location[0] / self.width, shot_location[1] / self.height)
                        # else:
                        #     scaled_shot_location = (None, None)
                        
                        # if shot_timestamp:
                        #     #TODO: logic for shot localization
                        #     logger.log(INFO, f"Shot detected at {shot_timestamp}  |  Location: {shot_location}")

                        # self.on_detect(self.timestamp, False, self.video_id, scaled_shot_location)
                        # move shoot detection here
                        # self.should_detect_shot = True
                        logger.log(INFO, f"[FALLBACK - MISS] [{get_time_string(detection_task['timestamp'])}] Attempt made at frame {self.frame_count}")
                        self.attempts += 1

                        self.attempt_cooldown = self.MISS_ATTEMPT_COOLDOWN
                        self.screen_shot_moment = True

                        self.attempt_time = 0
                        self.ball_entered = False
                        self.last_point_in_region = None

                        # CRITICAL: Clear pending_shot_group to prevent duplicate detections
                        # Fallback detection is authoritative (ball trajectory detected)
                        if len(self.pending_shot_group) > 0:
                            logger.log(INFO, f"[FALLBACK] Clearing {len(self.pending_shot_group)} pending shoot class detection(s) - fallback is authoritative")
                            self.pending_shot_group = []

                        # Track fallback detection for cross-method deduplication
                        # Use ball position as proxy for shooter position
                        if len(self.ball_pos) > 0:
                            ball_x, ball_y = self.ball_pos[-1][0]
                            fallback_shot = {
                                'frame': self.frame_count,
                                'timestamp': self.timestamp,
                                'shooter_position': {'x': ball_x, 'y': ball_y}
                            }
                            self._add_to_recent_shots(fallback_shot)
    

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
        cv2.putText(self.frame, str(self.attempt_time), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def get_side(self):
        if len(self.hoop_pos):
            return 1 if self.hoop_pos[-1][0][0] > self.width/2 else 0

        return None

    def save_localization_results(self, video_path, calibration_points):
        """Save shot localization results to JSON and generate final shot chart"""
        if not self.enable_localization or not self.shots_data:
            return

        from pathlib import Path
        import json

        # Create output directory
        os.makedirs('output', exist_ok=True)

        # Generate video name
        video_name = Path(video_path).stem

        # Save JSON data
        json_path = f"output/{video_name}_shots_data.json"

        output_data = {
            'video_info': {
                'path': video_path,
                'resolution': f"{self.width}x{self.height}",
                'fps': self.frame_rate,
                'duration_seconds': self.frame_count / self.frame_rate if self.frame_rate > 0 else 0
            },
            'calibration_6pt_FIBA': calibration_points if isinstance(calibration_points, list) else [],
            'total_shots': len(self.shots_data),
            'made_shots': sum(1 for s in self.shots_data if s['is_made']),
            'missed_shots': sum(1 for s in self.shots_data if not s['is_made']),
            'success_rate': (sum(1 for s in self.shots_data if s['is_made']) / len(self.shots_data) * 100) if self.shots_data else 0,
            'shots': self.shots_data
        }

        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.log(INFO, f"✓ Shot data saved: {json_path}")

        # Generate final shot chart
        if self.shot_localizer:
            chart_path = f"output/{video_name}_shot_chart.png"
            self.shot_localizer.get_shot_chart(save_path=chart_path)
            logger.log(INFO, f"✓ Shot chart saved: {chart_path}")

        logger.log(INFO, f"✓ Localization complete: {len(self.shots_data)} shots | "
                        f"{output_data['made_shots']}/{output_data['total_shots']} made ({output_data['success_rate']:.1f}%)")

    def _deduplicate_shot_group(self, shot_group):
        """
        Deduplicate shot group by grouping detections within frame/position thresholds.
        Returns the first frame from the first subgroup as the representative shot.

        Args:
            shot_group: List of shot dictionaries with 'frame', 'timestamp', 'shoot_confidence', 'shooter_position'

        Returns:
            dict: Representative shot (first frame) with added metadata, or None if empty
        """
        if not shot_group:
            return None

        # Sort by frame number
        sorted_shots = sorted(shot_group, key=lambda s: s['frame'])

        # Group by frame and position proximity
        subgroups = []
        current_subgroup = [sorted_shots[0]]

        for i in range(1, len(sorted_shots)):
            current = sorted_shots[i]
            last = current_subgroup[-1]

            # Check frame distance
            frame_dist = current['frame'] - last['frame']

            # Check position distance (if both have valid positions)
            pos_dist = float('inf')
            if current['shooter_position'] and last['shooter_position']:
                p1 = current['shooter_position']
                p2 = last['shooter_position']
                pos_dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

            # Group if within thresholds
            if (frame_dist <= self.DEDUPLICATION_FRAME_THRESHOLD and
                pos_dist <= self.DEDUPLICATION_POSITION_THRESHOLD):
                current_subgroup.append(current)
            else:
                # Start new subgroup
                subgroups.append(current_subgroup)
                current_subgroup = [current]

        # Don't forget last subgroup
        if current_subgroup:
            subgroups.append(current_subgroup)

        # Return FIRST frame from FIRST subgroup (user requirement)
        representative = subgroups[0][0].copy()
        representative['grouped_frames'] = len(subgroups[0])
        representative['total_raw_detections'] = len(shot_group)

        # Log deduplication info
        if len(shot_group) > 1:
            logger.log(INFO, f"Deduplication: {len(shot_group)} raw detections → "
                             f"{len(subgroups)} unique shot(s) | "
                             f"Selected frame {representative['frame']} (first in group)")

        return representative

    def _is_duplicate_of_recent_shot(self, shot_data):
        """
        Check if the given shot is a duplicate of a recently detected shot.
        This handles cross-method deduplication (e.g., shoot class vs fallback detection).

        Args:
            shot_data: Dictionary with 'frame', 'timestamp', 'shooter_position' (required)

        Returns:
            bool: True if this shot should be skipped as a duplicate
        """
        if not self.recent_shots:
            logger.log(INFO, f"[DEDUP] recent_shots is EMPTY - cannot deduplicate")
            return False

        # Shooter position is guaranteed to exist (checked before calling this)
        current_frame = shot_data['frame']
        current_pos = shot_data['shooter_position']

        logger.log(INFO, f"[DEDUP] Current shot: frame {current_frame} at ({current_pos['x']:.1f}, {current_pos['y']:.1f})")

        # Check against recent shots
        for i, recent_shot in enumerate(self.recent_shots):
            recent_frame, recent_timestamp, recent_pos = recent_shot

            # Skip if no position data (from fallback detections)
            if not recent_pos:
                logger.log(INFO, f"[DEDUP]   Recent #{i}: frame {recent_frame} - NO POS (skipping)")
                continue

            # Check frame distance
            frame_dist = abs(current_frame - recent_frame)
            if frame_dist > self.DEDUPLICATION_FRAME_THRESHOLD:
                logger.log(INFO, f"[DEDUP]   Recent #{i}: frame {recent_frame} - frame_dist={frame_dist} > {self.DEDUPLICATION_FRAME_THRESHOLD} (TOO FAR)")
                continue  # Too far apart in time

            # Check position distance
            pos_dist = np.sqrt(
                (current_pos['x'] - recent_pos['x'])**2 +
                (current_pos['y'] - recent_pos['y'])**2
            )

            logger.log(INFO, f"[DEDUP]   Recent #{i}: frame {recent_frame} at ({recent_pos['x']:.1f}, {recent_pos['y']:.1f}) - "
                            f"frame_dist={frame_dist}, pos_dist={pos_dist:.1f}px")

            # If within both thresholds, it's a duplicate
            if pos_dist <= self.DEDUPLICATION_POSITION_THRESHOLD:
                logger.log(INFO, f"[DEDUP]   → DUPLICATE FOUND! pos_dist={pos_dist:.1f}px <= {self.DEDUPLICATION_POSITION_THRESHOLD}px")
                return True

        logger.log(INFO, f"[DEDUP]   → No duplicates found")
        return False

    def _add_to_recent_shots(self, shot_data):
        """
        Add a shot to the recent_shots list for cross-method deduplication.
        Maintains a rolling window of recent shots.

        Args:
            shot_data: Dictionary with 'frame', 'timestamp', 'shooter_position'
        """
        if not shot_data:
            return

        pos = shot_data.get('shooter_position')
        pos_str = f"({pos['x']:.1f}, {pos['y']:.1f})" if pos else "NO POS"

        # Add to recent shots
        self.recent_shots.append((
            shot_data['frame'],
            shot_data['timestamp'],
            pos
        ))

        logger.log(INFO, f"[DEDUP] Added to recent_shots: frame {shot_data['frame']} at {pos_str} | Total: {len(self.recent_shots)}")

        # Keep only recent shots (within 2x the deduplication window)
        max_history = self.DEDUPLICATION_FRAME_THRESHOLD * 2
        self.recent_shots = [
            shot for shot in self.recent_shots
            if self.frame_count - shot[0] <= max_history
        ]

    def _detection_worker(self):
        """Background thread that processes shot detection tasks."""
        while self.detection_thread_active:
            try:
                # Get detection task dictionary from queue
                task = self.detection_queue.get(timeout=1.0)

                # Extract pre-captured position (NEW: no re-inference!)
                shot_data = task.get('shot_data')
                shot_location = None
                shot_timestamp = None

                if shot_data and shot_data.get('shooter_position'):
                    # Use captured position directly (NO re-inference!)
                    shooter_pos = shot_data['shooter_position']
                    shot_location = (shooter_pos['x'], shooter_pos['y'])
                    shot_timestamp = get_time_string(task['timestamp'])
                    logger.log(INFO, f"Using pre-captured shooter position: ({shot_location[0]:.1f}, {shot_location[1]:.1f})")
                else:
                    # Fallback: re-run inference if position not captured (backward compatibility)
                    logger.log(INFO, "No pre-captured position, falling back to _process_shot_detection()")
                    shot_location, shot_timestamp = self._process_shot_detection(task['frame_track'])

                # Initialize variables
                scaled_shot_location = (None, None)
                court_position = None
                zone = None
                shot_type = None

                if shot_location and shot_timestamp:
                    # Scale shot location for backward compatibility
                    scaled_shot_location = (shot_location[0] / self.width, shot_location[1] / self.height)

                    # Transform to court coordinates if localization enabled
                    if self.enable_localization and self.shot_localizer:
                        court_position = self.shot_localizer.map_to_court(shot_location)

                        # Classify zone and shot type
                        if court_position[0] is not None and court_position[1] is not None:
                            from statistics import TeamStatistics
                            stats = TeamStatistics(quarters=[float('inf')])
                            zone = stats.determine_zone(court_position[0], court_position[1])

                            if zone:
                                is_three = stats.determine_is_three_pt(zone)
                                shot_type = '3PT' if is_three else '2PT'

                                logger.log(INFO, f"Shot localized: ({court_position[0]:.2f}m, {court_position[1]:.2f}m) | "
                                           f"Zone: {zone} | {shot_type}")

                            # Visualize shot on court
                            self.shot_localizer.visualize_shot(
                                court_location=court_position,
                                is_made=task['is_scored'],
                                timestamp=get_time_string(task['timestamp'])
                            )

                            # Accumulate shot data for JSON export
                            shot_info = {
                                'shot_number': len(self.shots_data) + 1,
                                'timestamp': task['timestamp'],
                                'timestamp_string': get_time_string(task['timestamp']),
                                'is_made': task['is_scored'],
                                'pixel_position': {
                                    'x': float(shot_location[0]),
                                    'y': float(shot_location[1]),
                                    'normalized_x': float(scaled_shot_location[0]),
                                    'normalized_y': float(scaled_shot_location[1])
                                },
                                'court_position_meters': {
                                    'x': float(court_position[0]),
                                    'y': float(court_position[1])
                                },
                                'zone': int(zone) if zone else None,
                                'shot_type': shot_type
                            }
                            self.shots_data.append(shot_info)
                        else:
                            logger.log(INFO, f"Shot location out of court bounds: {court_position}")

                # Log detection results using preserved state
                log_msg = f"Shot detected at {get_time_string(task['timestamp'])} | "
                if court_position and court_position[0] is not None:
                    log_msg += f"Court: ({court_position[0]:.2f}m, {court_position[1]:.2f}m) | "
                else:
                    log_msg += f"Pixel: {scaled_shot_location} | "
                log_msg += f"Video: {task['video_id']} | {'Made' if task['is_scored'] else 'Missed'}"
                logger.log(INFO, log_msg)

                # Call on_detect with preserved state (pass court_position as 4th arg)
                self.on_detect(task['timestamp'], task['is_scored'], task['video_id'], scaled_shot_location, court_position)

                # Mark task as done
                self.detection_queue.task_done()

            except Empty:
                # Queue timeout - continue waiting
                continue
            except Exception as e:
                logger.log(INFO, f"Error in detection worker: {str(e)}")
                import traceback
                traceback.print_exc()

                # Mark task as done even on error to prevent queue.join() from hanging
                try:
                    self.detection_queue.task_done()
                except:
                    pass
                continue

    def _process_shot_detection(self, frame_track):
        """
        Process shot detection on a snapshot of frame_track.
        This is the same as the original shot_detection method but works on passed data.
        """
        shot_location = None
        shooter_positions = []
        debug_timestamp = None
        debug_frame = None
        debug_frame_count = None
        
        
        # Step 1: Find frames with "shoot" class
        for frame_data in frame_track:
            frame_det_img, frame_count, timestamp, frame_img = frame_data
            # frame_det_img is already being resized to match - force 1280 and 720 for better model results
            shoot_box = None
            person_boxes = []
            # Apply shoot detection model
            results = self.model(frame_det_img, stream=True, verbose=False, imgsz=self.inference_width, device=env.get('device', 0))

            for r in results:
                boxes = sorted([(box.xyxy[0], box.conf, box.cls) for box in r.boxes], key=lambda x: -x[1])
                #sort and get only top prediction for ball / hoop
                
                # Define confidence thresholds for each class
                conf_thresholds = {
                    'rim': 0.5,
                    'ball': 0.5,
                    'shoot': 0.3,  # Lowered from 0.65 to match annotate_video_fixed.py
                    'person': 0.5
                }
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box[0]

                    # Scale back up to original dimensions
                    x1, y1, x2, y2 = int(x1 * self.width/self.inference_width), int(y1 * self.height/self.inference_height), int(x2 * self.width/self.inference_width), int(y2* self.height/self.inference_height)
                    w, h = x2 - x1, y2 - y1
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Confidence
                    conf = math.ceil((box[1] * 100)) / 100

                    # Class Name
                    cls = int(box[2])
                    current_class = self.class_names[cls]
                    # Store box info based on class and confidence threshold
                    if conf > conf_thresholds[current_class]:
                        box_info = {
                            'center': center,
                            'width': w,
                            'height': h,
                            'confidence': conf,
                            'coords': (x1, y1, x2, y2)
                        }

                        if current_class == 'shoot' and not shoot_box:
                            # print("shoot detected, being processed...")
                            shoot_box = box_info
                            # Save annotated frame as image if shoot detected for debugging
                            annotated_frame = r.plot()
                            frame_filename = os.path.join(self.output_all_shot, f"all_shot_{self.frame_count}_{get_time_string(self.timestamp)}.jpg")
                            cv2.imwrite(frame_filename, annotated_frame)
                            # logger.log(INFO, f"shoot box recorded? {shoot_box} at {get_time_string(self.timestamp)}")
                        
                        elif current_class == 'person':
                            person_boxes.append(box_info)                        
            # if shoot_box:
            #     logger.log(INFO, f"hv shoot box? {shoot_box and 1}")
            if shoot_box and person_boxes:
                # Step 2: Find nearest person to the shoot box by calculating overlap    
                max_overlap = 0
                closest_person = None
                
                # Get shoot box coordinates (y increases downward)
                x1_shoot = shoot_box['coords'][0]  # Left
                y1_shoot = shoot_box['coords'][1]  # Top
                x2_shoot = shoot_box['coords'][2]  # Right 
                y2_shoot = shoot_box['coords'][3]  # Bottom
                
                for person_box in person_boxes:
                    # Get person box coordinates (y increases downward)
                    x1_person = person_box['coords'][0]  # Left
                    y1_person = person_box['coords'][1]  # Top
                    x2_person = person_box['coords'][2]  # Right
                    y2_person = person_box['coords'][3]  # Bottom
                    
                    # Calculate intersection
                    x_left = max(x1_shoot, x1_person)
                    y_top = max(y1_shoot, y1_person)
                    x_right = min(x2_shoot, x2_person)
                    y_bottom = min(y2_shoot, y2_person)
                    
                    if x_right > x_left and y_bottom > y_top:
                        overlap_area = (x_right - x_left) * (y_bottom - y_top)
                        if overlap_area > max_overlap:
                            max_overlap = overlap_area
                            closest_person = person_box
                
                # Calculate shoot box area
                shoot_box_area = (x2_shoot - x1_shoot) * (y2_shoot - y1_shoot)
                # logger.log(INFO, f"hv cloest person? {closest_person}")
                # logger.log(INFO, f"hv enough overlap? {max_overlap >= 0.7 * shoot_box_area}")
                if closest_person and max_overlap >= 0.7 * shoot_box_area:
                    x1, y1, x2, y2 = closest_person['coords']
                    bottom_center_x = x1 + (x2 - x1) // 2  # Center x coordinate
                    bottom_center_y = y2  # Bottom y coordinate
                    shooter_positions.append((bottom_center_x, bottom_center_y))
                    # logger.log(INFO, f"frame being added into shooter_positions")
                    # Mark the first frame with shoot box as the starting point of shooting, for debugging purpose
                    if not debug_timestamp:
                        debug_timestamp = get_time_string(timestamp)
                        debug_frame = frame_img
                        debug_frame_count = frame_count

        # Step 2.5: Fallback - If not enough shoot detections, use IoU with ball
        if len(shooter_positions) < 5:  # Threshold for minimum confidence
            logger.log(INFO, f"Only {len(shooter_positions)} shoot detections found, using IoU fallback...")

            for frame_data in frame_track:
                frame_det_img, frame_count, timestamp, frame_img = frame_data

                ball_box = None
                person_boxes = []

                # Run shoot detection model again to get ball and person boxes
                results = self.model(frame_det_img, stream=True, verbose=False, imgsz=self.inference_width, device=env.get('device', 0))

                for r in results:
                    boxes = sorted([(box.xyxy[0], box.conf, box.cls) for box in r.boxes], key=lambda x: -x[1])

                    conf_thresholds = {
                        'rim': 0.5,
                        'ball': 0.5,
                        'shoot': 0.3,
                        'person': 0.5
                    }

                    for box in boxes:
                        x1, y1, x2, y2 = box[0]
                        x1, y1, x2, y2 = int(x1 * self.width/self.inference_width), int(y1 * self.height/self.inference_height), int(x2 * self.width/self.inference_width), int(y2* self.height/self.inference_height)
                        w, h = x2 - x1, y2 - y1

                        conf = math.ceil((box[1] * 100)) / 100
                        cls = int(box[2])
                        current_class = self.class_names[cls]

                        if conf > conf_thresholds[current_class]:
                            box_info = {
                                'center': (int(x1 + w / 2), int(y1 + h / 2)),
                                'width': w,
                                'height': h,
                                'confidence': conf,
                                'coords': (x1, y1, x2, y2)
                            }

                            if current_class == 'ball' and not ball_box:
                                ball_box = box_info
                            elif current_class == 'person':
                                person_boxes.append(box_info)

                # Find person with highest IoU with ball
                if ball_box and person_boxes:
                    max_iou = 0
                    best_person = None

                    ball_coords = ball_box['coords']
                    for person_box in person_boxes:
                        person_coords = person_box['coords']

                        # Calculate IoU
                        x_left = max(ball_coords[0], person_coords[0])
                        y_top = max(ball_coords[1], person_coords[1])
                        x_right = min(ball_coords[2], person_coords[2])
                        y_bottom = min(ball_coords[3], person_coords[3])

                        if x_right > x_left and y_bottom > y_top:
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
                            ball_area = (ball_coords[2] - ball_coords[0]) * (ball_coords[3] - ball_coords[1])
                            person_area = (person_coords[2] - person_coords[0]) * (person_coords[3] - person_coords[1])
                            union_area = ball_area + person_area - intersection_area

                            iou = intersection_area / union_area if union_area > 0 else 0

                            if iou > max_iou:
                                max_iou = iou
                                best_person = person_box

                    # Use person with highest IoU (even if low)
                    if best_person and max_iou > 0.05:  # Very low threshold for fallback
                        x1, y1, x2, y2 = best_person['coords']
                        bottom_center_x = x1 + (x2 - x1) // 2
                        bottom_center_y = y2
                        shooter_positions.append((bottom_center_x, bottom_center_y))

            logger.log(INFO, f"After IoU fallback: {len(shooter_positions)} total positions")

        # Step 3: Use IQR to filter outliers after all 120 frames is processed and recorded
        if shooter_positions:
            # logger.log(INFO, f"enter shooter_positions")
            x_coords = [pos[0] for pos in shooter_positions]
            y_coords = [pos[1] for pos in shooter_positions]
            
            # Calculate Q1, Q3 and IQR for both x and y coordinates
            q1_x, q3_x = np.percentile(x_coords, [25, 75])
            q1_y, q3_y = np.percentile(y_coords, [25, 75])
            iqr_x = q3_x - q1_x
            iqr_y = q3_y - q1_y
            
            # Define bounds
            x_lower = q1_x - 1.5 * iqr_x
            x_upper = q3_x + 1.5 * iqr_x
            y_lower = q1_y - 1.5 * iqr_y
            y_upper = q3_y + 1.5 * iqr_y
            
            # Filter out outliers
            filtered_positions = [
                pos for pos in shooter_positions 
                if (x_lower <= pos[0] <= x_upper and y_lower <= pos[1] <= y_upper)
            ]
            
            if filtered_positions:
                # logger.log(INFO, f"enter filtered_position")
                # Calculate average position
                avg_x = sum(pos[0] for pos in filtered_positions) / len(filtered_positions)
                avg_y = sum(pos[1] for pos in filtered_positions) / len(filtered_positions)
                shot_location = (avg_x, avg_y)
        
            # Plot shot location on debug frame if available
            # if shot_location:
            #     # Convert coordinates to integers for cv2
            #     plot_x = int(avg_x)
            #     plot_y = int(avg_y)
                
                # Draw red circle at shot location
                # cv2.circle(debug_frame, (plot_x, plot_y), 5, (0,0,255), -1)
                
                # # Save the annotated frame
                # output_path = os.path.join(self.output_true_shot, f"true_shot_{debug_frame_count}_{debug_timestamp}.jpg")
                # cv2.imwrite(output_path, debug_frame)
        return shot_location, debug_timestamp

def get_time_string(timestamp):
    timestamp = max(0, timestamp)

    t = str(timedelta(milliseconds=timestamp)).split('.')[0]
    return datetime.strptime(t, "%H:%M:%S").strftime('%H:%M:%S')


if __name__ == "__main__":
    def print_stats(score_report, is_match):
        makes = len(score_report.get('makes', []))
        attempts = len(score_report.get('attempts', []))
        success_rate = (makes / attempts * 100) if attempts > 0 else 0
        print(f"Shot made: {makes}")
        print(f"Attempts: {attempts}")
        print(f"Success rate: {success_rate:.2f}%")
        
    def dummy_on_complete():
        return 0

    ShotDetector(env['input'], lambda x,y,z,k: 0, dummy_on_complete, False)




