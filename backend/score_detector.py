from ultralytics import YOLO
from utils import detect_score


class ScoreDetector:
    """
    Determines whether a detected shot attempt resulted in a score or a miss.

    Uses two signals:
    1. YOLO 'made' class from the score model: the model detects the ball
       occluded by the net, indicating the ball has passed through the hoop.
    2. Trajectory analysis: checks consecutive ball positions to see if
       the ball entered the hoop zone from above and passed through
       downward (via linear interpolation against the hoop boundary).
    """

    def __init__(self, config):
        print("Loading score detection YOLO model...")
        self.model = YOLO(config['weights_path'])

    def detect(self, shot_data, frame=None, device='cpu'):
        """
        Analyze shot attempt data to determine score or miss.

        Args:
            shot_data: dict from ShotDetector.detect_shot() containing:
                - ball_positions: list of ((x,y), frame, w, h, conf) tuples
                - rim_positions: list of ((x,y), frame, w, h, conf) tuples
                - last_point_in_region: last ball position in scoring region
            frame: BGR video frame to run 'made' detection on (optional)
            device: Device for YOLO inference

        Returns:
            bool: True if scored, False if missed
        """
        if shot_data is None:
            return False

        # Signal 1: Run model to check for 'made' class on current frame
        if frame is not None:
            results = self.model(frame, verbose=False, device=device)
            if len(results) > 0:
                names = self.model.names
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    class_name = names[cls]
                    if class_name == 'made' and float(box.conf[0]) > 0.6:
                        return True

        # Signal 2: Ball trajectory analysis
        ball_positions = shot_data.get('ball_positions', [])
        rim_positions = shot_data.get('rim_positions', [])
        last_point = shot_data.get('last_point_in_region')

        if len(ball_positions) < 2 or len(rim_positions) == 0:
            return False

        # Check consecutive position pairs for ball passing through hoop
        for i in range(1, len(ball_positions)):
            if detect_score(ball_positions[i - 1:i + 1], rim_positions, ball_positions[i - 1]):
                return True

        # Also check from the last tracked position in the scoring region
        if last_point is not None:
            if detect_score(ball_positions, rim_positions, last_point):
                return True

        return False
