import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize YOLO player detector.

        Args:
            model_path: Path to YOLO model weights (default: yolov8n.pt - nano model)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_players(self, frame):
        """
        Detect players in a frame using YOLO.

        Args:
            frame: Input image (BGR format)

        Returns:
            List of bounding boxes [x1, y1, x2, y2] for detected persons
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)

        player_bboxes = []

        # Extract person detections (class_id = 0 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Filter for 'person' class (class_id = 0) with sufficient confidence
                if class_id == 0 and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    player_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

        return player_bboxes


class JerseyTeamClassifier:
    def __init__(self, n_dominant_colors=3):
        """
        Initialize the jersey team classifier.

        Args:
            n_dominant_colors: Number of dominant colors to extract from each jersey
        """
        self.n_dominant_colors = n_dominant_colors
        self.team_colors = None

    def extract_dominant_colors(self, image_region):
        """
        Extract dominant colors from a jersey region using K-Means clustering.

        Args:
            image_region: BGR image of the jersey region

        Returns:
            Array of dominant colors in RGB format
        """
        # Reshape image to be a list of pixels
        pixels = image_region.reshape(-1, 3)

        # Convert BGR to RGB
        pixels = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB).reshape(-1, 3)

        # Apply K-Means clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.n_dominant_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get the dominant colors (cluster centers)
        colors = kmeans.cluster_centers_

        # Get the labels and count frequency of each color
        labels = kmeans.labels_
        label_counts = np.bincount(labels)

        # Sort colors by frequency (most dominant first)
        sorted_indices = np.argsort(-label_counts)
        dominant_colors = colors[sorted_indices]

        return dominant_colors.astype(int)

    def get_jersey_region(self, player_bbox, frame):
        """
        Extract the jersey region from a player bounding box.
        Focuses on upper body area where jersey is most visible.

        Args:
            player_bbox: [x1, y1, x2, y2] bounding box coordinates
            frame: Full frame image

        Returns:
            Cropped jersey region
        """
        x1, y1, x2, y2 = player_bbox

        # Focus on upper 60% of the bounding box (upper body/jersey area)
        height = y2 - y1
        jersey_y2 = y1 + int(height * 0.6)

        # Extract the region
        jersey_region = frame[y1:jersey_y2, x1:x2]

        return jersey_region

    def classify_teams(self, player_bboxes, frame):
        """
        Classify players into two teams based on jersey colors.

        Args:
            player_bboxes: List of [x1, y1, x2, y2] bounding boxes for each player
            frame: Full frame image

        Returns:
            team_assignments: Array of team labels (0 or 1) for each player
            team_colors: Dictionary with representative colors for each team
        """
        if len(player_bboxes) == 0:
            return [], {}

        # Extract dominant colors for each player
        player_color_features = []

        for bbox in player_bboxes:
            jersey_region = self.get_jersey_region(bbox, frame)

            if jersey_region.size == 0:
                # Handle empty regions
                player_color_features.append(np.zeros((self.n_dominant_colors, 3)))
                continue

            dominant_colors = self.extract_dominant_colors(jersey_region)
            player_color_features.append(dominant_colors)

        # Flatten color features for clustering
        # Each player is represented by their dominant colors concatenated
        player_color_features = np.array(player_color_features)
        features_flat = player_color_features.reshape(len(player_bboxes), -1)

        # Cluster players into 2 teams based on color similarity
        team_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        team_assignments = team_kmeans.fit_predict(features_flat)

        # Store representative colors for each team
        team_colors = {
            0: player_color_features[team_assignments == 0].mean(axis=0)[0],
            1: player_color_features[team_assignments == 1].mean(axis=0)[0]
        }

        self.team_colors = team_colors

        return team_assignments, team_colors

    def visualize_teams(self, frame, player_bboxes, team_assignments, team_colors):
        """
        Visualize the team classification by drawing bounding boxes.

        Args:
            frame: Original frame
            player_bboxes: List of player bounding boxes
            team_assignments: Team labels for each player
            team_colors: Representative colors for each team

        Returns:
            Annotated frame
        """
        result = frame.copy()

        # Define colors for visualization (BGR format)
        viz_colors = {
            0: (255, 0, 0),    # Blue for team 0
            1: (0, 0, 255)     # Red for team 1
        }

        for bbox, team in zip(player_bboxes, team_assignments):
            x1, y1, x2, y2 = bbox
            color = viz_colors[team]

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

            # Add team label
            label = f"Team {team}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result


def process_image(image_path, yolo_model='yolov8n.pt', output_path='team_classification_result.jpg'):
    """
    Complete pipeline: Detect players and classify teams by jersey color.

    Args:
        image_path: Path to basketball game image
        yolo_model: YOLO model to use (default: yolov8n.pt)
        output_path: Path to save the result image
    """
    # Load image
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Loaded image: {image_path}")
    print(f"Image shape: {frame.shape}")

    # Step 1: Detect players using YOLO
    print("\nStep 1: Detecting players with YOLO...")
    detector = PlayerDetector(model_path=yolo_model, confidence_threshold=0.5)
    player_bboxes = detector.detect_players(frame)

    print(f"Detected {len(player_bboxes)} players")

    if len(player_bboxes) == 0:
        print("No players detected! Try lowering the confidence threshold.")
        return

    # Step 2: Classify teams based on jersey colors
    print("\nStep 2: Classifying teams by jersey color...")
    classifier = JerseyTeamClassifier(n_dominant_colors=3)
    team_assignments, team_colors = classifier.classify_teams(player_bboxes, frame)

    print(f"\nTeam Assignments: {team_assignments}")
    print("Team Colors:")
    for team_id, color in team_colors.items():
        print(f"  Team {team_id}: RGB{tuple(color.astype(int))}")

    # Count players per team
    unique, counts = np.unique(team_assignments, return_counts=True)
    for team_id, count in zip(unique, counts):
        print(f"  Team {team_id}: {count} players")

    # Step 3: Visualize results
    print("\nStep 3: Visualizing results...")
    result = classifier.visualize_teams(frame, player_bboxes, team_assignments, team_colors)

    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Team Classification Based on Jersey Colors (YOLO + K-Means)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save result
    cv2.imwrite(output_path, result)
    print(f"\nResult saved to '{output_path}'")


def process_video(video_path, yolo_model='yolov8n.pt', output_path='team_classification_video.mp4'):
    """
    Process a video: Detect players and classify teams frame by frame.

    Args:
        video_path: Path to basketball game video
        yolo_model: YOLO model to use (default: yolov8n.pt)
        output_path: Path to save the output video
    """
    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize detector and classifier
    detector = PlayerDetector(model_path=yolo_model, confidence_threshold=0.5)
    classifier = JerseyTeamClassifier(n_dominant_colors=3)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Detect players
        player_bboxes = detector.detect_players(frame)

        # Classify teams (only if players detected)
        if len(player_bboxes) > 0:
            team_assignments, team_colors = classifier.classify_teams(player_bboxes, frame)
            result = classifier.visualize_teams(frame, player_bboxes, team_assignments, team_colors)
        else:
            result = frame

        # Write frame
        out.write(result)

        # Progress update
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    print(f"\nVideo processing complete!")
    print(f"Output saved to '{output_path}'")


if __name__ == "__main__":
    import sys

    print("Basketball Team Classifier - YOLO + K-Means")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python jersey_team_classifier.py <image_or_video_path> [output_path]")
        print("\nExamples:")
        print("  python jersey_team_classifier.py basketball_game.jpg")
        print("  python jersey_team_classifier.py game_video.mp4 output.mp4")
        print("\nOr use in your code:")
        print("  from jersey_team_classifier import process_image, process_video")
        print("  process_image('basketball_game.jpg')")
        print("  process_video('game_video.mp4')")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        # Check if input is image or video
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Process image
            if output_path is None:
                output_path = 'team_classification_result.jpg'
            process_image(input_path, output_path=output_path)

        elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Process video
            if output_path is None:
                output_path = 'team_classification_video.mp4'
            process_video(input_path, output_path=output_path)

        else:
            print(f"Error: Unsupported file format for {input_path}")
            print("Supported formats: .jpg, .jpeg, .png, .bmp (images) or .mp4, .avi, .mov, .mkv (videos)")
