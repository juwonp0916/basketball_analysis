from score_detection.team_detector import StreamingTeamDetector
import cv2
import numpy as np
import colorsys
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ultralytics import YOLO

# Try importing the TeamDetector logic from backend if possible
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))


def hsv_to_rgb(h, s, v):
    """Convert OpenCV HSV to RGB (0-1) for matplotlib"""
    r, g, b = colorsys.hsv_to_rgb(h / 179.0, s / 255.0, v / 255.0)
    return (r, g, b)


def hsv_to_hex(hsv):
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h / 179.0, s / 255.0, v / 255.0)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def main():
    video_path = 'backend/sample_videos/video1.mp4'
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    output_plot = 'clustering_rationale.png'
    output_video = 'team_clustering_output.mp4'

    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # Standard YOLO for person detection

    detector = StreamingTeamDetector()

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Extracting features from initial frames for clustering rationale...")
    all_features = []
    all_boxes = []
    all_frames = []

    # Collect features from first 5 frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

        results = model(frame, verbose=False, imgsz=640, classes=[0])  # 0 is person

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            region = detector._get_jersey_region((x1, y1, x2, y2), frame)
            if region is not None:
                feat = detector._extract_color_features(region)
                if feat is not None:
                    all_features.append(feat)

    if len(all_features) < 4:
        print("Not enough players detected for clustering.")
        return

    # Transform features for clustering
    X = []
    for f in all_features:
        h, s, v = f
        angle = h * 2.0 * np.pi / 180.0
        X.append([np.cos(angle) * 100, np.sin(angle) * 100, s, v * 0.5])
    X = np.array(X)

    # Run KMeans
    n_clusters = min(3, len(all_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X)
    labels = np.array(kmeans.labels_)

    # Extract actual colors for each of the K clusters (including the 3rd cluster)
    cluster_colors_hsv = {}
    cluster_colors_hex = {}
    for idx in range(n_clusters):
        cluster_items = [all_features[i] for i in range(len(all_features)) if labels[i] == idx]
        if cluster_items:
            median_feat = np.median(cluster_items, axis=0)
            cluster_colors_hsv[idx] = median_feat
            cluster_colors_hex[idx] = hsv_to_hex(median_feat)
        else:
            cluster_colors_hsv[idx] = np.array([0, 0, 0])
            cluster_colors_hex[idx] = "#000000"

    # Plotting the rationale
    print("Plotting clustering rationale...")
    plt.figure(figsize=(10, 8))

    # Scatter plot based on Sin/Cos of Hue
    scatter_colors = [hsv_to_rgb(*f) for f in all_features]
    plt.scatter(X[:, 0], X[:, 1], c=scatter_colors, s=100, edgecolor='black', linewidth=0.5, alpha=0.7)

    # Plot cluster centroids with their ACTUAL cluster color
    centroids = kmeans.cluster_centers_
    for idx in range(n_clusters):
        centroid_color_rgb = hsv_to_rgb(*cluster_colors_hsv[idx])
        c_hex = cluster_colors_hex[idx]
        plt.scatter(centroids[idx, 0], centroids[idx, 1], c=[centroid_color_rgb],
                    marker='X', s=400, edgecolors='black', linewidth=2,
                    label=f'Cluster {idx} ({c_hex})')

    plt.title('Team Color Clustering Rationale (Hue mapped to Sin/Cos space)')
    plt.xlabel('Cos(Hue) * 100')
    plt.ylabel('Sin(Hue) * 100')
    plt.legend()

    # Find top 2 clusters for detector calibration
    counts = np.bincount(labels)
    top_2_idx = np.argsort(counts)[-2:]

    detector.team0_color = cluster_colors_hsv[top_2_idx[0]]
    detector.team1_color = cluster_colors_hsv[top_2_idx[1]]
    detector._configured = True

    team0_hex = cluster_colors_hex[top_2_idx[0]]
    team1_hex = cluster_colors_hex[top_2_idx[1]]

    # Add text box with team colors
    plt.text(0.05, 0.95, f'Team 0: {team0_hex}\nTeam 1: {team1_hex}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved clustering rationale to {output_plot}")

    # Process video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Exporting annotated video to {output_video} (processing all {total_frames} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, imgsz=640, classes=[0])

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Extract features specifically for this box to map it to ALL clusters
            region = detector._get_jersey_region((x1, y1, x2, y2), frame)
            color_bgr = (128, 128, 128)
            label = "Unknown"

            if region is not None:
                feat = detector._extract_color_features(region)
                if feat is not None:
                    h, s, v = feat
                    angle = h * 2.0 * np.pi / 180.0
                    X_feat = np.array([[np.cos(angle) * 100, np.sin(angle) * 100, s, v * 0.5]])

                    cluster_idx = kmeans.predict(X_feat)[0]
                    cluster_hsv = cluster_colors_hsv[cluster_idx]
                    cluster_hex = cluster_colors_hex[cluster_idx]

                    r, g, b = hsv_to_rgb(*cluster_hsv)
                    color_bgr = (int(b * 255), int(g * 255), int(r * 255))
                    label = f"Cluster {cluster_idx} ({cluster_hex})"

                    # Highlight if it's one of the top 2 teams
                    if cluster_idx == top_2_idx[0]:
                        label = f"Team 0 {cluster_hex}"
                    elif cluster_idx == top_2_idx[1]:
                        label = f"Team 1 {cluster_hex}"

            # Draw a filled background for text to make it readable over the actual color
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color_bgr, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            # Pick black or white text depending on brightness of background
            brightness = sum(color_bgr) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Saved annotated video to {output_video}")
    print("Done!")


if __name__ == "__main__":
    main()
