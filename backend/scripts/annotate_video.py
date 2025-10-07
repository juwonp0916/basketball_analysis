"""
Create annotated video with detection overlays (ball, rim, person, shoot)
"""
import cv2
from ultralytics import YOLO
import os

# Configuration
VIDEO_PATH = "backend/data/raw_videos/video1.mp4"  # Place your video file here
OUTPUT_PATH = "backend/results/video1_annotated.mp4"
MODEL1_PATH = "backend/src/models/weights/03102025_best.pt"  # Ball/Rim detection
MODEL2_PATH = "backend/src/models/weights/04242025_shot_detector.pt"  # Person/Shoot detection

# Detection confidence thresholds
CONF_THRESHOLD = 0.3  # Lower threshold to see more detections

print("="*70)
print("BASKETBALL DETECTION - VIDEO ANNOTATION")
print("="*70)

# Load models
print("\n[1/4] Loading models...")
print(f"  Model 1 (Ball/Rim): {MODEL1_PATH}")
print(f"  Model 2 (Person/Shoot): {MODEL2_PATH}")
model1 = YOLO(MODEL1_PATH)
model2 = YOLO(MODEL2_PATH)
print("  [OK] Models loaded")

# Open video
print(f"\n[2/4] Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"  [ERROR] Could not open video: {VIDEO_PATH}")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.1f} seconds")

# Setup video writer
print(f"\n[3/4] Setting up output video: {OUTPUT_PATH}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print("  [ERROR] Could not create output video")
    cap.release()
    exit(1)
print("  [OK] Output video ready")

# Color scheme for different classes
COLORS = {
    'ball': (0, 165, 255),      # Orange (BGR)
    'rim': (255, 0, 0),          # Blue
    'person': (255, 0, 255),     # Magenta
    'shoot': (0, 0, 255),        # Red
}

# Process video
print(f"\n[4/4] Processing video...")
print(f"  Confidence threshold: {CONF_THRESHOLD}")
print(f"  Processing {total_frames} frames...")

frame_count = 0
detections_count = {'ball': 0, 'rim': 0, 'person': 0, 'shoot': 0}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run Model 1 (Ball/Rim)
    results1 = model1(frame, verbose=False, conf=CONF_THRESHOLD)

    # Run Model 2 (Person/Shoot) - sample every 5 frames for speed
    if frame_count % 5 == 0:
        results2 = model2(frame, verbose=False, conf=CONF_THRESHOLD)
    else:
        results2 = None

    # Draw detections from Model 1
    for result in results1:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]

            if class_name in COLORS:
                color = COLORS[class_name]
                detections_count[class_name] += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label with confidence
                label = f"{class_name}: {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw detections from Model 2
    if results2 is not None:
        for result in results2:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                if class_name in COLORS:
                    color = COLORS[class_name]
                    detections_count[class_name] += 1

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add frame counter and detection info overlay
    info_text = f"Frame: {frame_count}/{total_frames}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add legend
    legend_y = height - 120
    cv2.putText(frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25
    for i, (class_name, color) in enumerate(COLORS.items()):
        cv2.rectangle(frame, (10, legend_y + i*25 - 15), (30, legend_y + i*25 - 5), color, -1)
        cv2.putText(frame, class_name.capitalize(), (35, legend_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write frame
    out.write(frame)

    # Progress indicator
    if frame_count % 100 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")

# Cleanup
cap.release()
out.release()

print(f"\n  [OK] Processed {frame_count} frames")

print("\n" + "="*70)
print("ANNOTATION COMPLETE")
print("="*70)
print(f"Output saved to: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.1f} MB")

print(f"\nTotal Detections:")
print(f"  Ball:   {detections_count['ball']}")
print(f"  Rim:    {detections_count['rim']}")
print(f"  Person: {detections_count['person']}")
print(f"  Shoot:  {detections_count['shoot']}")

print("\nYou can now watch the annotated video to see all detections!")
print("="*70)
