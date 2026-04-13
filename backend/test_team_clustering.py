import yaml
from team_detector import StreamingTeamDetector
import cv2
import os
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

# Add score_detection to path to import team_detector
sys.path.append(os.path.join(os.path.dirname(__file__), "score_detection"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def hex_to_bgr(hex_color: str) -> tuple:
    """Convert #RRGGBB to BGR tuple for OpenCV"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


def main():
    base_dir = Path(__file__).parent
    videos_dir = base_dir / "sample_videos"
    output_dir = base_dir / "output_videos"
    output_dir.mkdir(exist_ok=True)

    config_path = base_dir / "score_detection" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    weights_path = base_dir / "score_detection" / config["weights_path"]
    model = YOLO(str(weights_path), verbose=False)
    class_names = config["classes"]
    device = config.get("device", "cpu")

    video_files = list(videos_dir.glob("*.mp4"))
    if not video_files:
        logging.warning(f"No mp4 videos found in {videos_dir}")
        return

    for video_path in video_files:
        logging.info(f"Processing {video_path.name}...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Failed to open {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        output_path = output_dir / f"{video_path.stem}_teams.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        team_detector = StreamingTeamDetector()

        # Inference dims - same as StreamingShotDetector
        inf_w = max(640, int(width * 0.5) // 32 * 32)
        inf_h = max(384, int(height * 0.5) // 32 * 32)
        inference_dims = (inf_w, inf_h)

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                logging.info(f"  Frame {frame_idx}...")

            if not team_detector.is_configured:
                # Accumulate features and try to calibrate
                team_detector.auto_calibrate(frame, model, class_names, inference_dims, device)

                # Draw calibration status
                cv2.putText(frame, "Calibrating Teams...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                # Every N frames, silent recheck to fix drift
                if frame_idx % 60 == 0:
                    team_detector.recheck(frame, model, class_names, inference_dims, device)

                # Get the team colors in BGR
                hex0, hex1 = team_detector.get_team_colors_hex()
                color0, color1 = hex_to_bgr(hex0), hex_to_bgr(hex1)

                # Detect persons directly to classify them for display
                det_frame = cv2.resize(frame, inference_dims)
                results = model(det_frame, stream=True, verbose=False, imgsz=inf_w, device=device, conf=0.3, max_det=30)

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls >= len(class_names) or class_names[cls] != 'person':
                            continue

                        if float(box.conf[0]) < 0.3:
                            continue

                        # Rescale coords
                        x1 = int(box.xyxy[0][0] * width / inf_w)
                        y1 = int(box.xyxy[0][1] * height / inf_h)
                        x2 = int(box.xyxy[0][2] * width / inf_w)
                        y2 = int(box.xyxy[0][3] * height / inf_h)

                        bbox = (x1, y1, x2, y2)

                        # Note: tracking ID is skipped here for simplicity, we classify frame-by-frame
                        team_id, confidence = team_detector.classify_from_bbox(frame, bbox)

                        if team_id is not None:
                            box_color = color0 if team_id == 0 else color1
                            label = f"Team {team_id} ({confidence:.2f})"
                        else:
                            box_color = (128, 128, 128)  # Gray for uncertain
                            label = "Unknown"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Draw legend
                cv2.rectangle(frame, (30, 30), (70, 70), color0, -1)
                cv2.putText(frame, f"Team 0", (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, color0, 2)

                cv2.rectangle(frame, (30, 90), (70, 130), color1, -1)
                cv2.putText(frame, f"Team 1", (80, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)

            out.write(frame)

        cap.release()
        out.release()
        logging.info(f"Finished {video_path.name}, saved to {output_path}")


if __name__ == "__main__":
    main()
