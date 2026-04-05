"""
Diagnostic script to test court localization.
Tests homography accuracy and zone classification.
"""

import sys
import json
import numpy as np
import cv2
from pathlib import Path
from collections import Counter

sys.path.insert(0, 'backend/score_detection')

from constants import (
    COURT_WIDTH, COURT_HALF_LENGTH, BASKET_Y,
    CALIBRATION_LABELS, CALIBRATION_POINTS_FIBA
)

CALIBRATION_FILE = 'court_calibration_6pt.json'
VIDEO_PATH = 'backend/score_detection/video1.mp4'


def compute_homography(src_pts, dst_pts, method_name, method=cv2.RANSAC, threshold=0.1):
    H, status = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        method=method,
        ransacReprojThreshold=threshold
    )
    if status is not None:
        inliers = int(status.sum())
    else:
        inliers = len(src_pts)
    return H, inliers


def validate_homography(H, src_pts, dst_pts, label=''):
    errors = []
    for orig, expected in zip(src_pts, dst_pts):
        pt = np.array([[orig]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), H)[0][0]
        err = np.sqrt((mapped[0] - expected[0])**2 + (mapped[1] - expected[1])**2)
        errors.append(err)
    avg = np.mean(errors)
    max_e = np.max(errors)
    print(f"  [{label}] avg={avg:.3f}m  max={max_e:.3f}m  per-point: {[f'{e:.3f}' for e in errors]}")
    return H, errors


def map_point(H, px, py):
    pt = np.array([[[px, py]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(pt, H)[0][0]
    return float(mapped[0]), float(mapped[1])


def test_zone_classification(H):
    from statistics import TeamStatistics
    stats = TeamStatistics(quarters=[float('inf')])

    print("\n=== Zone Classification at Known Court Positions ===")
    test_pts = [
        ("Under basket",         7.5,  1.5,  2),
        ("Left corner 3",        0.5,  0.5,  8),
        ("Right corner 3",      14.5,  0.5,  9),
        ("Top of key",           7.5,  6.5,  6),
        ("Left wing 3",          1.0,  5.0,  5),
        ("Right wing 3",        14.0,  5.0,  7),
        ("Free throw line",      7.5,  5.8,  4),
        ("Left mid-range",       5.0,  2.5,  1),
        ("Right mid-range",     10.0,  2.5,  3),
    ]
    all_correct = True
    for label, cx, cy, expected_zone in test_pts:
        x_centered = cx - COURT_WIDTH / 2
        zone = stats.determine_zone(x_centered, cy)
        dist = np.sqrt(x_centered**2 + (cy - BASKET_Y)**2)
        ok = zone == expected_zone
        if not ok:
            all_correct = False
        print(f"  {'OK' if ok else 'FAIL'} {label:25s}: zone={zone} (expected {expected_zone})  dist={dist:.2f}m  x_c={x_centered:.1f}")
    return all_correct


def test_video_localization(H, cal_dims, video_path, max_frames=300, sample_every=15):
    from ultralytics import YOLO
    from statistics import TeamStatistics
    import yaml

    stats = TeamStatistics(quarters=[float('inf')])
    config = yaml.safe_load(open('backend/score_detection/config.yaml'))
    weights = 'backend/score_detection/' + config['weights_path']
    print(f"\nLoading model: {weights}")
    model = YOLO(weights, verbose=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cal_w, cal_h = cal_dims

    print(f"\nVideo: {vid_w}x{vid_h}  |  Calibration: {cal_w}x{cal_h}")

    # Scale factors to convert video pixel coords → calibration pixel coords
    sx = cal_w / vid_w
    sy = cal_h / vid_h
    print(f"Scale factors for foot position: sx={sx:.3f} sy={sy:.3f}")

    results_list = []
    frame_idx = 0
    while frame_idx < min(total, max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            yolo_results = model(frame, verbose=False)[0]
            persons = []
            for box in yolo_results.boxes:
                name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if name == 'person' and conf > 0.4:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    persons.append((x1, y1, x2, y2))

            for i, (x1, y1, x2, y2) in enumerate(persons[:3]):
                foot_vid = ((x1 + x2) / 2, y2)
                # Scale foot position to calibration coordinate space
                foot_cal = (foot_vid[0] * sx, foot_vid[1] * sy)
                cx, cy = map_point(H, foot_cal[0], foot_cal[1])
                in_bounds = (0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_HALF_LENGTH)
                x_centered = cx - COURT_WIDTH / 2
                zone = stats.determine_zone(x_centered, cy) if in_bounds else None
                results_list.append({'in_bounds': in_bounds, 'zone': zone,
                                     'cx': cx, 'cy': cy, 'frame': frame_idx})
                if frame_idx % 90 == 0 and i == 0:
                    print(f"  Frame {frame_idx:3d} person {i+1}: "
                          f"foot_vid=({foot_vid[0]:.0f},{foot_vid[1]:.0f}) "
                          f"→ foot_cal=({foot_cal[0]:.0f},{foot_cal[1]:.0f}) "
                          f"→ court=({cx:.2f},{cy:.2f})m  "
                          f"{'IN' if in_bounds else 'OUT'} zone={zone}")
        frame_idx += 1

    cap.release()

    if results_list:
        in_bounds = sum(1 for r in results_list if r['in_bounds'])
        total_r = len(results_list)
        zones = [r['zone'] for r in results_list if r['zone'] is not None]
        print(f"\n=== Summary ===")
        print(f"  Persons mapped: {total_r}")
        print(f"  In bounds:  {in_bounds} ({100*in_bounds/total_r:.1f}%)")
        print(f"  Out of bounds: {total_r-in_bounds} ({100*(total_r-in_bounds)/total_r:.1f}%)")
        if zones:
            print(f"  Zone distribution: {dict(Counter(zones))}")


if __name__ == '__main__':
    print("=== Court Localization Diagnostic ===\n")

    with open(CALIBRATION_FILE) as f:
        data = json.load(f)
    cal_pts = data['points']
    cal_dims = data['image_dimensions']  # [width, height]

    print(f"Calibration image: {cal_dims[0]}x{cal_dims[1]}")
    print("Calibration points:")
    for i, (pt, label) in enumerate(zip(cal_pts, CALIBRATION_LABELS)):
        print(f"  {i+1}. {label:35s}: pixel=({pt[0]:.0f},{pt[1]:.0f})  →  court={CALIBRATION_POINTS_FIBA[i]}")

    # Try different homography methods
    print("\n=== Homography Method Comparison ===")
    H_ransac, inliers = compute_homography(cal_pts, CALIBRATION_POINTS_FIBA, 'RANSAC', cv2.RANSAC, 0.1)
    print(f"RANSAC (threshold=0.1m)  inliers={inliers}/{len(cal_pts)}")
    validate_homography(H_ransac, cal_pts, CALIBRATION_POINTS_FIBA, 'RANSAC')

    H_lmeds, inliers_lm = compute_homography(cal_pts, CALIBRATION_POINTS_FIBA, 'LMEDS', cv2.LMEDS, 0)
    print(f"\nLMEDS  inliers≈{inliers_lm}/{len(cal_pts)}")
    validate_homography(H_lmeds, cal_pts, CALIBRATION_POINTS_FIBA, 'LMEDS')

    H_dlt, _ = compute_homography(cal_pts, CALIBRATION_POINTS_FIBA, 'DLT', 0, 0)
    print(f"\nDLT (all points, no RANSAC)")
    validate_homography(H_dlt, cal_pts, CALIBRATION_POINTS_FIBA, 'DLT')

    # Use best homography (DLT uses all points)
    print("\n** Using DLT homography for remaining tests **")
    H = H_dlt

    # Test zone classification
    zones_ok = test_zone_classification(H)

    # Test on video
    if Path(VIDEO_PATH).exists():
        test_video_localization(H, cal_dims, VIDEO_PATH)
    else:
        print(f"\nVideo not found: {VIDEO_PATH}")
        for v in ['backend/score_detection/shots.mp4', 'backend/score_detection/shot.mp4']:
            if Path(v).exists():
                test_video_localization(H, cal_dims, v)
                break
