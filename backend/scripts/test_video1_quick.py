#!/usr/bin/env python3
"""
Quick test script for video1.mp4 - processes first 300 frames only
"""

import cv2
import os
from ultralytics import YOLO
import time

# Paths
video_path = "backend/data/raw_videos/video1.mp4"  # Place your video file here
model1_path = "backend/src/models/weights/04242025_shot_detector.pt"
model2_path = "backend/src/models/weights/03102025_best.pt"

print("="*60)
print("QUICK TEST: video1.mp4 (First 300 frames)")
print("="*60)

# Check if video exists
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    exit(1)

# Check if models exist
if not os.path.exists(model1_path):
    print(f"ERROR: Model 1 not found at {model1_path}")
    exit(1)
if not os.path.exists(model2_path):
    print(f"ERROR: Model 2 not found at {model2_path}")
    exit(1)

print(f"\nVideo: {video_path}")
print(f"Model 1: {model1_path}")
print(f"Model 2: {model2_path}")

# Load models
print("\nLoading models...")
model1 = YOLO(model1_path)
print(f"  Model 1 loaded: {model1.names}")
model2 = YOLO(model2_path)
print(f"  Model 2 loaded: {model2.names}")

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Could not open video")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nVideo Info:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps:.2f}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {int(total_frames/fps)} seconds")

# Statistics
stats = {
    'frames_processed': 0,
    'ball_detected': 0,
    'rim_detected': 0,
    'both_detected': 0,
    'person_detected': 0,
    'shoot_detected': 0,
    'ball_confs': [],
    'rim_confs': [],
}

# Process first 300 frames
max_frames = min(300, total_frames)
print(f"\nProcessing first {max_frames} frames (10 seconds)...")
print("-"*60)

start_time = time.time()
frame_count = 0

try:
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Model 1: Ball/Rim detection
        results1 = model1(frame, verbose=False)

        ball_found = False
        rim_found = False

        for r in results1:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model1.names[cls]

                if class_name == 'ball' and conf > 0.5:
                    ball_found = True
                    stats['ball_confs'].append(conf)
                elif class_name == 'rim' and conf > 0.5:
                    rim_found = True
                    stats['rim_confs'].append(conf)

        if ball_found:
            stats['ball_detected'] += 1
        if rim_found:
            stats['rim_detected'] += 1
        if ball_found and rim_found:
            stats['both_detected'] += 1

        # Model 2: Sample every 5 frames
        if frame_count % 5 == 0:
            results2 = model2(frame, verbose=False)

            for r in results2:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model2.names[cls]

                    if class_name == 'person' and conf > 0.5:
                        stats['person_detected'] += 1
                    elif class_name == 'shoot' and conf > 0.5:
                        stats['shoot_detected'] += 1

        frame_count += 1
        stats['frames_processed'] = frame_count

        # Progress
        if frame_count % 30 == 0:
            progress = frame_count / max_frames * 100
            print(f"  Frame {frame_count}/{max_frames} ({progress:.1f}%)", end='\r')

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    cap.release()

elapsed = time.time() - start_time
print(f"\n\nProcessed {frame_count} frames in {elapsed:.1f} seconds ({frame_count/elapsed:.1f} FPS)")

# Print results
print("\n" + "="*60)
print("DETECTION RESULTS (First 300 frames)")
print("="*60)

total = stats['frames_processed']
print(f"\nFrames Processed: {total}")

print(f"\nDetection Rates:")
print(f"  Ball detected:  {stats['ball_detected']:4d} / {total} ({stats['ball_detected']/total*100:.1f}%)")
print(f"  Rim detected:   {stats['rim_detected']:4d} / {total} ({stats['rim_detected']/total*100:.1f}%)")
print(f"  Both detected:  {stats['both_detected']:4d} / {total} ({stats['both_detected']/total*100:.1f}%)")
print(f"  Person (sampled): {stats['person_detected']:4d} / {total//5} ({stats['person_detected']/(total//5)*100:.1f}%)")
print(f"  Shoot (sampled):  {stats['shoot_detected']:4d} / {total//5} ({stats['shoot_detected']/(total//5)*100:.1f}%)")

if stats['ball_confs']:
    print(f"\nAverage Confidence:")
    print(f"  Ball: {sum(stats['ball_confs'])/len(stats['ball_confs']):.3f}")
if stats['rim_confs']:
    print(f"  Rim:  {sum(stats['rim_confs'])/len(stats['rim_confs']):.3f}")

# Recommendations
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

ball_rate = stats['ball_detected'] / total if total > 0 else 0
rim_rate = stats['rim_detected'] / total if total > 0 else 0
both_rate = stats['both_detected'] / total if total > 0 else 0

print(f"\nThis is a {int(total/fps)}-second sample from your amateur half-court video.")

if rim_rate > 0.9:
    print(f"✅ Excellent rim detection ({rim_rate*100:.1f}%)")
    print("   Fixed camera setup is working well!")
else:
    print(f"⚠ Rim detection could be better ({rim_rate*100:.1f}%)")

if ball_rate > 0.7:
    print(f"✅ Good ball detection ({ball_rate*100:.1f}%)")
else:
    print(f"⚠ Ball detection needs improvement ({ball_rate*100:.1f}%)")
    print("   Fine-tuning recommended for amateur video quality")

if both_rate > 0.6:
    print(f"✅ Good simultaneous detection ({both_rate*100:.1f}%)")
else:
    print(f"⚠ Simultaneous detection needs improvement ({both_rate*100:.1f}%)")

print("\nNext Steps:")
if ball_rate < 0.7 or both_rate < 0.6:
    print("1. Proceed with dataset collection (see Week 2 in SETUP_AND_QUICKSTART.md)")
    print("2. Target: Extract ~10,000 frames from your amateur videos")
    print("3. Fine-tune Model 1 to improve ball detection on amateur footage")
    print("4. Expected improvement: +20-25% detection rate")
else:
    print("1. Current weights performing well on your amateur video")
    print("2. Fine-tuning will provide incremental improvements")
    print("3. Consider testing on more videos to confirm")

print("\n" + "="*60)
