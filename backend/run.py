import cv2
import json
import yaml
import os

from shot_detector import ShotDetector
from score_detector import ScoreDetector
from team_detector import TeamDetector


def process_video(video_path, output_path, config, device,
                  shot_detector, score_detector, team_detector):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    # Reset shot detector state for new video
    shot_detector.set_fps(fps)
    shot_detector.ball_positions = []
    shot_detector.rim_positions = []
    shot_detector.person_boxes = []
    shot_detector.ball_entered = False
    shot_detector.last_point_in_region = None
    shot_detector.attempt_time = 0
    shot_detector.last_shot_frame = 0
    shot_detector.ball_velocity[:] = 0
    shot_detector.frames_since_ball_detection = 0
    shot_detector.predicted_ball_position = None

    # Video writer
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats
    team_stats = {0: {'makes': 0, 'attempts': 0}, 1: {'makes': 0, 'attempts': 0}}
    shot_log = []

    frame_count = 0
    attempt_cooldown = 0

    # Deferred score checking: after a shot triggers, keep checking for
    # "made" over the next score_check_window frames so we don't miss
    # the ball going through the net.
    score_check_window = int(fps * 2)  # 2 seconds
    pending_shot = None  # dict with shot info while waiting for score check

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps

        # 1. Shot detector: track ball, rim, person, shoot pose
        ball_detected, rim_detected, shoot_detected = \
            shot_detector.process_frame(frame, frame_count, device=device)

        # 2. Shot detector: check for shot attempt
        shot_data = shot_detector.detect_shot(
            frame_count, ball_detected, attempt_cooldown, shoot_detected
        )

        # If a new shot is detected, record it as pending
        if shot_data is not None:
            # If there's already a pending shot, finalize it first
            if pending_shot is not None:
                _finalize_shot(pending_shot, team_stats, shot_log)

            # Team detector: which team? (uses person boxes from shot_detector)
            rim_pos = shot_detector.rim_positions[-1] if shot_detector.rim_positions else None
            team, confidence = team_detector.detect(
                frame, shot_detector.person_boxes, rim_pos
            )

            pending_shot = {
                'shot_data': shot_data,
                'team': team,
                'confidence': confidence,
                'time': round(current_time, 2),
                'made': False,
                'frames_remaining': score_check_window,
                'trigger_frame': frame_count,
            }

            team_str = f"Team {team}" if team is not None else "Unknown"
            print(f"  [{current_time:.2f}s] {team_str} shot detected, checking score...")

            # Cooldown
            attempt_cooldown = shot_detector.shot_cooldown_frames

        # Check score on pending shot (run score model each frame during window)
        if pending_shot is not None and not pending_shot['made']:
            # Check this frame for "made"
            made = score_detector.detect(
                pending_shot['shot_data'], frame=frame, device=device
            )
            if made:
                pending_shot['made'] = True
                elapsed = current_time - pending_shot['time']
                print(f"    -> MADE detected (+{elapsed:.2f}s)")

            pending_shot['frames_remaining'] -= 1

            # Window expired — finalize
            if pending_shot['frames_remaining'] <= 0:
                _finalize_shot(pending_shot, team_stats, shot_log)
                pending_shot = None

        if attempt_cooldown > 0:
            attempt_cooldown -= 1

        # Visualization
        vis = frame.copy()

        # Ball
        if ball_detected and len(shot_detector.ball_positions) > 0:
            center, _, w, h, _ = shot_detector.ball_positions[-1]
            x, y = center[0] - w // 2, center[1] - h // 2
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, 'BALL', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif shot_detector.predicted_ball_position is not None:
            px, py = shot_detector.predicted_ball_position
            cv2.circle(vis, (px, py), 10, (0, 165, 255), 2)

        # Ball trajectory
        if len(shot_detector.ball_positions) > 1:
            points = [p[0] for p in shot_detector.ball_positions[-20:]]
            for i in range(1, len(points)):
                thickness = max(1, int(3 * i / len(points)))
                cv2.line(vis, points[i - 1], points[i], (100, 100, 100), thickness)

        # Rim
        if len(shot_detector.rim_positions) > 0:
            center, _, w, h, _ = shot_detector.rim_positions[-1]
            x, y = center[0] - w // 2, center[1] - h // 2
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(vis, 'RIM', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Stats overlay
        t0, t1 = team_stats[0], team_stats[1]
        cv2.putText(vis, f"Team 0: {t0['makes']}/{t0['attempts']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Team 1: {t1['makes']}/{t1['attempts']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        writer.write(vis)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)")

    # Finalize any remaining pending shot at end of video
    if pending_shot is not None:
        _finalize_shot(pending_shot, team_stats, shot_log)

    cap.release()
    writer.release()

    # Report
    print(f"\nResults for {os.path.basename(video_path)}:")
    for tid in [0, 1]:
        s = team_stats[tid]
        pct = (s['makes'] / s['attempts'] * 100) if s['attempts'] > 0 else 0
        print(f"  Team {tid}: {s['makes']}/{s['attempts']} ({pct:.1f}%)")
    print(f"Output saved: {output_path}\n")

    return {
        'video_name': os.path.basename(video_path),
        'total_shots': len(shot_log),
        'team_stats': team_stats,
        'shots': shot_log,
    }


def _finalize_shot(pending_shot, team_stats, shot_log):
    """Finalize a pending shot: update stats and log."""
    team = pending_shot['team']
    made = pending_shot['made']
    result = "MADE" if made else "MISSED"
    team_str = f"Team {team}" if team is not None else "Unknown"
    print(f"    -> Final: {team_str} {result} (conf: {pending_shot['confidence']:.2f})")

    shot_log.append({
        'time': pending_shot['time'],
        'team': team,
        'made': made,
        'confidence': pending_shot['confidence'],
    })
    if team is not None:
        team_stats[team]['attempts'] += 1
        if made:
            team_stats[team]['makes'] += 1


if __name__ == '__main__':
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config.get('device', 'cpu')

    # Initialize modules once (models loaded once, reused across videos)
    shot_detector = ShotDetector(config)
    score_detector = ScoreDetector(config)
    team_detector = TeamDetector('white', 'red')

    videos = [
        {'input': '../Test/video5.mp4', 'output': '../output/video5_out.mp4'},
        {'input': '../Test/video6.mp4', 'output': '../output/video6_out.mp4'},
        {'input': '../Test/video7.mp4', 'output': '../output/video7_out.mp4'},
    ]

    all_results = []

    for v in videos:
        # Delete existing output if present
        if os.path.exists(v['output']):
            os.remove(v['output'])
            print(f"Deleted existing: {v['output']}")

        print("=" * 60)
        result = process_video(
            v['input'], v['output'], config, device,
            shot_detector, score_detector, team_detector
        )
        if result:
            all_results.append(result)

    # Write JSON summary
    json_output = []
    for r in all_results:
        video_entry = {
            'video_name': r['video_name'],
            'number_of_shots': r['total_shots'],
            'shots': []
        }
        for shot in r['shots']:
            video_entry['shots'].append({
                'scored': shot['made'],
                'team': shot['team'],
                'time': shot['time'],
                'confidence': shot['confidence'],
            })
        json_output.append(video_entry)

    json_path = '../output/shot_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON results saved: {json_path}")
