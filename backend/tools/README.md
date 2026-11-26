# Offline tools and diagnostics

[Project overview](../../README.md) · [Setup and usage](../../INSTRUCTION.md)

Run these tools with the backend Python dependencies installed. Commands below start from the repository root. You can also invoke a script by absolute path from another directory: imports and bundled model/calibration paths resolve relative to the repository, while explicitly supplied input paths and generated outputs retain their working-directory behavior.

## Localization

| Script | Purpose |
| --- | --- |
| [static_shot_localization.py](localization/static_shot_localization.py) | Interactive calibration and shooter localization on an image |
| [video_shot_localization.py](localization/video_shot_localization.py) | Offline video shot detection and court localization |
| [static_court_keypoint_detection.py](localization/static_court_keypoint_detection.py) | Explore learned court keypoints or use them for homography estimation |

```bash
python backend/tools/localization/static_shot_localization.py --image path/to/shot.jpg
python backend/tools/localization/video_shot_localization.py --video path/to/game.mp4
python backend/tools/localization/static_court_keypoint_detection.py --image path/to/shot.jpg --mode explore --save
```

Use `--help` for each tool's options. Interactive calibration/display requires a graphical desktop. The image and video localizers share the existing [court_calibration_6pt.json](../../court_calibration_6pt.json); interactive recalibration can update that file. Check that the saved calibration belongs to your camera view before reusing it.

Model defaults remain in `backend/score_detection/weights/`. Explicit `--model`, `--shot-model`, and `--mapping` paths are resolved from your current working directory.

## Diagnostics

```bash
python backend/tools/diagnostics/verify_coordinates.py
python backend/tools/diagnostics/test_localization.py
```

[verify_coordinates.py](diagnostics/verify_coordinates.py) prints coordinate-conversion checks without model dependencies. [test_localization.py](diagnostics/test_localization.py) is a standalone diagnostic, not a pytest suite: it compares homographies using the saved calibration and checks zone classification. Its optional video stage looks for `video1.mp4`, `shots.mp4`, or `shot.mp4` in `backend/score_detection/` and requires the configured detector weights.

The existing zone diagnostic reports a mismatch for its “Top of key” sample: it expects zone 6 at `(7.5, 6.5)`, while the classifier returns zone 4. This behavior predates the script reorganization.

## Visualization experiment

[visualize_clustering.py](visualization/visualize_clustering.py) preserves an older HSV clustering experiment. It expects `backend/sample_videos/video1.mp4` and the root `yolov8n.pt`, and writes `clustering_rationale.png` and `team_clustering_output.mp4` in the current working directory.

This experiment interprets color features as HSV, while the current detector returns CIELAB features; its color calculations need adaptation before its results can be relied on. For current team-clustering work, use [backend/run_team_clustering.py](../run_team_clustering.py) and the annotation/evaluation tools below.

## Team annotation and evaluation

- [annotate_teams.py](annotate_teams.py): manually label saved player crops for team evaluation.
- [evaluate_teams.py](evaluate_teams.py): evaluate team-color classification against annotations.

Their locations are unchanged. Run either with `--help` to select the debug-frame directory and other inputs.
