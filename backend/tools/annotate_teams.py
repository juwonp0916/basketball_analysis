#!/usr/bin/env python3
"""
Simple annotation CLI tool for labeling team assignments in debug frames.

Displays each debug frame with player bounding boxes, and prompts the user
to label each player as Team 0, Team 1, or skip. Writes annotations back
into the JSON sidecar files' ground_truth_team fields.

Usage:
    python annotate_teams.py [--dir <debug_shot_frames_dir>]

Controls (per player):
    0 - Assign to Team 0
    1 - Assign to Team 1
    s - Skip this player
    q - Quit annotation (saves progress)
    n - Skip to next shot frame

Example:
    python annotate_teams.py --dir ../../debug_shot_frames
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: opencv-python and numpy are required. Install with:")
    print("  pip install opencv-python numpy")
    sys.exit(1)


def load_debug_frames(directory: Path) -> List[Dict]:
    """Load all debug shot JSON files sorted by shot_id."""
    jsons = sorted(directory.glob("debug_shot_*.json"))
    frames = []
    for jpath in jsons:
        with open(jpath) as f:
            data = json.load(f)
        data["_json_path"] = str(jpath)

        # Find matching image
        stem = jpath.stem
        img_path = jpath.parent / f"{stem}.jpg"
        if img_path.exists():
            data["_img_path"] = str(img_path)
        else:
            data["_img_path"] = None
        frames.append(data)
    return frames


def count_annotated(data: Dict) -> tuple:
    """Count (annotated, total) players in a shot frame."""
    persons = data.get("persons_detected", [])
    annotated = sum(1 for p in persons if p.get("ground_truth_team") is not None)
    return annotated, len(persons)


def annotate_frame(data: Dict) -> Optional[str]:
    """
    Display a shot frame and annotate each player.

    Returns:
        'continue' to proceed to next frame,
        'quit' to stop annotation,
        None on error.
    """
    img_path = data.get("_img_path")
    if not img_path:
        print(f"  [SKIP] No image found for shot {data.get('shot_id', '?')}")
        return "continue"

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] Could not read image: {img_path}")
        return "continue"

    persons = data.get("persons_detected", [])
    if not persons:
        print(f"  [SKIP] No persons detected in shot {data.get('shot_id', '?')}")
        return "continue"

    shot_id = data.get("shot_id", "?")
    team_colors_info = data.get("team_colors", data.get("team_colors_hex", {}))
    hex0 = team_colors_info.get("team_0", {}).get("hex", "?") if isinstance(team_colors_info, dict) and "team_0" in team_colors_info else team_colors_info.get("team0", "?")
    hex1 = team_colors_info.get("team_1", {}).get("hex", "?") if isinstance(team_colors_info, dict) and "team_1" in team_colors_info else team_colors_info.get("team1", "?")

    print(f"\n--- Shot #{shot_id} ---")
    print(f"  Team 0: {hex0}  |  Team 1: {hex1}")
    print(f"  Players: {len(persons)}")

    for i, person in enumerate(persons):
        existing_gt = person.get("ground_truth_team")
        if existing_gt is not None:
            print(f"  Player {i}: already annotated as Team {existing_gt} (skipping)")
            continue

        bbox = person.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        tid = person.get("team_id")
        conf = person.get("confidence", 0)
        is_shooter = person.get("is_shooter", False)
        extracted_hex = person.get("extracted_hex", "?")

        # Draw the frame with this player highlighted
        vis = img.copy()

        # Draw all player bboxes dimly
        for j, other in enumerate(persons):
            ox1, oy1, ox2, oy2 = other.get("bbox", [0, 0, 0, 0])
            color = (80, 80, 80) if j != i else (0, 255, 255)  # Yellow for current
            thick = 2 if j != i else 3
            cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), color, thick)
            if j == i:
                cv2.putText(
                    vis, f">>> Player {i} <<<", (ox1, oy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

        # Show instructions overlay
        instructions = [
            f"Player {i}/{len(persons)-1} | Predicted: Team {tid} ({conf:.0%}) | Color: {extracted_hex}",
            "Press: 0=Team0  1=Team1  s=Skip  n=Next shot  q=Quit",
        ]
        for idx, text in enumerate(instructions):
            y_pos = 30 + idx * 30
            cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Team Annotation", vis)

        role = " [SHOOTER]" if is_shooter else ""
        print(
            f"  Player {i}{role}: predicted Team {tid} ({conf:.0%}), "
            f"color={extracted_hex}, bbox={bbox}"
        )

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("0"):
                person["ground_truth_team"] = 0
                print(f"    -> Labeled Team 0")
                break
            elif key == ord("1"):
                person["ground_truth_team"] = 1
                print(f"    -> Labeled Team 1")
                break
            elif key == ord("s"):
                print(f"    -> Skipped")
                break
            elif key == ord("n"):
                print(f"    -> Skipping to next shot")
                return "continue"
            elif key == ord("q"):
                return "quit"

    return "continue"


def save_annotations(data: Dict) -> None:
    """Write the annotations back to the JSON file."""
    json_path = data["_json_path"]
    # Remove internal fields before saving
    save_data = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate team assignments in debug shot frames."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "debug_shot_frames"),
        help="Path to debug_shot_frames directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-annotate players that already have ground truth labels",
    )

    args = parser.parse_args()
    debug_dir = Path(args.dir)

    if not debug_dir.exists():
        print(f"Error: Directory not found: {debug_dir}", file=sys.stderr)
        sys.exit(1)

    frames = load_debug_frames(debug_dir)
    if not frames:
        print("No debug shot frames found.", file=sys.stderr)
        sys.exit(1)

    # If overwrite, clear existing annotations
    if args.overwrite:
        for frame in frames:
            for person in frame.get("persons_detected", []):
                person["ground_truth_team"] = None

    # Summary
    total_persons = sum(len(f.get("persons_detected", [])) for f in frames)
    annotated = sum(
        sum(1 for p in f.get("persons_detected", []) if p.get("ground_truth_team") is not None)
        for f in frames
    )
    print(f"Found {len(frames)} shot frames with {total_persons} total players.")
    print(f"Already annotated: {annotated}/{total_persons}")
    print(f"\nControls: 0=Team0, 1=Team1, s=Skip, n=Next shot, q=Quit\n")

    for frame in frames:
        a, t = count_annotated(frame)
        if a == t and not args.overwrite:
            continue  # All players already annotated

        result = annotate_frame(frame)
        save_annotations(frame)
        print(f"  [SAVED] {frame['_json_path']}")

        if result == "quit":
            break

    cv2.destroyAllWindows()

    # Final summary
    final_annotated = sum(
        sum(1 for p in f.get("persons_detected", []) if p.get("ground_truth_team") is not None)
        for f in frames
    )
    print(f"\nDone. Annotated {final_annotated}/{total_persons} players across {len(frames)} frames.")
    print("Run evaluate_teams.py to compute accuracy metrics.")


if __name__ == "__main__":
    main()
