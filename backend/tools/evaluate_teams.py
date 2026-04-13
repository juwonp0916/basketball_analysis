#!/usr/bin/env python3
"""
Offline evaluation script for team differentiation accuracy.

Reads annotated debug JSON sidecar files (where ground_truth_team has been
filled in by the annotation tool) and computes:
  - Classification accuracy (overall + per-team)
  - Color fidelity (mean Delta-E between extracted color and ground truth jersey color)
  - Inter-team separation (Delta-E between team centroids)
  - Confidence calibration (accuracy binned by confidence level)
  - Confusion matrix (2x2)

Usage:
    python evaluate_teams.py [--dir <debug_shot_frames_dir>] [--gt-colors <hex0> <hex1>]

Example:
    python evaluate_teams.py --dir ../../debug_shot_frames --gt-colors "#000000" "#003399"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np


def hex_to_lab(hex_color: str) -> np.ndarray:
    """Convert hex color string to OpenCV LAB."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0][0].astype(np.float64)


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76 Delta-E between two LAB colors."""
    diff = lab1.astype(np.float64) - lab2.astype(np.float64)
    return float(np.sqrt(np.sum(diff ** 2)))


def load_annotated_jsons(directory: Path) -> List[Dict]:
    """Load all debug JSON files that have at least one annotated player."""
    jsons = sorted(directory.glob("debug_shot_*.json"))
    annotated = []
    for jpath in jsons:
        with open(jpath) as f:
            data = json.load(f)
        # Check if any player has ground_truth_team set
        has_gt = any(
            p.get("ground_truth_team") is not None
            for p in data.get("persons_detected", [])
        )
        if has_gt:
            data["_json_path"] = str(jpath)
            annotated.append(data)
    return annotated


def evaluate(
    data: List[Dict],
    gt_team0_hex: Optional[str] = None,
    gt_team1_hex: Optional[str] = None,
) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        data: List of annotated debug JSON dicts.
        gt_team0_hex: Ground truth hex color for team 0 (for color fidelity).
        gt_team1_hex: Ground truth hex color for team 1 (for color fidelity).

    Returns:
        Dictionary of metrics.
    """
    # Collect predictions
    predictions: List[Tuple[int, int, float]] = []  # (predicted, ground_truth, confidence)
    extracted_labs_by_team: Dict[int, List[np.ndarray]] = {0: [], 1: []}

    for shot in data:
        for person in shot.get("persons_detected", []):
            gt = person.get("ground_truth_team")
            pred = person.get("team_id")
            conf = person.get("confidence", 0.0)

            if gt is None or pred is None:
                continue

            predictions.append((pred, gt, conf))

            if person.get("extracted_lab") is not None:
                extracted_labs_by_team[gt].append(np.array(person["extracted_lab"]))

    if not predictions:
        return {"error": "No annotated predictions found."}

    preds = np.array([(p, g) for p, g, _ in predictions])
    confs = np.array([c for _, _, c in predictions])

    # --- Classification accuracy ---
    correct = preds[:, 0] == preds[:, 1]
    accuracy = float(np.mean(correct))

    # Per-team accuracy
    team_accuracy = {}
    for t in [0, 1]:
        mask = preds[:, 1] == t
        if mask.sum() > 0:
            team_accuracy[f"team_{t}"] = float(np.mean(correct[mask]))
        else:
            team_accuracy[f"team_{t}"] = None

    # --- Confusion matrix ---
    # Rows = ground truth, Cols = predicted
    cm = np.zeros((2, 2), dtype=int)
    for pred, gt in preds:
        cm[gt][pred] += 1

    # --- Confidence calibration ---
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    calibration = []
    for low, high in bins:
        mask = (confs >= low) & (confs < high)
        if mask.sum() > 0:
            calibration.append({
                "confidence_range": f"[{low:.2f}, {high:.2f})",
                "count": int(mask.sum()),
                "accuracy": float(np.mean(correct[mask])),
                "mean_confidence": float(np.mean(confs[mask])),
            })

    # --- Inter-team separation ---
    inter_team_delta_es = []
    for shot in data:
        de = shot.get("inter_team_delta_e")
        if de is not None:
            inter_team_delta_es.append(de)

    # --- Color fidelity (Delta-E vs ground truth jersey colors) ---
    color_fidelity = {}
    gt_labs = {}
    if gt_team0_hex:
        gt_labs[0] = hex_to_lab(gt_team0_hex)
    if gt_team1_hex:
        gt_labs[1] = hex_to_lab(gt_team1_hex)

    for t in [0, 1]:
        if t in gt_labs and extracted_labs_by_team[t]:
            deltas = [delta_e(lab, gt_labs[t]) for lab in extracted_labs_by_team[t]]
            color_fidelity[f"team_{t}"] = {
                "mean_delta_e": float(np.mean(deltas)),
                "median_delta_e": float(np.median(deltas)),
                "max_delta_e": float(np.max(deltas)),
                "count": len(deltas),
                "gt_hex": gt_team0_hex if t == 0 else gt_team1_hex,
            }

    results = {
        "total_predictions": len(predictions),
        "classification_accuracy": round(accuracy, 4),
        "per_team_accuracy": team_accuracy,
        "confusion_matrix": {
            "rows_are_ground_truth": True,
            "cols_are_predicted": True,
            "matrix": cm.tolist(),
        },
        "confidence_calibration": calibration,
        "inter_team_separation": {
            "mean_delta_e": float(np.mean(inter_team_delta_es)) if inter_team_delta_es else None,
            "min_delta_e": float(np.min(inter_team_delta_es)) if inter_team_delta_es else None,
            "count": len(inter_team_delta_es),
        },
        "color_fidelity": color_fidelity if color_fidelity else None,
    }

    return results


def print_report(results: Dict) -> None:
    """Print a human-readable evaluation report."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("=" * 60)
    print("  TEAM DIFFERENTIATION EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal annotated predictions: {results['total_predictions']}")
    print(f"Overall accuracy: {results['classification_accuracy']:.1%}")

    pa = results["per_team_accuracy"]
    for team, acc in pa.items():
        if acc is not None:
            print(f"  {team}: {acc:.1%}")

    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    cm = results["confusion_matrix"]["matrix"]
    print(f"           Pred 0   Pred 1")
    print(f"  GT 0:   {cm[0][0]:>6d}   {cm[0][1]:>6d}")
    print(f"  GT 1:   {cm[1][0]:>6d}   {cm[1][1]:>6d}")

    print("\nConfidence Calibration:")
    for bin_info in results["confidence_calibration"]:
        print(
            f"  {bin_info['confidence_range']}: "
            f"n={bin_info['count']}, "
            f"accuracy={bin_info['accuracy']:.1%}, "
            f"mean_conf={bin_info['mean_confidence']:.2f}"
        )

    sep = results["inter_team_separation"]
    if sep["mean_delta_e"] is not None:
        print(f"\nInter-team Separation:")
        print(f"  Mean Delta-E: {sep['mean_delta_e']:.1f}")
        print(f"  Min Delta-E:  {sep['min_delta_e']:.1f}")

    cf = results.get("color_fidelity")
    if cf:
        print("\nColor Fidelity (vs ground truth jersey colors):")
        for team, info in cf.items():
            print(
                f"  {team} (GT: {info['gt_hex']}): "
                f"mean={info['mean_delta_e']:.1f}, "
                f"median={info['median_delta_e']:.1f}, "
                f"max={info['max_delta_e']:.1f} "
                f"(n={info['count']})"
            )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate team differentiation accuracy from annotated debug frames."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "debug_shot_frames"),
        help="Path to debug_shot_frames directory",
    )
    parser.add_argument(
        "--gt-colors",
        nargs=2,
        metavar=("TEAM0_HEX", "TEAM1_HEX"),
        help='Ground truth jersey hex colors, e.g. --gt-colors "#000000" "#003399"',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable report",
    )

    args = parser.parse_args()
    debug_dir = Path(args.dir)

    if not debug_dir.exists():
        print(f"Error: Directory not found: {debug_dir}", file=sys.stderr)
        sys.exit(1)

    data = load_annotated_jsons(debug_dir)
    if not data:
        print(
            "No annotated JSON files found. Run annotate_teams.py first to add "
            "ground_truth_team labels.",
            file=sys.stderr,
        )
        sys.exit(1)

    gt0, gt1 = (args.gt_colors[0], args.gt_colors[1]) if args.gt_colors else (None, None)
    results = evaluate(data, gt0, gt1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)


if __name__ == "__main__":
    main()
