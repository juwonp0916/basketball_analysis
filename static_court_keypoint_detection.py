"""
Static Court Keypoint Detection Tool

Uses court_point.pt (YOLOv8n-pose) to detect basketball court keypoints,
then optionally computes a homography for mapping video pixels to FIBA court
coordinates.

Model attributes
----------------
  Task       : pose (keypoint detection)
  Class      : basketball_court (1 class)
  Keypoints  : 13 per detection, each (x, y, visibility)
  Architecture: YOLOv8n-pose, trained 100 epochs on basketball-court-keypoints-1
  mAP50(B)   : 0.946   mAP50(P): 0.972

Modes
-----
  explore   Visualize every detected keypoint labeled by index.
            Use this first to identify which index maps to which court landmark.

  localize  Compute a homography from detected keypoints + a configurable
            mapping, overlay FIBA court lines, and optionally localize a
            shooter if a shot-detection model is provided.

FIBA coordinate system
----------------------
  Origin at baseline-left corner
  X ∈ [0, 15] m  (left → right along baseline)
  Y ∈ [0, 14] m  (baseline → half-court line)
  Basket at (7.5, 1.575)

Usage
-----
  # Step 1: explore — identify keypoint indices
  python static_court_keypoint_detection.py --image shot.jpg --mode explore

  # Step 2: localize — verify / adjust DEFAULT_KEYPOINT_TO_FIBA below, then:
  python static_court_keypoint_detection.py --image shot.jpg --mode localize
  python static_court_keypoint_detection.py --image shot.jpg --mode localize \\
      --mapping my_mapping.json \\
      --shot-model backend/score_detection/weights/new_weight.pt

  # Save outputs automatically (skips interactive display):
  python static_court_keypoint_detection.py --image shot.jpg --mode explore --save
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, 'backend/score_detection')
from constants import (
    COURT_WIDTH, COURT_HALF_LENGTH,
    BASKET_X, BASKET_Y,
    FREE_THROW_LINE_Y,
    PENALTY_BOX_WIDTH,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT KEYPOINT → FIBA MAPPING
#
# court_point.pt detects 13 keypoints (indices 0–12) per detection.
# This mapping is an *initial guess* based on visual inspection; use
# --mode explore to confirm which index lands on which court feature,
# then edit this dict or pass --mapping <file.json>.
#
# Only keypoints listed here are used when computing the homography.
# You need at least 4 non-collinear pairs.  The 4 paint corners below
# (2 on the baseline + 2 on the FT line) are ideal: they are always
# non-collinear and are the most stably detected landmarks.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_KEYPOINT_TO_FIBA: dict[int, tuple[float, float]] = {
    # index: (court_x_m, court_y_m)
    7:  (5.05, 5.8),   # FT line — left paint corner
    8:  (9.95, 5.8),   # FT line — right paint corner
    10: (5.05, 0.0),   # Baseline — left paint corner
    11: (9.95, 0.0),   # Baseline — right paint corner
    # Uncomment / add extras below once you have confirmed their positions:
    # 9:  (7.5,  5.8),   # FT line — centre (top of key)
    # 3:  (7.5,  1.575), # Basket centre
    # 1:  (0.0,  0.0),   # Baseline left sideline corner
    # 12: (15.0, 0.0),   # Baseline right sideline corner
}

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for keypoint visualisation (BGR)
# ─────────────────────────────────────────────────────────────────────────────
_KP_COLORS = [
    (255,   0,   0),  #  0 blue
    (  0, 200,   0),  #  1 green
    (  0,   0, 255),  #  2 red
    (  0, 220, 220),  #  3 yellow-green
    (200,   0, 200),  #  4 magenta
    (  0, 200, 255),  #  5 orange-yellow
    (128,   0, 200),  #  6 purple
    (  0, 140, 255),  #  7 orange
    (255, 128,   0),  #  8 sky-blue
    ( 80, 255,  80),  #  9 lime
    (  0, 200, 100),  # 10 teal
    (255,  50, 150),  # 11 pink
    ( 80, 180,   0),  # 12 olive
]


# ─────────────────────────────────────────────────────────────────────────────
class CourtKeypointDetector:
    """Detects court keypoints and optionally computes a homography."""

    def __init__(
        self,
        image_path: str,
        court_model_path: str,
        conf: float = 0.3,
    ):
        self.image_path = image_path
        self.conf = conf

        print(f"\nLoading image: {image_path}")
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        self.h, self.w = self.image.shape[:2]
        print(f"Resolution: {self.w}x{self.h}")

        print(f"Loading court keypoint model: {court_model_path}")
        self.court_model = YOLO(court_model_path, verbose=False)
        print("  kpt_shape:", self.court_model.model.kpt_shape)

        self.keypoints: np.ndarray | None = None   # shape [13, 3] — x, y, vis
        self.H: np.ndarray | None = None

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_keypoints(self) -> np.ndarray | None:
        """
        Run court_point.pt on the image.

        Returns the keypoint array [13, 3] of the highest-confidence
        detection, or None if nothing detected above self.conf.
        """
        results = self.court_model(
            self.image,
            verbose=False,
            conf=self.conf,
            iou=0.5,
        )
        r = results[0]

        if r.keypoints is None or r.keypoints.data.shape[0] == 0:
            print("  No court detection above confidence threshold.")
            return None

        # Pick the box with the highest class confidence
        best_idx = int(r.boxes.conf.argmax())
        det_conf = float(r.boxes.conf[best_idx])
        print(f"  Best detection confidence: {det_conf:.3f} "
              f"(of {r.keypoints.data.shape[0]} total)")

        kpts = r.keypoints.data[best_idx].cpu().numpy()  # [13, 3]
        self.keypoints = kpts
        return kpts

    # ── Explore mode ──────────────────────────────────────────────────────────

    def visualize_explore(self) -> np.ndarray:
        """
        Draw all detected keypoints with index labels.

        Each keypoint is drawn only if its visibility score exceeds
        self.conf.  The index number, coordinates, and confidence are
        shown beside the dot.
        """
        kpts = self.keypoints
        img = self.image.copy()

        if kpts is None:
            cv2.putText(img, "No court keypoints detected",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return img

        print(f"\nDetected keypoints (conf > {self.conf}):")
        for i in range(13):
            x, y, vis = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
            if vis < self.conf:
                continue
            xi, yi = int(x), int(y)
            color = _KP_COLORS[i]

            # Draw circle + white border
            cv2.circle(img, (xi, yi), 14, (255, 255, 255), -1)
            cv2.circle(img, (xi, yi), 11, color, -1)

            # Label: index + conf
            label = f"[{i}] {vis:.2f}"
            cv2.putText(img, label, (xi + 16, yi + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(img, label, (xi + 16, yi + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            print(f"  kp[{i:2d}]: ({xi:4d}, {yi:4d})  vis={vis:.3f}")

        # Legend strip at top
        legend_bg = img[0:40, :].copy()
        cv2.rectangle(img, (0, 0), (self.w, 40), (0, 0, 0), -1)
        cv2.addWeighted(legend_bg, 0.4, img[0:40, :], 0.6, 0, img[0:40, :])
        cv2.putText(img,
                    "EXPLORE MODE — keypoint indices shown. "
                    "Adjust DEFAULT_KEYPOINT_TO_FIBA then run --mode localize",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    # ── Localize mode ─────────────────────────────────────────────────────────

    def compute_homography(
        self,
        mapping: dict[int, tuple[float, float]],
    ) -> np.ndarray | None:
        """
        Compute H: pixel → FIBA-court from detected keypoints + mapping.

        Returns the 3×3 homography matrix or None if insufficient points.
        """
        kpts = self.keypoints
        if kpts is None:
            print("  No keypoints available — cannot compute homography.")
            return None

        src_pts, dst_pts = [], []
        print("\nKeypoints used for homography:")
        for idx, (cx, cy) in mapping.items():
            vis = float(kpts[idx, 2])
            if vis < self.conf:
                print(f"  kp[{idx:2d}] → ({cx:.2f}, {cy:.2f}) m  SKIPPED (vis={vis:.2f} < {self.conf})")
                continue
            px, py = float(kpts[idx, 0]), float(kpts[idx, 1])
            src_pts.append([px, py])
            dst_pts.append([cx, cy])
            print(f"  kp[{idx:2d}]: pixel ({px:.0f}, {py:.0f}) → court ({cx:.2f}, {cy:.2f}) m  vis={vis:.2f}")

        if len(src_pts) < 4:
            print(f"  Only {len(src_pts)} usable keypoints — need ≥4 for homography.")
            return None

        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)
        H, status = cv2.findHomography(src, dst, method=0)

        if H is None:
            print("  findHomography failed.")
            return None

        inliers = int(np.sum(status)) if status is not None else len(src_pts)
        print(f"  Homography computed from {inliers}/{len(src_pts)} inliers")

        # Reprojection error
        src_h = np.concatenate([src, np.ones((len(src), 1))], axis=1).T  # 3×N
        proj = H @ src_h
        proj /= proj[2:3, :]
        errs = np.linalg.norm(proj[:2, :].T - dst, axis=1)
        print(f"  Reprojection errors (pixels→court m): "
              f"mean={errs.mean():.4f}  max={errs.max():.4f}")

        self.H = H
        return H

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float] | None:
        """Transform a pixel coordinate to FIBA court coords using self.H."""
        if self.H is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.H)[0, 0]
        cx, cy = float(out[0]), float(out[1])
        if not (0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_HALF_LENGTH):
            return None
        return cx, cy

    def overlay_court_lines(self) -> np.ndarray:
        """
        Project FIBA court lines onto the image using the inverse homography
        (H maps pixel→court, so H⁻¹ maps court→pixel).
        """
        img = self.image.copy()
        if self.H is None:
            cv2.putText(img, "No homography available",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img

        H_inv = np.linalg.inv(self.H)

        def court_to_pixel(cx: float, cy: float) -> tuple[int, int] | None:
            pt = np.array([[[cx, cy]]], dtype=np.float64)
            out = cv2.perspectiveTransform(pt, H_inv)[0, 0]
            px, py = int(round(out[0])), int(round(out[1]))
            if -500 < px < self.w + 500 and -500 < py < self.h + 500:
                return px, py
            return None

        def draw_line(p1_court, p2_court, color, thickness=2):
            a = court_to_pixel(*p1_court)
            b = court_to_pixel(*p2_court)
            if a and b:
                cv2.line(img, a, b, color, thickness)

        def draw_arc(cx_m, cy_m, r_m, a_start, a_end, color, thickness=2, steps=60):
            """Draw a circular arc sampled at `steps` segments."""
            angles = np.linspace(np.radians(a_start), np.radians(a_end), steps)
            pts = []
            for ang in angles:
                px_ = cx_m + r_m * np.cos(ang)
                py_ = cy_m + r_m * np.sin(ang)
                p = court_to_pixel(px_, py_)
                if p:
                    pts.append(p)
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, thickness)

        W, L = COURT_WIDTH, COURT_HALF_LENGTH
        PAINT_L = 5.05
        PAINT_R = 9.95
        FT_Y = FREE_THROW_LINE_Y

        YELLOW = (0, 220, 220)
        GREEN  = (0, 200,   0)
        CYAN   = (220, 220,   0)
        ORANGE = (0, 140, 255)
        WHITE  = (255, 255, 255)
        THICK  = 3

        # Half-court outline
        draw_line((0, 0),   (W, 0),  YELLOW, THICK)      # baseline
        draw_line((0, 0),   (0, L),  YELLOW, THICK)      # left sideline
        draw_line((W, 0),   (W, L),  YELLOW, THICK)      # right sideline
        draw_line((0, L),   (W, L),  YELLOW, THICK)      # half-court

        # Paint box
        draw_line((PAINT_L, 0),   (PAINT_L, FT_Y), GREEN, THICK)
        draw_line((PAINT_R, 0),   (PAINT_R, FT_Y), GREEN, THICK)
        draw_line((PAINT_L, FT_Y),(PAINT_R, FT_Y), GREEN, THICK)

        # Free-throw circle (radius 1.8 m from basket, centred at (7.5, 5.8))
        draw_arc(BASKET_X, FT_Y, 1.8, 0, 180, CYAN, THICK)   # upper half
        # Lower half dashed — skip for simplicity

        # 3-point arc (radius 6.75 m, basket at (7.5, 1.575))
        # Straight corner sections
        THREE_R = 6.75
        CORNER_X_L = 0.9   # metres from left sideline (FIBA)
        CORNER_X_R = W - 0.9
        CORNER_Y   = 2.99  # transition from straight to arc
        draw_line((CORNER_X_L, 0), (CORNER_X_L, CORNER_Y), ORANGE, THICK)
        draw_line((CORNER_X_R, 0), (CORNER_X_R, CORNER_Y), ORANGE, THICK)
        # Arc from left elbow to right elbow
        ang_left  = np.degrees(np.arctan2(CORNER_Y - BASKET_Y,
                                          CORNER_X_L - BASKET_X))
        ang_right = np.degrees(np.arctan2(CORNER_Y - BASKET_Y,
                                          CORNER_X_R - BASKET_X))
        draw_arc(BASKET_X, BASKET_Y, THREE_R, ang_left, ang_right, ORANGE, THICK, steps=80)

        # Basket
        basket_px = court_to_pixel(BASKET_X, BASKET_Y)
        if basket_px:
            cv2.circle(img, basket_px, 6, WHITE, -1)
            cv2.circle(img, basket_px, 8, (0, 0, 200), 2)

        # Calibration keypoints overlay (those used in homography)
        if self.keypoints is not None:
            for i in range(13):
                vis = float(self.keypoints[i, 2])
                if vis < self.conf:
                    continue
                xi = int(self.keypoints[i, 0])
                yi = int(self.keypoints[i, 1])
                color = _KP_COLORS[i]
                cv2.circle(img, (xi, yi), 9, (255, 255, 255), -1)
                cv2.circle(img, (xi, yi), 7, color, -1)
                cv2.putText(img, str(i), (xi + 10, yi + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return img

    # ── Shooter detection (optional) ──────────────────────────────────────────

    def detect_and_localize_shooter(
        self,
        shot_model_path: str,
        vis_img: np.ndarray,
    ) -> np.ndarray:
        """
        Run a ball/person YOLO model, find the shooter, and mark their
        court coordinates on the image.
        """
        print(f"\nLoading shot detection model: {shot_model_path}")
        shot_model = YOLO(shot_model_path, verbose=False)
        results = shot_model(self.image, verbose=False)[0]

        ball_boxes, person_boxes = [], []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            name = results.names[cls]
            entry = {'bbox': xyxy.tolist(), 'conf': conf}
            if name == 'ball' and conf > 0.3:
                ball_boxes.append(entry)
            elif name == 'person' and conf > 0.5:
                person_boxes.append(entry)

        print(f"  Balls: {len(ball_boxes)}  Persons: {len(person_boxes)}")

        if not ball_boxes or not person_boxes:
            print("  Cannot identify shooter (missing ball or person).")
            return vis_img

        # Best ball
        ball = max(ball_boxes, key=lambda b: b['conf'])
        bx1, by1, bx2, by2 = ball['bbox']

        # Person with greatest overlap fraction (intersection / person_area)
        best_person, best_ratio = None, 0.0
        for person in person_boxes:
            px1, py1, px2, py2 = person['bbox']
            ix1, iy1 = max(bx1, px1), max(by1, py1)
            ix2, iy2 = min(bx2, px2), min(by2, py2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            p_area = (px2 - px1) * (py2 - py1)
            ratio = inter / p_area if p_area > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_person = person

        if best_person is None or best_ratio < 0.05:
            print("  No person overlaps the ball sufficiently.")
            return vis_img

        px1, py1, px2, py2 = best_person['bbox']
        foot_x = (px1 + px2) / 2
        foot_y = float(py2)

        court_pos = self.pixel_to_court(foot_x, foot_y)
        print(f"  Shooter foot pixel: ({foot_x:.0f}, {foot_y:.0f})")
        if court_pos:
            cx, cy = court_pos
            print(f"  Court position: ({cx:.2f} m, {cy:.2f} m)")
            in_bounds = 0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_HALF_LENGTH
            print(f"  In-bounds: {in_bounds}")
        else:
            print("  Court position: out-of-bounds / transform failed")

        # Draw on image
        cv2.rectangle(vis_img, (px1, py1), (px2, py2), (0, 255, 0), 3)
        cv2.circle(vis_img, (int(foot_x), int(foot_y)), 12, (0, 0, 255), -1)
        cv2.circle(vis_img, (int(foot_x), int(foot_y)), 15, (255, 255, 255), 2)
        if court_pos:
            label = f"({cx:.1f}, {cy:.1f}) m"
            cv2.putText(vis_img, label, (px1, py1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Draw ball
        cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (0, 220, 220), 2)
        cv2.putText(vis_img, f"ball {ball['conf']:.2f}", (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)

        return vis_img

    # ── Run pipelines ─────────────────────────────────────────────────────────

    def run_explore(self, save: bool = False) -> None:
        print("\n" + "=" * 60)
        print("EXPLORE MODE")
        print("=" * 60)

        self.detect_keypoints()
        vis = self.visualize_explore()

        out_path = f"court_kp_explore_{Path(self.image_path).stem}.png"
        cv2.imwrite(out_path, vis)
        print(f"\nSaved: {out_path}")

        if save:
            return

        win = "Court Keypoints — Explore  (Q to quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.imshow(win, vis)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
        cv2.destroyAllWindows()

    def run_localize(
        self,
        mapping: dict[int, tuple[float, float]],
        shot_model_path: str | None = None,
        save: bool = False,
    ) -> None:
        print("\n" + "=" * 60)
        print("LOCALIZE MODE")
        print("=" * 60)

        self.detect_keypoints()
        if self.keypoints is None:
            print("Aborted: no court detection.")
            return

        H = self.compute_homography(mapping)
        if H is None:
            print("Aborted: homography could not be computed.")
            print("  → Run --mode explore and check keypoint visibility.")
            print("  → Lower --conf or add more points to your mapping.")
            return

        vis = self.overlay_court_lines()

        if shot_model_path:
            vis = self.detect_and_localize_shooter(shot_model_path, vis)

        # Validate mapping points (reprojection from FIBA → pixel)
        H_inv = np.linalg.inv(H)
        print("\nMapping validation (FIBA → pixel → compare to detected):")
        for idx, (cx, cy) in mapping.items():
            vis_score = float(self.keypoints[idx, 2])
            if vis_score < self.conf:
                continue
            det_px = (float(self.keypoints[idx, 0]), float(self.keypoints[idx, 1]))
            pt = np.array([[[cx, cy]]], dtype=np.float64)
            proj = cv2.perspectiveTransform(pt, H_inv)[0, 0]
            err = np.linalg.norm(np.array(det_px) - proj)
            print(f"  kp[{idx:2d}] ({cx:.2f},{cy:.2f})m → H⁻¹ → ({proj[0]:.0f},{proj[1]:.0f})  "
                  f"detected ({det_px[0]:.0f},{det_px[1]:.0f})  err={err:.1f}px")

        out_path = f"court_kp_localize_{Path(self.image_path).stem}.png"
        cv2.imwrite(out_path, vis)
        print(f"\nSaved: {out_path}")

        if save:
            return

        win = "Court Keypoints — Localize  (Q to quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.imshow(win, vis)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

def load_mapping(path: str) -> dict[int, tuple[float, float]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): tuple(v) for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Static court keypoint detection using court_point.pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1 — identify keypoint indices:
  python static_court_keypoint_detection.py --image shot.jpg --mode explore

  # Step 2 — compute homography and overlay court lines:
  python static_court_keypoint_detection.py --image shot.jpg --mode localize

  # Step 2 with custom mapping JSON and shooter localisation:
  python static_court_keypoint_detection.py --image shot.jpg --mode localize \\
      --mapping my_mapping.json \\
      --shot-model backend/score_detection/weights/new_weight.pt

Mapping JSON format (keypoint_index → [court_x_m, court_y_m]):
  {
    "7":  [5.05, 5.8],
    "8":  [9.95, 5.8],
    "10": [5.05, 0.0],
    "11": [9.95, 0.0]
  }
        """,
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--model",
        default="backend/score_detection/weights/court_point.pt",
        help="Path to court keypoint model (default: court_point.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=["explore", "localize"],
        default="explore",
        help="explore: visualise keypoints | localize: compute H + overlay (default: explore)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="JSON file with keypoint→FIBA mapping (localize mode; uses built-in default if omitted)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Keypoint visibility threshold (default: 0.3)",
    )
    parser.add_argument(
        "--shot-model",
        default=None,
        help="YOLO model for ball/person detection (optional, localize mode only)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output and skip interactive display",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    detector = CourtKeypointDetector(args.image, args.model, conf=args.conf)

    if args.mode == "explore":
        detector.run_explore(save=args.save)

    else:  # localize
        if args.mapping:
            if not Path(args.mapping).exists():
                print(f"ERROR: mapping file not found: {args.mapping}")
                sys.exit(1)
            mapping = load_mapping(args.mapping)
            print(f"Loaded mapping from {args.mapping}: {mapping}")
        else:
            mapping = DEFAULT_KEYPOINT_TO_FIBA
            print(f"Using built-in DEFAULT_KEYPOINT_TO_FIBA mapping "
                  f"({len(mapping)} keypoints)")

        if args.shot_model and not Path(args.shot_model).exists():
            print(f"WARNING: shot model not found: {args.shot_model} — skipping shooter detection")
            args.shot_model = None

        detector.run_localize(
            mapping=mapping,
            shot_model_path=args.shot_model,
            save=args.save,
        )


if __name__ == "__main__":
    main()
