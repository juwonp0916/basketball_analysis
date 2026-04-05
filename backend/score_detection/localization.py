# shot_localizer.py

import numpy as np
import cv2
import os
from pathlib import Path
from datetime import datetime

from constants import (
    MAP_WIDTH,
    MAP_HEIGHT,
)

from logger import (
    INFO,
    SOCKET,
    Logger
)

logger = Logger([
    INFO
])


class ShotLocalizer:
    def __init__(self, calibration_points, image_dimensions, court_img_path='court_img.png', enable_visualization=False, calibration_mode='6-point'):
        """
        Initialize localizer with 4 or 6 calibration points

        Args:
            calibration_points: List of 4 or 6 points [(x,y), ...] in video coordinates
            image_dimensions: Image dimensions as (width, height)
            court_img_path: Path to court diagram image for visualization
            enable_visualization: Whether to enable shot visualization
            calibration_mode: "4-point" (paint box) or "6-point" (full baseline)
        """
        self.calibration_mode = calibration_mode
        self.calibration_points = self._parse_points(calibration_points, calibration_mode)
        self.image_width, self.image_height = self._parse_dimensions(image_dimensions)

        # FIBA court reference points (in meters)
        from constants import (CALIBRATION_POINTS_FIBA, CALIBRATION_POINTS_FIBA_PAINT,
                               CALIBRATION_POINTS_FIBA_PAINT_3PT, CALIBRATION_POINTS_FIBA_COURT)
        if calibration_mode == '4-point':
            self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA_PAINT, dtype=float)
        elif calibration_mode == '4-point-court':
            self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA_COURT, dtype=float)
        elif calibration_mode == '6-point-3pt':
            self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA_PAINT_3PT, dtype=float)
        else:  # '6-point': full baseline (sideline corners + paint corners + FT corners)
            self.court_calibration_points = np.array(CALIBRATION_POINTS_FIBA, dtype=float)

        # Calculate the homography matrix when initialized
        self.shot_data = []
        self.homography_matrix = self._calculate_homography()

        # For 6-point mode: also compute a paint-only homography (exact fit, no tilt)
        # used for shot localization. The full H is kept for the court overlay.
        # Paint corners are indices 1,2,4,5 (0-based) in CALIBRATION_POINTS_FIBA:
        #   (5.05,0), (9.95,0), (5.05,5.8), (9.95,5.8)
        # For 4-point-court mode: homography_matrix is already exact — no subset needed.
        self._paint_homography = None
        if calibration_mode == '6-point':
            paint_idx = np.array([1, 2, 4, 5])
            H_paint, _ = cv2.findHomography(
                self.calibration_points[paint_idx],
                self.court_calibration_points[paint_idx],
                method=0
            )
            if H_paint is not None:
                self._paint_homography = H_paint
                logger.log(INFO, "6-point mode: paint-only homography computed for shot localization")

        # Visualization setup (now uses SVG-based renderer instead of PNG)
        self.enable_visualization = enable_visualization
        # Legacy attributes kept for backward compatibility but not used
        self.court_img_path = court_img_path
        self.court_img = None
        self.court_img_width = None
        self.court_img_height = None
        self.shot_markers = []  # Deprecated - using shot_data instead

        # No need to load court image - using SVG renderer now
        if self.enable_visualization:
            logger.log(INFO, "Shot visualization enabled (SVG-based renderer)")
    
    def _parse_points(self, points_input, calibration_mode='6-point'):
        """Parse calibration points into numpy array"""
        if calibration_mode in ('4-point', '4-point-court'):
            expected_points = 4
        else:
            expected_points = 6

        if isinstance(points_input, str):
            # Format: "x1,y1,x2,y2,..."
            coords = list(map(float, points_input.split(',')))
            expected_coords = expected_points * 2
            if len(coords) != expected_coords:
                raise ValueError(f"Expected {expected_coords} coordinates ({expected_points} points), got {len(coords)}")
            return np.array([
                [coords[i], coords[i+1]] for i in range(0, len(coords), 2)
            ], dtype=float)
        elif isinstance(points_input, list):
            points_array = np.array(points_input, dtype=np.float32)
            if points_array.shape[0] != expected_points or points_array.shape[1] != 2:
                raise ValueError(f"Expected {expected_points} points with (x,y), got shape {points_array.shape}")
            return points_array
        else:
            raise ValueError("Invalid format for calibration points")
    
    def _parse_dimensions(self, dimensions_str):
        """Parse dimensions string into width and height"""
        if isinstance(dimensions_str, str):
            # Format: "width,height"
            return tuple(map(float, dimensions_str.split(',')))
        elif isinstance(dimensions_str, tuple) or isinstance(dimensions_str, list):
            # Already in tuple/list format
            return dimensions_str
        else:
            raise ValueError("Invalid format for image dimensions")
    
    def _calculate_homography(self):
        """
        Calculate homography matrix from video points to court points.

        For 6-point mode the naive approach is to feed all 6 points to DLT, but
        this is a degenerate configuration: 4 of the 6 points lie on the baseline
        (Y=0) and therefore form a collinear set in BOTH the pixel image (the
        baseline appears as a straight line in perspective) AND the FIBA destination
        space.  Four collinear correspondences contribute only ~2 independent
        constraints to the 8-DOF linear system instead of the expected 8, leaving
        the system underdetermined by ~2 DOF.  The DLT SVD resolves the ambiguity
        by picking a minimum-norm solution in the 2-D null space, which introduces
        large, unpredictable error (~50-60 cm) specifically at the interior baseline
        points (the paint corners p[1] and p[2]).

        Fix: for 6-point mode compute H from the 4 outer non-collinear points
          p[0]=(0,0)    p[3]=(15,0)    p[4]=(5.05,5.8)    p[5]=(9.95,5.8)
        No subset of 3 of these is collinear, so the system is exactly determined
        (4 pts = 8 constraints = 8 DOF → unique solution, 0 residual).  The two
        inner baseline points p[1] and p[2] are deliberately excluded from the H
        computation because they provide zero additional independent constraint when
        the 4 outer points already fix the homography.  They are still recorded and
        used for the court overlay.
        """
        expected_points = 4 if self.calibration_mode in ('4-point', '4-point-court') else 6
        if self.calibration_points.shape[0] != expected_points:
            raise ValueError(
                f"Need exactly {expected_points} calibration points for "
                f"{self.calibration_mode} mode, got {self.calibration_points.shape[0]}"
            )

        if self.calibration_mode == '6-point':
            # Non-collinear 4-point subset: sideline corners + FT paint corners.
            # Indices 0,3,4,5 → (0,0), (15,0), (5.05,5.8), (9.95,5.8).
            idx = np.array([0, 3, 4, 5])
            src_pts = self.calibration_points[idx]
            dst_pts = self.court_calibration_points[idx]
        else:
            src_pts = self.calibration_points
            dst_pts = self.court_calibration_points

        H, status = cv2.findHomography(src_pts, dst_pts, method=0)

        if H is None:
            raise ValueError("Failed to compute homography matrix. Check calibration points.")

        self._validate_homography(H)
        return H

    def _validate_homography(self, H):
        """
        Validate homography quality by back-projecting all calibration points.

        For 6-point mode, only indices [0,3,4,5] were used to compute H.  Indices
        [1,2] (paint baseline corners) were deliberately excluded because they are
        collinear with the other baseline points and add no independent constraint.
        Their reprojection error reflects how far the user's click for those points
        differs from H's prediction — it is NOT a homography error and should not
        trigger a "re-click" warning.
        """
        from constants import (CALIBRATION_LABELS, CALIBRATION_LABELS_PAINT,
                               CALIBRATION_LABELS_PAINT_3PT, CALIBRATION_LABELS_COURT)
        if self.calibration_mode == '4-point':
            labels = CALIBRATION_LABELS_PAINT
        elif self.calibration_mode == '4-point-court':
            labels = CALIBRATION_LABELS_COURT
        elif self.calibration_mode == '6-point-3pt':
            labels = CALIBRATION_LABELS_PAINT_3PT
        else:
            labels = CALIBRATION_LABELS

        # Indices of points actually used to compute H (others are informational only)
        if self.calibration_mode == '6-point':
            h_point_indices = {0, 3, 4, 5}
        else:
            h_point_indices = set(range(len(self.calibration_points)))

        max_error_h = 0.0
        errors = []

        for i, (orig_pt, court_pt) in enumerate(zip(self.calibration_points, self.court_calibration_points)):
            point_array = np.array([[orig_pt[0], orig_pt[1]]], dtype=float)
            transformed = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), H)

            error = np.sqrt((transformed[0][0][0] - court_pt[0])**2 +
                            (transformed[0][0][1] - court_pt[1])**2)
            errors.append(error)
            label = labels[i] if i < len(labels) else f"Point {i+1}"

            if i in h_point_indices:
                # Point was used to fit H — any error here is a genuine homography issue.
                max_error_h = max(max_error_h, error)
                if error > 0.3:
                    logger.log(INFO, f"CALIBRATION ERROR point {i+1} ({label}): {error:.3f}m — re-click this point")
                elif error > 0.1:
                    logger.log(INFO, f"Calibration warning point {i+1} ({label}): {error:.3f}m")
            else:
                # Point was NOT used to fit H — error reflects click imprecision,
                # not a homography problem.  Log as informational only.
                logger.log(INFO,
                    f"Point {i+1} ({label}): click offset from H prediction = {error:.3f}m "
                    f"[not used for H — overlay uses H-projected position]")

        avg_error = np.mean(errors)
        logger.log(INFO, f"Homography validation — avg={avg_error:.3f}m  max(H-points)={max_error_h:.3f}m")

        if max_error_h > 0.2:
            logger.log(INFO, "Calibration accuracy is marginal — recalibrating may improve zone detection")

        return max_error_h

    def get_calibration_diagnostics(self):
        """
        Return per-point reprojection errors for UI feedback.

        Returns:
            dict with 'errors' (list of floats in meters) and 'avg_error'
        """
        H = self.homography_matrix
        errors = []
        for orig_pt, court_pt in zip(self.calibration_points, self.court_calibration_points):
            pt = np.array([[orig_pt[0], orig_pt[1]]], dtype=float)
            mapped = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), H)[0][0]
            err = float(np.sqrt((mapped[0] - court_pt[0])**2 + (mapped[1] - court_pt[1])**2))
            errors.append(err)
        return {'errors': errors, 'avg_error': float(np.mean(errors))}

    def get_court_outline_pixels(self):
        """
        Project FIBA court lines back to pixel coordinates.

        For 6-point mode:
          - Lines through calibrated points use exact pixel positions.
          - Sidelines use the vanishing point of the depth direction (from paint
            edges) to compute the correct perspective direction, then clip to the
            image boundary.  No cross-ratio is used for the far end because the
            camera is typically positioned near the half-court line, making y=14
            project far off-screen.
          - 3PT arc / straight sections use H_full_inv (DLT fit over all 6 points,
            which covers x=0..15 and is better than H_paint_inv for x outside the
            paint box).
        For other modes a single homography projects everything.
        """
        import math

        H_full_inv = np.linalg.inv(self.homography_matrix)

        def proj(court_pts, H):
            pts = np.array(court_pts, dtype=np.float64).reshape(-1, 1, 2)
            px = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            return [[float(x), float(y)] for x, y in px]

        def pt(arr):
            return [float(arr[0]), float(arr[1])]

        def extend_to_image_edge(start, direction, W, H_im):
            """
            Starting at `start`, extend the ray in `direction` until it hits an
            image boundary (x=0, x=W, y=0, or y=H_im).  Returns the clipped
            endpoint.  If no positive-t boundary is found, returns start.
            """
            sx, sy = float(start[0]), float(start[1])
            dx, dy = float(direction[0]), float(direction[1])
            candidates = []
            if dx < -1e-6:
                candidates.append(-sx / dx)           # hits x=0
            elif dx > 1e-6:
                candidates.append((W - sx) / dx)      # hits x=W
            if dy < -1e-6:
                candidates.append(-sy / dy)            # hits y=0
            elif dy > 1e-6:
                candidates.append((H_im - sy) / dy)   # hits y=H
            valid = [t for t in candidates if t > 1e-6]
            if not valid:
                return [sx, sy]
            t = min(valid)
            return [sx + t * dx, sy + t * dy]

        # --- 3PT arc points (overlay geometry only) ---
        BASKET_X, BASKET_Y = 7.5, 1.575
        R_3PT = 6.75   # arc radius in meters (overlay only)
        # ARC_Y: solve (0.9 - 7.5)^2 + (ARC_Y - 1.575)^2 = 6.75^2  → ARC_Y = 2.99m
        ARC_Y = 2.99   # y where the straight 3PT section meets the arc
        theta_left  = math.atan2(ARC_Y - BASKET_Y, 0.9  - BASKET_X)
        theta_right = math.atan2(ARC_Y - BASKET_Y, 14.1 - BASKET_X)
        arc_pts = []
        for i in range(41):
            t = theta_left - (theta_left - theta_right) * i / 40
            arc_pts.append([BASKET_X + R_3PT * math.cos(t),
                            BASKET_Y + R_3PT * math.sin(t)])

        if self.calibration_mode == '6-point':
            # Calibration points in pixel space:
            #   p[0]=(0,0)   p[1]=(5.05,0)  p[2]=(9.95,0)  p[3]=(15,0)
            #   p[4]=(5.05,5.8)              p[5]=(9.95,5.8)
            #
            # H was computed from {p[0], p[3], p[4], p[5]} only (non-collinear
            # subset).  p[1] and p[2] were excluded to avoid the degenerate DLT
            # caused by 4 collinear baseline points.  Consequently the user's
            # clicked pixels for p[1]/p[2] may differ from where H correctly
            # projects (5.05,0) and (9.95,0).  We must use the H-projected
            # positions for all overlay geometry that depends on these points
            # (baseline, paint box, VP, 3PT baseline interpolation) so that the
            # overlay is consistent with the localization coordinate system.
            p = self.calibration_points
            W   = float(self.image_width)
            H_im = float(self.image_height)

            # H-projected pixel positions for the two paint baseline corners.
            p1_h = np.array(proj([[5.05, 0.0]], H_full_inv)[0], dtype=float)
            p2_h = np.array(proj([[9.95, 0.0]], H_full_inv)[0], dtype=float)

            # --- Vanishing point of all depth-direction lines ---
            # Use H-projected p1_h/p2_h (not user clicks) so the VP is derived
            # from internally consistent positions.
            d1 = p[4].astype(float) - p1_h
            d2 = p[5].astype(float) - p2_h
            cross = float(d1[0] * d2[1] - d1[1] * d2[0])

            if abs(cross) > 1.0:
                t_vp = float(((p2_h[0] - p1_h[0]) * d2[1] -
                              (p2_h[1] - p1_h[1]) * d2[0]) / cross)
                V = p1_h + t_vp * d1  # vanishing point (pixels)

                left_dir  = p[0].astype(float) - V
                right_dir = p[3].astype(float) - V

                left_end  = extend_to_image_edge(p[0], left_dir,  W, H_im)
                right_end = extend_to_image_edge(p[3], right_dir, W, H_im)

                left_line  = [pt(p[0]), left_end]
                right_line = [pt(p[3]), right_end]

                # --- 3PT straight sections: VP-based rendering ---
                # Interpolate baseline pixels using H-projected p1_h/p2_h so the
                # 3PT line starts on the correct (H-consistent) baseline.
                frac_left  = 0.9 / 5.05                      # X=0.9 between p[0](X=0) and p1_h(X=5.05)
                frac_right = (14.1 - 9.95) / (15.0 - 9.95)  # X=14.1 between p2_h(X=9.95) and p[3](X=15)
                b_3pt_left  = p[0].astype(float) + frac_left  * (p1_h - p[0].astype(float))
                b_3pt_right = p2_h               + frac_right * (p[3].astype(float) - p2_h)

                # β = perspective depth ratio for Y=2.99m relative to baseline.
                r0_l = np.linalg.norm(p1_h - V)
                r4_l = np.linalg.norm(p[4].astype(float) - V)
                r0_r = np.linalg.norm(p2_h - V)
                r4_r = np.linalg.norm(p[5].astype(float) - V)
                if r0_l > 1e-6 and r0_r > 1e-6:
                    alpha = ((r4_l / r0_l) + (r4_r / r0_r)) / 2.0
                    Y_elbow, Y_ft = ARC_Y, 5.8
                    beta = (Y_ft * alpha) / (Y_ft * alpha + Y_elbow * (1.0 - alpha))
                    top_3pt_left  = V + beta * (b_3pt_left  - V)
                    top_3pt_right = V + beta * (b_3pt_right - V)
                else:
                    top_3pt_left  = np.array(proj([[0.9,  ARC_Y]], H_full_inv)[0])
                    top_3pt_right = np.array(proj([[14.1, ARC_Y]], H_full_inv)[0])

            else:
                # Near top-down camera — paint edges nearly parallel, VP at ∞.
                left_line  = proj([[0,  0], [0,  7]], H_full_inv)
                right_line = proj([[15, 0], [15, 7]], H_full_inv)
                b_3pt_left  = np.array(proj([[0.9,  0]], H_full_inv)[0])
                b_3pt_right = np.array(proj([[14.1, 0]], H_full_inv)[0])
                top_3pt_left  = np.array(proj([[0.9,  ARC_Y]], H_full_inv)[0])
                top_3pt_right = np.array(proj([[14.1, ARC_Y]], H_full_inv)[0])

            arc_projected = proj(arc_pts, H_full_inv)
            arc_projected[0]  = pt(top_3pt_left)
            arc_projected[-1] = pt(top_3pt_right)

            polylines = [
                # Baseline: outer corners exact (calibration); inner paint corners
                # from H projection (more accurate than user clicks at the far baseline)
                [pt(p[0]), pt(p1_h), pt(p2_h), pt(p[3])],
                left_line,
                right_line,
                # Paint box: baseline corners from H projection, FT corners exact
                [pt(p1_h), pt(p[4]), pt(p[5]), pt(p2_h)],
                [pt(p[4]), pt(p[5])],
                [pt(b_3pt_left),  pt(top_3pt_left)],
                [pt(b_3pt_right), pt(top_3pt_right)],
                arc_projected,
            ]
        elif self.calibration_mode == '4-point':
            # Paint-box 4-point mode (NBA dimensions).
            #
            # Calibration points (pixel coords):
            #   p[0] → (5.05, 0)   Baseline Left Paint Corner
            #   p[1] → (9.95, 0)   Baseline Right Paint Corner
            #   p[2] → (5.05, 5.8) FT Line Left
            #   p[3] → (9.95, 5.8) FT Line Right
            #
            # Overlay strategy:
            #   Paint box / FT line  — exact calibration pixel positions (zero error).
            #   All y=0 baseline points — H_inv projection.  H encodes the full
            #       projective geometry so the closer side correctly maps to more
            #       pixels per meter and the far side to fewer.  Linear extrapolation
            #       would apply uniform scale and collapse the asymmetry.
            #   Sidelines — extended from H_inv baseline corners in the paint edge
            #       direction (d_left = p[2]-p[0]).  All Y-parallel lines share the
            #       same VP, so this direction is correct for the sidelines too.
            #   3PT arc / straight sections — H_inv for all points, no endpoint
            #       overrides (avoids the "mushroom" kink from method mixing).

            p   = self.calibration_points.astype(float)
            W   = float(self.image_width)
            H_im = float(self.image_height)

            # ── X-direction vanishing point (VP_X) ───────────────────────────────
            # The baseline (p[0]→p[1]) and the FT line (p[2]→p[3]) are both
            # x-direction court lines — they converge at VP_X in the image.
            # VP_X lets us use the cross-ratio to place any x-coordinate along the
            # baseline correctly, without relying on H_inv extrapolation which
            # amplifies click errors ~3× at 103% beyond the calibration edge.
            d01 = p[1] - p[0]   # baseline direction (pixels)
            d23 = p[3] - p[2]   # FT line direction (pixels)
            dp  = p[2] - p[0]
            det_x = d01[1] * d23[0] - d01[0] * d23[1]

            if abs(det_x) > 1.0:
                # Lines converge at a finite VP_X — use cross-ratio
                t_x  = (dp[1] * d23[0] - dp[0] * d23[1]) / det_x
                VP_X = p[0] + t_x * d01

                d_len = float(np.linalg.norm(d01))           # pixels over 4.9 m
                baseline_unit = d01 / d_len
                v_vp = float(np.dot(VP_X - p[0], baseline_unit))  # signed px to VP_X

                def _cr_baseline(X_fiba):
                    """Cross-ratio: FIBA x → 2D pixel on the baseline line."""
                    Q = (9.95 - X_fiba) / 4.9
                    denom = v_vp - Q * d_len
                    if abs(denom) < 1e-4:
                        return None
                    a = d_len * v_vp * (1.0 - Q) / denom
                    return p[0] + a * baseline_unit

                p_left_base_cr  = _cr_baseline(0.0)
                p_right_base_cr = _cr_baseline(15.0)
                b_3pt_left_cr   = _cr_baseline(0.9)
                b_3pt_right_cr  = _cr_baseline(14.1)

                if all(x is not None for x in
                       [p_left_base_cr, p_right_base_cr, b_3pt_left_cr, b_3pt_right_cr]):
                    p_left_base  = p_left_base_cr
                    p_right_base = p_right_base_cr
                    b_3pt_left   = pt(b_3pt_left_cr)
                    b_3pt_right  = pt(b_3pt_right_cr)
                    logger.debug("[OVERLAY] baseline via cross-ratio (VP_X=%.1f,%.1f)", *VP_X)
                else:
                    p_left_base  = np.array(proj([[0.0,  0.0]], H_full_inv)[0])
                    p_right_base = np.array(proj([[15.0, 0.0]], H_full_inv)[0])
                    b_3pt_left   = proj([[0.9,  0.0]], H_full_inv)[0]
                    b_3pt_right  = proj([[14.1, 0.0]], H_full_inv)[0]
            else:
                # VP_X at infinity (baseline and FT line parallel in image —
                # top-down or perfectly level camera).  Fall back to H_inv.
                p_left_base  = np.array(proj([[0.0,  0.0]], H_full_inv)[0])
                p_right_base = np.array(proj([[15.0, 0.0]], H_full_inv)[0])
                b_3pt_left   = proj([[0.9,  0.0]], H_full_inv)[0]
                b_3pt_right  = proj([[14.1, 0.0]], H_full_inv)[0]

            # ── Sideline directions: project a second point on each sideline ───────
            # Using the paint edge direction (p[2]-p[0]) makes the sideline
            # pixel-parallel to the paint edge, which is wrong — they must converge
            # at the same VP.  Projecting (0, 5.8) and (15, 5.8) via H_inv gives
            # the correct per-sideline direction that accounts for convergence.
            p_left_far  = np.array(proj([[0.0,  5.8]], H_full_inv)[0])
            p_right_far = np.array(proj([[15.0, 5.8]], H_full_inv)[0])
            left_dir  = p_left_far  - p_left_base
            right_dir = p_right_far - p_right_base

            # ── Sidelines: extend from baseline corners in the correct direction ───
            left_end  = extend_to_image_edge(p_left_base,  left_dir,  W, H_im)
            right_end = extend_to_image_edge(p_right_base, right_dir, W, H_im)
            left_line  = [pt(p_left_base),  left_end]
            right_line = [pt(p_right_base), right_end]

            # ── 3PT sections and arc: H_inv for all points ───────────────────────
            # arc_pts[0] = (0.9, ARC_Y) and arc_pts[-1] = (14.1, ARC_Y) by
            # construction, so t_3pt_left/right are identical to arc endpoints — no
            # gap or kink between the straight sections and the arc.
            t_3pt_left  = proj([[0.9,  ARC_Y]], H_full_inv)[0]
            t_3pt_right = proj([[14.1, ARC_Y]], H_full_inv)[0]
            arc_projected = proj(arc_pts, H_full_inv)

            polylines = [
                [pt(p_left_base), pt(p[0]), pt(p[1]), pt(p_right_base)],  # baseline
                left_line,                                                  # left sideline
                right_line,                                                 # right sideline
                [pt(p[0]), pt(p[2]), pt(p[3]), pt(p[1])],   # paint box (exact)
                [pt(p[2]), pt(p[3])],                         # FT line (exact)
                [b_3pt_left,  t_3pt_left],                    # 3PT left straight
                [b_3pt_right, t_3pt_right],                   # 3PT right straight
                arc_projected,                                 # 3PT arc (H_inv, consistent)
            ]
        else:
            # 4-point-court and other modes: single homography for everything
            polylines = [
                proj([[0,  0], [15,  0]], H_full_inv),
                proj([[0,  0], [0,  14]], H_full_inv),
                proj([[15, 0], [15, 14]], H_full_inv),
                proj([[0, 14], [15, 14]], H_full_inv),
                proj([[5.05, 0], [5.05, 5.8], [9.95, 5.8], [9.95, 0]], H_full_inv),
                proj([[5.05, 5.8], [9.95, 5.8]], H_full_inv),
                proj([[0.9,  0], [0.9,  2.99]], H_full_inv),
                proj([[14.1, 0], [14.1, 2.99]], H_full_inv),
                proj(arc_pts, H_full_inv),
            ]

        return polylines

    def map_to_court(self, point):
        """
        Map a point from video coordinates to court coordinates (in meters)

        Args:
            point: (x, y) coordinates in the video (in pixels)

        Returns:
            (x, y) coordinates in meters on the FIBA half-court
        """
        if point[0] is None or point[1] is None:
            return (None, None)

        logger.log(INFO, f"Mapping point: {point}")

        # Convert point to proper format (pixel coordinates)
        point_array = np.array([[point[0], point[1]]], dtype=float)

        # Always use the full homography for shot localization.
        # The paint-only subset homography extrapolates badly for shots outside the paint
        # (wings, corners, 3PT positions), producing wildly out-of-bounds coordinates.
        # The full DLT homography has a small distributed residual error (~30-60 cm at
        # individual calibration points) but is far more accurate for the entire court.
        H = self.homography_matrix

        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), H)

        # Extract result
        court_x = transformed_point[0][0][0]
        court_y = transformed_point[0][0][1]

        # Validate bounds (court half-court)
        from constants import COURT_WIDTH, COURT_HALF_LENGTH

        if court_x < 0 or court_x > COURT_WIDTH or court_y < 0 or court_y > COURT_HALF_LENGTH:
            logger.log(INFO, f"WARNING: Transformed point ({court_x:.2f}m, {court_y:.2f}m) is out of court bounds " +
                            f"(0 to {COURT_WIDTH}m, 0 to {COURT_HALF_LENGTH}m)")

        return (court_x, court_y)

    def _load_court_image(self):
        """Load the court diagram image for visualization"""
        if not os.path.exists(self.court_img_path):
            logger.log(INFO, f"Warning: Court image not found at {self.court_img_path}")
            self.enable_visualization = False
            return

        self.court_img = cv2.imread(self.court_img_path)
        if self.court_img is None:
            logger.log(INFO, f"Warning: Failed to load court image from {self.court_img_path}")
            self.enable_visualization = False
            return

        self.court_img_height, self.court_img_width = self.court_img.shape[:2]
        logger.log(INFO, f"Court image loaded: {self.court_img_width}x{self.court_img_height}")

    def _court_to_image_coords(self, court_x, court_y):
        """
        Convert court coordinates to image pixel coordinates

        Args:
            court_x: X coordinate in meters [0, 15] (left to right)
            court_y: Y coordinate in meters [0, 14] (baseline to half-court)

        Returns:
            (img_x, img_y): Pixel coordinates in the court image
        """
        if self.court_img_width is None or self.court_img_height is None:
            return None, None

        from constants import COURT_WIDTH, COURT_HALF_LENGTH

        # Convert from court coordinates to image coordinates
        # Court: X ∈ [0, 15]m, Y ∈ [0, 14]m (baseline to half-court)
        # Image: top-left = baseline-left, bottom-left = half-court-left (Y-axis is FLIPPED!)

        # X: [0, 15] -> [0, img_width] (left to right)
        normalized_x = court_x / COURT_WIDTH

        # Y: [0, 14] -> [0, img_height] (NOT flipped - 0 at top, 14 at bottom)
        normalized_y = court_y / COURT_HALF_LENGTH

        img_x = int(normalized_x * self.court_img_width)
        img_y = int(normalized_y * self.court_img_height)

        return img_x, img_y

    def visualize_shot(self, court_location, is_made=True, timestamp=None, save_path='shot_visualizations'):
        """
        Visualize a shot on the court diagram (deprecated - shots are now stored in shot_data)

        This method is kept for backward compatibility but no longer generates individual
        shot images. Use get_shot_chart() to generate the final shot chart with all shots.

        Args:
            court_location: (x, y) coordinates in court space
            is_made: Whether the shot was made (green) or missed (red)
            timestamp: Optional timestamp string for labeling
            save_path: Directory to save visualization images (not used)

        Returns:
            None (no longer generates individual shot images)
        """
        if not self.enable_visualization:
            return None

        court_x, court_y = court_location
        if court_x is None or court_y is None:
            logger.log(INFO, "Cannot visualize shot: Invalid court location")
            return None

        # Shots are now stored in shot_data during detection
        # Individual shot visualization is deprecated - use get_shot_chart() instead
        logger.log(INFO, f"Shot logged at ({court_x:.2f}m, {court_y:.2f}m) - {'Made' if is_made else 'Missed'}")

        return None

    def get_shot_chart(self, save_path='shot_chart.png'):
        """
        Generate a final shot chart with all shots using SVG-based renderer

        Args:
            save_path: Path to save the final shot chart

        Returns:
            Path to saved shot chart image
        """
        if not self.enable_visualization:
            return None

        if not self.shot_data:
            logger.log(INFO, "No shots to visualize")
            return None

        # Use the new SVG-based court renderer
        from court_renderer import CourtRenderer

        # Extract shot data: (x_m, y_m, is_made)
        shots = [
            (shot['court_position_meters']['x'],
             shot['court_position_meters']['y'],
             shot['is_made'])
            for shot in self.shot_data
            if shot.get('court_position_meters')
        ]

        if not shots:
            logger.log(INFO, "No valid shots with court coordinates")
            return None

        # Generate statistics for title
        made_count = sum(1 for _, _, is_made in shots if is_made)
        total_count = len(shots)
        pct = (made_count / total_count * 100) if total_count > 0 else 0
        title = f"Shot Chart: {made_count}/{total_count} ({pct:.1f}%)"

        # Create shot chart using SVG renderer
        CourtRenderer.create_shot_chart(
            shots=shots,
            title=title,
            save_path=save_path,
            show_stats=True
        )

        logger.log(INFO, f"Final shot chart saved to: {save_path}")
        return save_path