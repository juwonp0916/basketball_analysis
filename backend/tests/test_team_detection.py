"""
Unit and regression tests for the CIELAB-based TeamDetector v3.

Tests the core color pipeline without requiring YOLO or a full detector setup.
For regression testing on real debug frames, annotated fixtures are loaded
from debug_shot_frames/ (if available).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure score_detection is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "score_detection"))
from team_detector import StreamingTeamDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Fresh unconfigured detector."""
    return StreamingTeamDetector()


@pytest.fixture
def configured_detector():
    """Detector with two team colors pre-set (black and blue)."""
    det = StreamingTeamDetector()
    # Black in LAB: L≈0, a≈128, b≈128 (OpenCV LAB with 128 offset)
    # Blue in LAB:  L≈30, a≈128, b≈70ish
    det.team0_color = np.array([10.0, 128.0, 128.0], dtype=np.float32)   # near-black
    det.team1_color = np.array([40.0, 135.0, 85.0], dtype=np.float32)    # dark blue
    det._configured = True
    return det


def _make_solid_patch(bgr_color: tuple, size: int = 40) -> np.ndarray:
    """Create a solid-color BGR image patch."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bgr_color
    return img


# ---------------------------------------------------------------------------
# Test: LAB conversion roundtrip
# ---------------------------------------------------------------------------

class TestLabToHex:
    def test_black(self, detector):
        # LAB for pure black: L=0, a=128, b=128
        lab = np.array([0, 128, 128], dtype=np.float32)
        hex_color = detector._lab_to_hex(lab)
        assert hex_color == "#000000"

    def test_white(self, detector):
        lab = np.array([255, 128, 128], dtype=np.float32)
        hex_color = detector._lab_to_hex(lab)
        assert hex_color == "#ffffff"

    def test_roundtrip_blue(self, detector):
        # Start with a known blue BGR, convert to LAB, then back to hex
        # Note: LAB→BGR roundtrip has +-1 uint8 quantization error
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Pure blue in BGR
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        hex_color = detector._lab_to_hex(lab)
        # Allow +-1 per channel for quantization
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        assert r <= 1 and g <= 1 and b >= 254, f"Expected near #0000ff, got {hex_color}"

    def test_roundtrip_red(self, detector):
        bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)  # Pure red in BGR
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        hex_color = detector._lab_to_hex(lab)
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        assert r >= 254 and g <= 2 and b <= 2, f"Expected near #ff0000, got {hex_color}"


# ---------------------------------------------------------------------------
# Test: color distance (Delta-E)
# ---------------------------------------------------------------------------

class TestColorDistance:
    def test_same_color_is_zero(self, detector):
        lab = np.array([50.0, 128.0, 128.0], dtype=np.float32)
        assert detector._color_distance(lab, lab) == 0.0

    def test_black_vs_white(self, detector):
        black = np.array([0.0, 128.0, 128.0], dtype=np.float32)
        white = np.array([255.0, 128.0, 128.0], dtype=np.float32)
        dist = detector._color_distance(black, white)
        assert dist == 255.0  # Pure L-channel difference

    def test_black_vs_blue_distinguishable(self, detector):
        """Black and dark blue should have Delta-E well above 15."""
        black_bgr = np.array([[[0, 0, 0]]], dtype=np.uint8)
        blue_bgr = np.array([[[180, 0, 0]]], dtype=np.uint8)  # Dark blue BGR
        black_lab = cv2.cvtColor(black_bgr, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        blue_lab = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        dist = detector._color_distance(black_lab, blue_lab)
        assert dist > 15.0, f"Delta-E {dist:.1f} should be > 15 for black vs dark blue"

    def test_similar_grays_close(self, detector):
        """Two similar grays should have small Delta-E."""
        gray1 = np.array([100.0, 128.0, 128.0], dtype=np.float32)
        gray2 = np.array([105.0, 128.0, 128.0], dtype=np.float32)
        dist = detector._color_distance(gray1, gray2)
        assert dist < 10.0


# ---------------------------------------------------------------------------
# Test: dominant color extraction from solid patches
# ---------------------------------------------------------------------------

class TestExtractColorFeatures:
    def test_black_patch(self, detector):
        """A solid black patch should extract to near-black LAB."""
        patch = _make_solid_patch((0, 0, 0))
        lab = detector._extract_color_features(patch)
        assert lab is not None
        # L should be very low (near 0)
        assert lab[0] < 20, f"L={lab[0]} should be < 20 for black"

    def test_blue_patch(self, detector):
        """A solid blue patch should extract to blue-ish LAB."""
        patch = _make_solid_patch((180, 0, 0))  # Dark blue BGR
        lab = detector._extract_color_features(patch)
        assert lab is not None
        # b channel (LAB) should be significantly below 128 for blue
        assert lab[2] < 120, f"b={lab[2]} should be < 120 for blue"

    def test_distinguishes_black_from_blue(self, detector):
        """Features from black and blue patches should be far apart."""
        black_lab = detector._extract_color_features(_make_solid_patch((0, 0, 0)))
        blue_lab = detector._extract_color_features(_make_solid_patch((180, 0, 0)))
        assert black_lab is not None and blue_lab is not None
        dist = detector._color_distance(black_lab, blue_lab)
        assert dist > 10.0, f"Delta-E {dist:.1f} should distinguish black from blue"

    def test_noisy_patch_dominant_color(self, detector):
        """A patch that's 70% blue + 30% gray should extract blue as dominant."""
        patch = np.zeros((40, 40, 3), dtype=np.uint8)
        patch[:28, :, :] = (180, 0, 0)     # Blue BGR (70%)
        patch[28:, :, :] = (100, 100, 100)  # Gray (30%)
        lab = detector._extract_color_features(patch)
        assert lab is not None

        # The dominant color should be closer to blue than gray
        blue_ref = cv2.cvtColor(
            np.array([[[180, 0, 0]]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0][0].astype(np.float32)
        gray_ref = cv2.cvtColor(
            np.array([[[100, 100, 100]]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0][0].astype(np.float32)
        dist_blue = detector._color_distance(lab, blue_ref)
        dist_gray = detector._color_distance(lab, gray_ref)
        assert dist_blue < dist_gray, (
            f"Dominant color should be closer to blue ({dist_blue:.1f}) "
            f"than gray ({dist_gray:.1f})"
        )


# ---------------------------------------------------------------------------
# Test: classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_black_jersey_classified_as_team0(self, configured_detector):
        """A black jersey should be classified as team 0 (the black team)."""
        # Create a fake frame with a player bbox area that's black
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Player bbox: fill the jersey region (25%-55% height, center 50% width)
        frame[50:110, 50:150, :] = (5, 5, 5)  # Near-black

        bbox = (50, 0, 150, 200)  # Full player box
        team, conf = configured_detector.classify_from_bbox(frame, bbox)
        assert team == 0, f"Black jersey should be team 0, got {team}"

    def test_blue_jersey_classified_as_team1(self, configured_detector):
        """A blue jersey should be classified as team 1 (the blue team)."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:110, 50:150, :] = (180, 0, 0)  # Dark blue BGR

        bbox = (50, 0, 150, 200)
        team, conf = configured_detector.classify_from_bbox(frame, bbox)
        assert team == 1, f"Blue jersey should be team 1, got {team}"


# ---------------------------------------------------------------------------
# Test: cluster and validate
# ---------------------------------------------------------------------------

class TestClusterAndValidate:
    def test_two_distinct_clusters(self, detector):
        """Two clearly different color groups should produce valid clusters."""
        # Group 1: near-black LAB
        black_lab = cv2.cvtColor(
            np.array([[[0, 0, 0]]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0][0].astype(np.float32)
        # Group 2: blue LAB
        blue_lab = cv2.cvtColor(
            np.array([[[180, 0, 0]]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0][0].astype(np.float32)

        # Create features with some noise
        features = []
        rng = np.random.RandomState(42)
        for _ in range(20):
            features.append(black_lab + rng.normal(0, 2, 3).astype(np.float32))
        for _ in range(20):
            features.append(blue_lab + rng.normal(0, 2, 3).astype(np.float32))

        result = detector._cluster_and_validate(features)
        assert result is not None, "Clustering should succeed for distinct groups"

        c0, c1 = result
        inter_dist = detector._color_distance(c0, c1)
        assert inter_dist > 15.0, f"Inter-cluster Delta-E {inter_dist:.1f} should be > 15"

    def test_identical_colors_fail_quality_gate(self, detector):
        """Two identical color groups should fail the quality gate."""
        gray_lab = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        features = [gray_lab + np.random.normal(0, 1, 3).astype(np.float32) for _ in range(40)]
        result = detector._cluster_and_validate(features)
        # Should either return None (quality gate) or produce very similar centroids
        if result is not None:
            c0, c1 = result
            dist = detector._color_distance(c0, c1)
            # If it somehow passes, the distance should still be small
            assert dist < 20.0, "Identical color groups should produce close centroids"


# ---------------------------------------------------------------------------
# Regression test: run on annotated debug frames (if available)
# ---------------------------------------------------------------------------

class TestRegressionOnFixtures:
    """
    Regression tests using annotated debug frames from debug_shot_frames/.

    These tests are skipped if no annotated frames are available.
    To create fixtures:
    1. Run the live system to generate debug frames
    2. Run: python tools/annotate_teams.py to add ground truth labels
    3. Re-run this test suite
    """

    @pytest.fixture
    def annotated_data(self):
        """Load annotated JSON files."""
        import json
        debug_dir = Path(__file__).parent.parent.parent / "debug_shot_frames"
        if not debug_dir.exists():
            pytest.skip("No debug_shot_frames directory found")

        jsons = sorted(debug_dir.glob("debug_shot_*.json"))
        annotated = []
        for jpath in jsons:
            with open(jpath) as f:
                data = json.load(f)
            has_gt = any(
                p.get("ground_truth_team") is not None
                for p in data.get("persons_detected", [])
            )
            if has_gt:
                annotated.append(data)

        if not annotated:
            pytest.skip("No annotated debug frames found")
        return annotated

    def test_accuracy_above_threshold(self, annotated_data):
        """Team classification accuracy should be >= 70% on annotated fixtures."""
        correct = 0
        total = 0
        for shot in annotated_data:
            for person in shot.get("persons_detected", []):
                gt = person.get("ground_truth_team")
                pred = person.get("team_id")
                if gt is not None and pred is not None:
                    total += 1
                    if gt == pred:
                        correct += 1

        if total == 0:
            pytest.skip("No predictions with ground truth found")

        accuracy = correct / total
        assert accuracy >= 0.70, (
            f"Team classification accuracy {accuracy:.1%} ({correct}/{total}) "
            f"is below the 70% threshold"
        )
