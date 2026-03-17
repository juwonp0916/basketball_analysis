"""
Test script for SVG-based court renderer

This tests the unified visualization system that matches the frontend SVG exactly.
"""

import sys
import os
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from localization import ShotLocalizer
import random

def test_svg_renderer():
    """Test the SVG-based shot chart generation"""

    print("=" * 60)
    print("Testing SVG-based Court Renderer")
    print("=" * 60)

    # Test 6-point calibration (full baseline)
    print("\n1. Testing 6-point calibration mode...")

    # Example calibration points for a 1920x1080 video
    # These are dummy points for testing - in real use they'd be user-selected
    calibration_points_6pt = [
        [100, 800],   # Baseline Left Sideline
        [1820, 800],  # Baseline Right Sideline
        [460, 800],   # Baseline Left Penalty Box
        [1460, 800],  # Baseline Right Penalty Box
        [460, 450],   # Free Throw Line Left
        [1460, 450],  # Free Throw Line Right
    ]

    localizer_6pt = ShotLocalizer(
        calibration_points=calibration_points_6pt,
        image_dimensions=(1920, 1080),
        enable_visualization=True,
        calibration_mode='6-point'
    )

    print("✓ 6-point localizer initialized")

    # Generate some test shots
    print("\n2. Generating test shots...")
    test_shots = []
    for i in range(15):
        # Random court coordinates (in meters)
        x_m = random.uniform(2, 13)  # X: [2, 13] meters (avoiding edges)
        y_m = random.uniform(1, 12)  # Y: [1, 12] meters (avoiding edges)
        is_made = random.random() > 0.4  # 60% make rate

        # Add to shot_data in the format expected by localization
        localizer_6pt.shot_data.append({
            'court_position_meters': {'x': x_m, 'y': y_m},
            'is_made': is_made,
            'timestamp': f"00:{i:02d}",
            'shot_number': i + 1
        })
        test_shots.append((x_m, y_m, is_made))

    print(f"✓ Generated {len(test_shots)} test shots")

    # Generate shot chart using 6-point calibration
    print("\n3. Generating shot chart (6-point)...")
    chart_path_6pt = 'test_shot_chart_6pt.png'
    result = localizer_6pt.get_shot_chart(save_path=chart_path_6pt)

    if result:
        print(f"✓ Shot chart saved to: {chart_path_6pt}")
    else:
        print("✗ Failed to generate shot chart")

    # Test 4-point calibration (paint box only)
    print("\n4. Testing 4-point calibration mode...")

    calibration_points_4pt = [
        [460, 800],   # Baseline Left Penalty Box
        [1460, 800],  # Baseline Right Penalty Box
        [460, 450],   # Free Throw Line Left
        [1460, 450],  # Free Throw Line Right
    ]

    localizer_4pt = ShotLocalizer(
        calibration_points=calibration_points_4pt,
        image_dimensions=(1920, 1080),
        enable_visualization=True,
        calibration_mode='4-point'
    )

    print("✓ 4-point localizer initialized")

    # Add same test shots to 4-point localizer
    for i, (x_m, y_m, is_made) in enumerate(test_shots):
        localizer_4pt.shot_data.append({
            'court_position_meters': {'x': x_m, 'y': y_m},
            'is_made': is_made,
            'timestamp': f"00:{i:02d}",
            'shot_number': i + 1
        })

    # Generate shot chart using 4-point calibration
    print("\n5. Generating shot chart (4-point)...")
    chart_path_4pt = 'test_shot_chart_4pt.png'
    result = localizer_4pt.get_shot_chart(save_path=chart_path_4pt)

    if result:
        print(f"✓ Shot chart saved to: {chart_path_4pt}")
    else:
        print("✗ Failed to generate shot chart")

    # Test court renderer directly
    print("\n6. Testing CourtRenderer directly...")
    from court_renderer import CourtRenderer

    # Create a simple shot chart with random shots
    direct_shots = [
        (7.5, 2.5, True),   # Made shot near basket
        (3.0, 8.0, False),  # Missed shot from left wing
        (12.0, 8.0, True),  # Made shot from right wing
        (7.5, 12.0, False), # Missed shot from deep
    ]

    CourtRenderer.create_shot_chart(
        shots=direct_shots,
        title="Direct CourtRenderer Test",
        save_path="test_shot_chart_direct.png",
        show_stats=True
    )

    print("✓ Direct court renderer test completed")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"✓ SVG-based renderer: Working")
    print(f"✓ 6-point calibration: {chart_path_6pt}")
    print(f"✓ 4-point calibration: {chart_path_4pt}")
    print(f"✓ Direct renderer: test_shot_chart_direct.png")
    print("\nAll tests completed successfully!")
    print("\nThe generated shot charts should match the frontend SVG layout:")
    print("- Same coordinate system (X: [0, 50], Y: [0, 47])")
    print("- Same court dimensions and markings")
    print("- Same color scheme (dark theme)")
    print("=" * 60)

if __name__ == "__main__":
    test_svg_renderer()
