"""
Coordinate System Verification Script

Tests the FIBA-to-Frontend coordinate conversion to ensure shots appear
in the correct positions on the frontend court visualization.
"""

def court_to_frontend(court_x, court_y):
    """
    Convert court coordinates to frontend chart coordinates

    Args:
        court_x: X coordinate in meters [0, 15]
        court_y: Y coordinate in meters [0, 14]

    Returns:
        (coord_x, coord_y) in frontend chart space [0, 50] x [0, 47]
    """
    coord_x = court_x / 15.0 * 50.0
    coord_y = (14.0 - court_y) / 14.0 * 47.0
    return (coord_x, coord_y)


def test_conversions():
    """Test key court positions"""

    print("=" * 70)
    print("COORDINATE CONVERSION VERIFICATION")
    print("=" * 70)
    print()

    # Test cases: (name, court_x, court_y, expected behavior)
    test_cases = [
        ("Basket (center, near baseline)", 7.5, 1.575, "Should be near TOP center"),
        ("Left baseline corner", 0.0, 0.0, "Should be TOP-LEFT corner"),
        ("Right baseline corner", 15.0, 0.0, "Should be TOP-RIGHT corner"),
        ("Center baseline", 7.5, 0.0, "Should be TOP center"),
        ("Left half-court corner", 0.0, 14.0, "Should be BOTTOM-LEFT corner"),
        ("Right half-court corner", 15.0, 14.0, "Should be BOTTOM-RIGHT corner"),
        ("Center half-court", 7.5, 14.0, "Should be BOTTOM center"),
        ("Free throw line center", 7.5, 5.8, "Should be upper-middle"),
        ("Left 3PT corner", 8.4, 0.9, "Should be near TOP-LEFT (inside corner)"),
        ("Right 3PT corner", 6.6, 0.9, "Should be near TOP-RIGHT (inside corner)"),
        ("Top of key (left)", 4.6, 5.8, "Should be left of center, mid-upper"),
        ("Top of key (right)", 10.4, 5.8, "Should be right of center, mid-upper"),
    ]

    print(f"{'Location':<35} {'Court (m)':<20} {'Frontend':<20} {'Expected'}")
    print("-" * 110)

    for name, court_x, court_y, expected in test_cases:
        coord_x, coord_y = court_to_frontend(court_x, court_y)

        # Determine visual position
        x_pos = "LEFT" if coord_x < 16.7 else "CENTER" if coord_x < 33.3 else "RIGHT"
        y_pos = "TOP" if coord_y > 31.3 else "MIDDLE" if coord_y > 15.7 else "BOTTOM"
        position = f"{y_pos}-{x_pos}" if x_pos != "CENTER" else y_pos

        print(f"{name:<35} ({court_x:>5.1f}, {court_y:>5.1f})  "
              f"->  ({coord_x:>5.1f}, {coord_y:>5.1f})  "
              f"{position:<15} <- {expected}")

    print()
    print("=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    print()

    # Critical validation checks
    checks = []

    # Check 1: Basket should be in upper portion (Y > 35)
    basket_x, basket_y = court_to_frontend(7.5, 1.575)
    check1 = basket_y > 35
    checks.append(("Basket in upper region (Y > 35)", check1, f"Y={basket_y:.1f}"))

    # Check 2: Baseline Y should map to top (Y ≈ 47)
    baseline_x, baseline_y = court_to_frontend(7.5, 0.0)
    check2 = abs(baseline_y - 47.0) < 0.1
    checks.append(("Baseline maps to top (Y ≈ 47)", check2, f"Y={baseline_y:.1f}"))

    # Check 3: Half-court should map to bottom (Y ≈ 0)
    halfcourt_x, halfcourt_y = court_to_frontend(7.5, 14.0)
    check3 = abs(halfcourt_y - 0.0) < 0.1
    checks.append(("Half-court maps to bottom (Y ≈ 0)", check3, f"Y={halfcourt_y:.1f}"))

    # Check 4: Left sideline should map to left (X ≈ 0)
    left_x, left_y = court_to_frontend(0.0, 7.0)
    check4 = abs(left_x - 0.0) < 0.1
    checks.append(("Left sideline maps to left (X ≈ 0)", check4, f"X={left_x:.1f}"))

    # Check 5: Right sideline should map to right (X ≈ 50)
    right_x, right_y = court_to_frontend(15.0, 7.0)
    check5 = abs(right_x - 50.0) < 0.1
    checks.append(("Right sideline maps to right (X ≈ 50)", check5, f"X={right_x:.1f}"))

    # Check 6: Free throw line should be in upper half
    ft_x, ft_y = court_to_frontend(7.5, 5.8)
    check6 = ft_y > 23.5  # Should be above middle
    checks.append(("Free throw line in upper half", check6, f"Y={ft_y:.1f}"))

    # Display results
    all_passed = True
    for check_name, passed, details in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:<10} {check_name:<45} {details}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("[SUCCESS] ALL CHECKS PASSED! Coordinate system is correctly configured.")
    else:
        print("[WARNING] SOME CHECKS FAILED! Review the coordinate conversion.")

    print("=" * 70)
    print()

    return all_passed


if __name__ == "__main__":
    test_conversions()
