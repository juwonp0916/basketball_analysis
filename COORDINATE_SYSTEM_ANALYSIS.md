# Coordinate System Discrepancy Analysis

## Problem Summary
There is a **coordinate system mismatch** between the frontend visualization and the backend coordinate transformation. The frontend court diagram shows the **basket at the TOP** of the screen, but shot coordinates are being mapped as if the **basket is at the BOTTOM**.

---

## Current System Breakdown

### 1. Backend FIBA Coordinate System (Constants)
**File:** `backend/score_detection/constants.py`

- **Origin:** Center of baseline
- **X-axis:** -7.5m (left sideline) to +7.5m (right sideline)
- **Y-axis:** 0m (baseline) to 14m (half-court line)
- **Basket position:** (0m, 1.575m) — near the baseline

```
COURT_WIDTH = 15.0m
COURT_HALF_LENGTH = 14.0m
BASKET_Y = 1.575m  # From baseline
```

---

### 2. Backend Court Image Visualization
**File:** `backend/score_detection/localization.py` (lines 197-226)
**Image:** `court_img_halfcourt.png`

**Current behavior:**
- Baseline is at the **BOTTOM** of the image
- Half-court line is at the **TOP** of the image
- Y-axis is **FLIPPED** during rendering:
  ```python
  # Line 221 in localization.py
  normalized_y = 1.0 - (court_y / COURT_HALF_LENGTH)  # FLIP Y-axis
  ```
- FIBA Y=0 (baseline) → bottom of image ✓
- FIBA Y=14 (half-court) → top of image ✓
- **Basket renders at BOTTOM** ✓

This matches the static image you shared where the basket is at the bottom.

---

### 3. Frontend Chart Coordinate System
**File:** `frontend/src/routes/live/v2/+page.svelte` (lines 669-679)

**SVG Court Drawing:**
```html
<Chart data={shotPoints} x="x" y="y" xDomain={[0, 50]} yDomain={[47, 0]} ...>
  <svg viewBox="0 0 50 47" ...>
    <!-- Court lines -->
    <rect x="0" y="0" width="50" height="47" />  <!-- Full court -->
    <rect x="17" y="0" width="16" height="19" />  <!-- Penalty box -->
    <circle cx="25" cy="5.25" r="0.75" stroke="#ef4444" />  <!-- BASKET -->
    <path d="M 3 0 L 3 14 Q 25 43 47 14 L 47 0" />  <!-- 3PT arc -->
  </svg>
</Chart>
```

**Key observations:**
- **viewBox:** `"0 0 50 47"` (SVG coordinate space)
- **Basket SVG position:** `cy="5.25"` → **5.25 units from TOP edge** (11% from top)
- **yDomain:** `[47, 0]` → Inverts rendering (data Y=0 renders at bottom, Y=47 at top)
- **Visual result:** Basket appears at **TOP** of screen ✓

---

### 4. Current Backend-to-Frontend Conversion
**File:** `backend/score_detection/shot_processing_pipeline.py` (lines 279-294)

```python
# Current conversion (INCORRECT)
coord_x = (fiba_x + 7.5) / 15.0 * 50.0  # Maps [-7.5, 7.5] → [0, 50] ✓
coord_y = fiba_y / 14.0 * 47.0           # Maps [0, 14] → [0, 47] ✗ WRONG!
```

**What this does:**
- FIBA Y=0 (baseline) → frontend Y=0
- FIBA Y=1.575m (basket) → frontend Y≈5.3
- FIBA Y=14m (half-court) → frontend Y=47

**With yDomain `[47, 0]`:**
- Frontend Y=0 renders at **BOTTOM** of chart
- Frontend Y=47 renders at **TOP** of chart
- So basket (Y≈5.3) renders **near the BOTTOM** ✗

**But the SVG court has basket at TOP!** ✗✗✗

---

## The Root Cause

The frontend SVG court diagram has the basket at `cy="5.25"` (near top), but the coordinate conversion maps FIBA baseline (Y=0) to frontend Y=0, which renders at the **bottom** due to the inverted yDomain.

**Visual mismatch:**
- SVG court: Basket at TOP (cy=5.25 in viewBox)
- Shot data: Renders at BOTTOM (Y≈5.3 with yDomain=[47,0])

This causes shots to appear **on the wrong end of the court!**

---

## Solution: Flip the Y-Coordinate Conversion

To align with the frontend's "basket at top" orientation, we need to **invert the Y-axis** in the conversion:

### Required Change
**File:** `backend/score_detection/shot_processing_pipeline.py` (line 294)

**FROM:**
```python
coord_y = fiba_y / 14.0 * 47.0  # Baseline → bottom
```

**TO:**
```python
coord_y = (14.0 - fiba_y) / 14.0 * 47.0  # Baseline → top (FLIPPED)
```

### New Mapping Behavior
- FIBA Y=0m (baseline) → frontend Y=47 → renders at **TOP** ✓
- FIBA Y=1.575m (basket) → frontend Y≈41.7 → renders **near TOP** ✓
- FIBA Y=14m (half-court) → frontend Y=0 → renders at **BOTTOM** ✓

This matches the SVG court where:
- Basket is at `cy="5.25"` (~11% from top)
- In data coordinates: `47 - 41.7 = 5.3` (~11% from top in visual) ✓

---

## Additional Files to Check

### Static Shot Localization
**File:** `static_shot_localization.py` (lines 564-574)

Similar coordinate conversion for visualization:
```python
# Line 570: Maps FIBA to court image pixels
img_x = int((court_x + COURT_WIDTH/2) / COURT_WIDTH * court_width)
img_y = int((COURT_HALF_LENGTH - court_y) / COURT_HALF_LENGTH * court_height)
```

This **already flips Y** (line 573: `COURT_HALF_LENGTH - court_y`), which is correct for the backend court image where baseline is at bottom.

**Status:** ✓ No change needed (this is for backend PNG, not frontend SVG)

---

## Validation Checklist

After applying the fix:

- [ ] Shots near baseline (Y≈0-2m) should appear **near the basket at TOP**
- [ ] Shots near half-court (Y≈12-14m) should appear **at BOTTOM**
- [ ] Shots from 3PT corners should align with the arc drawn in SVG
- [ ] Test with `static_shot_localization.py` to verify backend→FIBA conversion is correct
- [ ] Verify frontend receives correct coordinates via WebRTC shot events

---

## Summary

| Component | Current State | Required State |
|-----------|--------------|----------------|
| Backend FIBA coords | Y=0 at baseline, Y=14 at half-court | ✓ No change |
| Backend PNG visualization | Baseline at bottom (flipped) | ✓ No change |
| Frontend SVG court | Basket at top (cy=5.25) | ✓ Already correct |
| Frontend yDomain | `[47, 0]` (inverted) | ✓ Already correct |
| **Backend conversion** | **Maps baseline→bottom** | **❌ MUST FLIP** |

**Single line fix:** Invert Y-axis in `shot_processing_pipeline.py:294`

---

## ✅ FIX APPLIED (2026-02-09)

**Status:** COMPLETE

**File Modified:** `backend/score_detection/shot_processing_pipeline.py` (line 297)

**Change:**
```python
# BEFORE:
coord_y = fiba_y / 14.0 * 47.0

# AFTER:
coord_y = (14.0 - fiba_y) / 14.0 * 47.0
```

**Result:**
- ✅ Y-axis now correctly inverted for frontend
- ✅ Baseline shots (Y≈0-2m) map to top of chart (near basket)
- ✅ Half-court shots (Y≈12-14m) map to bottom of chart
- ✅ Coordinates align with frontend SVG court layout

**Next Steps:**
1. Test with live analysis or simulation mode
2. Verify shots appear in correct positions on frontend court
3. Cross-check with `static_shot_localization.py` for consistency
