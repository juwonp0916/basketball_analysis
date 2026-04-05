# NBA Half-Court Dimensions (in meters)

# Court dimensions
COURT_WIDTH = 15.0  # meters (full width)
COURT_HALF_LENGTH = 14.0  # meters (baseline to half-court line)

# Coordinate system: Origin at left corner of baseline
# X: 0m (left) to 15m (right)
# Y: 0m (baseline) to +14m (half-court line)

# Basket position
BASKET_X = 7.5  # Center of court (0-15 range)
BASKET_Y = 1.575  # meters from baseline

# Penalty box / Key dimensions (FIBA standard)
PENALTY_BOX_WIDTH = 4.9   # meters (side-to-side, FIBA: 4.9m wide)
PENALTY_BOX_DEPTH = 5.8   # meters (from baseline to free throw line)
PENALTY_BOX_HALF_WIDTH = PENALTY_BOX_WIDTH / 2  # 2.45m

# Free throw line
FREE_THROW_LINE_Y = 5.8  # meters from baseline

# 3-point line (NBA: 23.75 ft arc = 7.24m; corner straight section 3 ft from sideline = 0.9m)
THREE_PT_ARC_RADIUS = 7.24  # meters from basket center (NBA: 23.75 ft)
THREE_PT_CORNER_DISTANCE = 6.6  # meters from court center to straight section (7.5 - 0.9)

# 4-Point Court-Boundary Calibration System
# Click the 4 corners of the visible half-court rectangle.
# These are paint-box-independent: works for any court regardless of lane dimensions.
CALIBRATION_POINTS_FIBA_COURT = [
    (0.0,  0.0),   # 1. Baseline Left Corner  (where baseline meets left sideline)
    (15.0, 0.0),   # 2. Baseline Right Corner (where baseline meets right sideline)
    (0.0,  14.0),  # 3. Half-court Left Corner
    (15.0, 14.0),  # 4. Half-court Right Corner
]

CALIBRATION_LABELS_COURT = [
    "Baseline Left Corner",
    "Baseline Right Corner",
    "Half-court Left Corner",
    "Half-court Right Corner",
]

# 6-Point Calibration System
# Points in clockwise order: Baseline (L→R), then Free Throw Line (L→R)
CALIBRATION_POINTS_FIBA = [
    (0.0, 0.0),    # 1. Baseline Left Sideline
    (5.05, 0.0),   # 2. Baseline Left Penalty (7.5-2.45)
    (9.95, 0.0),   # 3. Baseline Right Penalty (7.5+2.45)
    (15.0, 0.0),   # 4. Baseline Right Sideline
    (5.05, 5.8),   # 5. FT Line Left Penalty
    (9.95, 5.8),   # 6. FT Line Right Penalty
]

CALIBRATION_LABELS = [
    "Baseline Left Sideline",
    "Baseline Left Penalty Box",
    "Baseline Right Penalty Box",
    "Baseline Right Sideline",
    "Free Throw Line Left",
    "Free Throw Line Right",
]

# 4-Point Calibration System (Paint Box Only)
# Use when full baseline sidelines are not visible
CALIBRATION_POINTS_FIBA_PAINT = [
    (5.05, 0.0),   # 1. Baseline Left Penalty Box
    (9.95, 0.0),   # 2. Baseline Right Penalty Box
    (5.05, 5.8),   # 3. Free Throw Line Left
    (9.95, 5.8),   # 4. Free Throw Line Right
]

CALIBRATION_LABELS_PAINT = [
    "Baseline Left Penalty Box",
    "Baseline Right Penalty Box",
    "Free Throw Line Left",
    "Free Throw Line Right",
]

# 6-Point Calibration: Paint Box + 3PT Arc Elbows
# Use this mode for accurate 3PT line localization on any court standard.
# Points 5 & 6 are the 3PT arc elbows: where the straight sideline sections
# transition to the curved arc (FIBA: 0.9m from sideline, y≈2.99m from baseline).
THREE_PT_ELBOW_X_LEFT  = 0.9    # meters from left sideline
THREE_PT_ELBOW_X_RIGHT = 14.1  # meters from left sideline (= 15 - 0.9)
# NBA: at x=0.9, solve (0.9-7.5)^2 + (y-1.575)^2 = 7.24^2 → y ≈ 4.55m
THREE_PT_ELBOW_Y = 4.55        # meters from baseline (where arc meets straight section, NBA)

CALIBRATION_POINTS_FIBA_PAINT_3PT = [
    (5.05, 0.0),              # 1. Baseline Left Paint Corner
    (9.95, 0.0),              # 2. Baseline Right Paint Corner
    (5.05, 5.8),              # 3. FT Line Left Corner
    (9.95, 5.8),              # 4. FT Line Right Corner
    (THREE_PT_ELBOW_X_LEFT,  THREE_PT_ELBOW_Y),  # 5. 3PT Left Corner
    (THREE_PT_ELBOW_X_RIGHT, THREE_PT_ELBOW_Y),  # 6. 3PT Right Corner
]

CALIBRATION_LABELS_PAINT_3PT = [
    "Baseline Left Paint Corner",
    "Baseline Right Paint Corner",
    "FT Line Left Corner",
    "FT Line Right Corner",
    "3PT Corner Left",
    "3PT Corner Right",
]

# Zone system (for shot statistics)
# Keep existing zone definitions but update coordinates if needed
WING_ZONE_Y = 4.0  # Approximate wing zone depth in meters
NUM_ZONES = 9

# Legacy constants for backward compatibility (deprecated)
# These are kept for existing code that hasn't been migrated yet
MAP_WIDTH = COURT_WIDTH  # 15.0m
MAP_HEIGHT = COURT_HALF_LENGTH  # 14.0m

# Penalty box coordinates in FIBA system (origin at center of baseline)
PENALTY_BOX_X1 = -PENALTY_BOX_HALF_WIDTH  # -2.9m (left edge)
PENALTY_BOX_X2 = PENALTY_BOX_HALF_WIDTH   # +2.9m (right edge)
PENALTY_BOX_Y1 = 0.0  # Baseline
PENALTY_BOX_Y2 = PENALTY_BOX_DEPTH  # 5.8m (free throw line)

# 3-point line boundaries (X coordinates where 3PT line meets baseline/near baseline)
# In FIBA, 3PT corner is ~6.6m horizontally from basket center
THREE_PT_LEFT = -THREE_PT_CORNER_DISTANCE  # -6.6m
THREE_PT_RIGHT = THREE_PT_CORNER_DISTANCE   # +6.6m
THREE_PT_RADIUS = THREE_PT_ARC_RADIUS  # Alias for backward compatibility

# Shot deduplication parameters
DEDUPLICATION_FRAME_THRESHOLD = 45  # frames (1.5 seconds @ 30fps)
DEDUPLICATION_POSITION_THRESHOLD = 50  # pixels
DEDUPLICATION_SAFETY_COOLDOWN_SEC = 1.0  # seconds
