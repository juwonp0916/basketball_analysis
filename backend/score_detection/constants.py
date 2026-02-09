# FIBA International Half-Court Dimensions (in meters)

# Court dimensions
COURT_WIDTH = 15.0  # meters (full width)
COURT_HALF_LENGTH = 14.0  # meters (baseline to half-court line)

# Coordinate system: Origin at left corner of baseline
# X: 0m (left) to 15m (right)
# Y: 0m (baseline) to +14m (half-court line)

# Basket position
BASKET_X = 7.5  # Center of court (0-15 range)
BASKET_Y = 1.575  # meters from baseline

# Penalty box / Key dimensions
PENALTY_BOX_WIDTH = 5.8  # meters
PENALTY_BOX_DEPTH = 5.8  # meters (from baseline to free throw line)
PENALTY_BOX_HALF_WIDTH = PENALTY_BOX_WIDTH / 2  # 2.9m

# Free throw line
FREE_THROW_LINE_Y = 5.8  # meters from baseline

# 3-point line
THREE_PT_ARC_RADIUS = 6.75  # meters from basket center
THREE_PT_CORNER_DISTANCE = 6.6  # meters along baseline direction

# 6-Point Calibration System
# Points in clockwise order: Baseline (L→R), then Free Throw Line (L→R)
CALIBRATION_POINTS_FIBA = [
    (0.0, 0.0),    # 1. Baseline Left Sideline
    (4.6, 0.0),    # 2. Baseline Left Penalty (7.5-2.9)
    (10.4, 0.0),   # 3. Baseline Right Penalty (7.5+2.9)
    (15.0, 0.0),   # 4. Baseline Right Sideline
    (4.6, 5.8),    # 5. FT Line Left Penalty
    (10.4, 5.8),   # 6. FT Line Right Penalty
]

CALIBRATION_LABELS = [
    "Baseline Left Sideline",
    "Baseline Left Penalty Box",
    "Baseline Right Penalty Box",
    "Baseline Right Sideline",
    "Free Throw Line Left",
    "Free Throw Line Right",
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
