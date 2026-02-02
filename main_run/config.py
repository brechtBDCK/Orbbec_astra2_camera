"""Top-level options for the Astra2 basic runner.

Edit these values directly. This keeps the script simple and avoids CLI args.
"""

from pyorbbecsdk import OBHoleFillingMode

# ============================================================================
# High-level behavior
# ============================================================================
# MODE = "depth_with_color"      -> view depth + color (RGB-D)
# MODE = "pointcloud_with_color" -> generate depth-only point clouds
MODE = "pointcloud_with_color"

# Single switch for printing info offline and exiting.
# When True: prints device info, profiles, and recommended filters, then exits.
# When False: runs live mode.
PRINT_INFO_OFFLINE = False

# ============================================================================
# Profile selection
# ============================================================================
# If PROFILE_INDEX is set, width/height/fps are taken from that profile.
# You do NOT need to set width/height/fps in that case.
DEPTH_PROFILE_INDEX = None  # e.g., 0, 1, 2...
DEPTH_WIDTH = None          # e.g., 640
DEPTH_HEIGHT = None         # e.g., 400
DEPTH_FPS = None            # e.g., 30
DEPTH_FORMAT_NAME = "Y16"    # e.g., "Y16", 

COLOR_PROFILE_INDEX = 1
COLOR_WIDTH = None
COLOR_HEIGHT = None
COLOR_FPS = None
COLOR_FORMAT_NAME = "RGB"    # e.g., "RGB", "MJPG", "NV12"


# ============================================================================
# Depth filters (software post-processing)
# ============================================================================
# --- DecimationFilter ---
ENABLE_DECIMATION = False
DECIMATION_SCALE_VALUE = None  # e.g., 2, 4, 8 (None = SDK default)
# --- DisparityTransform ---
ENABLE_DISPARITY = True

# --- SpatialAdvancedFilter ---
ENABLE_SPATIAL = False
# Higher alpha/diff/magnitude/radius => stronger smoothing (but more blur).
SPATIAL_ALPHA = 0.5
SPATIAL_DIFF_THRESHOLD = 160
SPATIAL_MAGNITUDE = 1
SPATIAL_RADIUS = 1

# --- TemporalFilter ---
ENABLE_TEMPORAL = True
# Lower diff/weight => more smoothing (but more lag).
TEMPORAL_DIFF_SCALE = 0.1
TEMPORAL_WEIGHT = 0.4

# --- HoleFillingFilter ---
ENABLE_HOLE_FILLING = False
HOLE_FILLING_MODE = OBHoleFillingMode.NEAREST  # FURTHEST / NEAREST / TOP

# --- ThresholdFilter ---
ENABLE_THRESHOLD = True
# Optional depth range clamp (units depend on device; often mm scale in Y16)
THRESHOLD_MIN = 200  #20cm
THRESHOLD_MAX = 1000 #1m


# ============================================================================
# Display options
# ============================================================================
# Discard the first N frames after stream start (sensor warmup).
WARMUP_FRAMES = 10

SHOW_WINDOW = True
PRINT_INTERVAL_S = 1.0
TIMEOUT_MS = 100
MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000
WINDOW_NAME = "Depth"
WINDOW_POS_X = 50
WINDOW_POS_Y = 50

# If True, scale colors per-frame using valid depth range for better gradients.
AUTO_SCALE = True #Set to true when using the threshold filter to see depth variations better.

# ============================================================================
# Point-cloud options
# ============================================================================
# Average N point cloud frames before saving.
POINTCLOUD_AVERAGE_N_FRAMES = 1 #keep 1, do averaging in the temporal depth domain, not pointcloud domain.

# Save PLY files to disk.
POINTCLOUD_SAVE_PLY = True
POINTCLOUD_SAVE_DIR = "point_clouds"
POINTCLOUD_SAVE_PREFIX = "astra2_pc"
