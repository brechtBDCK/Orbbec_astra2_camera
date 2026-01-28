"""Top-level options for the Astra2 basic runner.

Edit these values directly. This keeps the script simple and avoids CLI args.
"""

# ============================================================================
# High-level behavior
# ============================================================================
# MODE = "depth"  -> view depth (and optional color) frames
# MODE = "pointcloud" -> generate point clouds
MODE = "depth"

# If LIVE=True, open OpenCV windows and update continuously.
# If LIVE=False, run once and print a summary.
LIVE = True

# Set True to print all available profiles then exit.
LIST_PROFILES = False

# Print camera info at startup.
PRINT_DEVICE_INFO = True

# Print available profiles (without exiting).
PRINT_AVAILABLE_PROFILES = False

# Print available (recommended) filters for this device.
PRINT_AVAILABLE_FILTERS = False


# ============================================================================
# Stream selection
# ============================================================================
# Color stream is optional.
ENABLE_COLOR = False

# If True, align depth to color. This requires color stream enabled.
ALIGN_DEPTH_TO_COLOR = False


# ============================================================================
# Profile selection
# ============================================================================
# If PROFILE_INDEX is set, width/height/fps are taken from that profile.
# You do NOT need to set width/height/fps in that case.
DEPTH_PROFILE_INDEX = None  # e.g., 0, 1, 2...
DEPTH_WIDTH = None          # e.g., 640
DEPTH_HEIGHT = None         # e.g., 400
DEPTH_FPS = None            # e.g., 30
DEPTH_FORMAT_NAME = None    # e.g., "Y16"

COLOR_PROFILE_INDEX = None
COLOR_WIDTH = None
COLOR_HEIGHT = None
COLOR_FPS = None
COLOR_FORMAT_NAME = None    # e.g., "RGB", "MJPG", "NV12"


# ============================================================================
# Depth filters (software post-processing)
# ============================================================================
# Use device recommended filters OR select individual filters below.
USE_RECOMMENDED_FILTERS = False
ENABLE_DECIMATION = False
ENABLE_TEMPORAL = False
ENABLE_SPATIAL = False
ENABLE_HOLE_FILLING = False

# Optional depth range clamp (units depend on device; often mm scale in Y16)
THRESHOLD_MIN = None
THRESHOLD_MAX = None


# ============================================================================
# Visualization / loop behavior
# ============================================================================
TIMEOUT_MS = 100
MAX_VIS_M = 4.0
PRINT_INTERVAL_S = 1.0


# ============================================================================
# Point-cloud options
# ============================================================================
# If True and color stream is available, generate RGB point clouds.
POINTCLOUD_WITH_COLOR = True

# Normalize RGB colors in the SDK point cloud filter
POINTCLOUD_COLOR_NORMALIZE = True

# Save PLY files to disk
POINTCLOUD_SAVE_PLY = True
POINTCLOUD_SAVE_DIR = "point_clouds"
POINTCLOUD_SAVE_PREFIX = "astra2_pc"

# 0 => save once then stop. N>0 => save every N frames.
POINTCLOUD_SAVE_EVERY_N_FRAMES = 0
