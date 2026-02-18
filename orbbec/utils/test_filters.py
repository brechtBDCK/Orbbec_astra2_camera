"""Simple depth output with optional SDK filters.

Edit the flags below to turn filters on/off.
"""
import time
import numpy as np
import cv2

from pyorbbecsdk import (
    Config,
    DisparityTransform,
    HoleFillingFilter,
    OBHoleFillingMode,
    OBFormat,
    OBSensorType,
    OBSpatialAdvancedFilterParams,
    Pipeline,
    SpatialAdvancedFilter,
    TemporalFilter,
    ThresholdFilter,
)

# ============================================================================
# User Options (grouped per filter)
# ============================================================================

# --- DisparityTransform ---
ENABLE_DISPARITY = True

# --- SpatialAdvancedFilter --- 
ENABLE_SPATIAL = False
# SpatialAdvancedFilter parameters (see Orbbec docs):
# - SPATIAL_ALPHA: smoothing strength. Higher = smoother (more blur), lower = preserves edges.
# - SPATIAL_DIFF_THRESHOLD: edge threshold. Higher = more smoothing across edges.
# - SPATIAL_MAGNITUDE: iteration count. Higher = stronger smoothing, more compute.
# - SPATIAL_RADIUS: neighborhood size. Higher = stronger hole filling/smoothing.
SPATIAL_ALPHA = 0.5
SPATIAL_DIFF_THRESHOLD = 160
SPATIAL_MAGNITUDE = 1
SPATIAL_RADIUS = 1

# --- TemporalFilter ---
ENABLE_TEMPORAL = True
# TemporalFilter parameters:
# - TEMPORAL_DIFF_SCALE: allowed inter-frame depth change. Lower = more stable, higher = more responsive.
# - TEMPORAL_WEIGHT: weight of current frame. Higher = shorter memory, lower = longer memory (smoother).
TEMPORAL_DIFF_SCALE = 0.1
TEMPORAL_WEIGHT = 0.4

# --- HoleFillingFilter ---
ENABLE_HOLE_FILLING = False
# HoleFillingFilter mode:
# - FURTHEST: fill with farthest neighbor (preserves background)
# - NEAREST: fill with nearest neighbor (preserves foreground)
# - TOP: fill with top (up) neighbor
HOLE_FILLING_MODE = OBHoleFillingMode.NEAREST  # FURTHEST / NEAREST / TOP

# --- ThresholdFilter (min/max depth clamp, in millimeters) ---
ENABLE_THRESHOLD = True
THRESHOLD_MIN = 200 #20cm
THRESHOLD_MAX = 1000 #100cm

# --- Display options ---
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
# Time sync (helps reduce timestamp anomaly warnings)
# ============================================================================
ENABLE_TIME_SYNC = True
SYNC_INTERVAL_S = 60


# ============================================================================
# Helpers
# ============================================================================

def apply_filters(depth_frame, filters):
    """Apply enabled filters in sequence, returning a depth frame."""
    current = depth_frame
    for f in filters:
        if f is None:
            continue
        try:
            new_frame = f.process(current)
        except Exception:
            continue
        if new_frame is None:
            continue
        try:
            current = new_frame.as_depth_frame()
        except Exception:
            current = new_frame
    return current


def main():
    config = Config()
    pipeline = Pipeline()

    # Enable default depth stream
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.start(config)

    # Optional periodic host-device time sync
    last_sync = time.time()
    if ENABLE_TIME_SYNC:
        try:
            device = pipeline.get_device()
            device.timer_sync_with_host()
            last_sync = time.time()
        except Exception as exc:
            print(f"Time sync failed: {exc}")

    # Fix window position (avoids random placement)
    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WINDOW_NAME, WINDOW_POS_X, WINDOW_POS_Y)

    # Build filter chain based on toggles
    filters = []
    if ENABLE_DISPARITY:
        filters.append(DisparityTransform())
    if ENABLE_SPATIAL:
        sf = SpatialAdvancedFilter()
        params = OBSpatialAdvancedFilterParams()
        params.alpha = float(SPATIAL_ALPHA)
        params.disp_diff = int(SPATIAL_DIFF_THRESHOLD)
        params.magnitude = int(SPATIAL_MAGNITUDE)
        params.radius = int(SPATIAL_RADIUS)
        sf.set_filter_params(params)
        filters.append(sf)
    if ENABLE_TEMPORAL:
        tf = TemporalFilter()
        tf.set_diff_scale(float(TEMPORAL_DIFF_SCALE))
        tf.set_weight(float(TEMPORAL_WEIGHT))
        filters.append(tf)
    if ENABLE_HOLE_FILLING:
        hf = HoleFillingFilter()
        hf.set_filling_mode(HOLE_FILLING_MODE)
        filters.append(hf)
    if ENABLE_THRESHOLD:
        tf = ThresholdFilter()
        tf.set_value_range(int(THRESHOLD_MIN), int(THRESHOLD_MAX))
        filters.append(tf)

    # Print filter states
    print("Filters enabled:")
    for f in filters:
        try:
            print(f"  - {f.get_name()}")
        except Exception:
            print(f"  - {f}")

    last_print = 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames(TIMEOUT_MS)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            if depth_frame.get_format() != OBFormat.Y16:
                continue

            depth_frame = apply_filters(depth_frame, filters)

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            if ENABLE_TIME_SYNC and (time.time() - last_sync) >= SYNC_INTERVAL_S:
                try:
                    device.timer_sync_with_host()
                    last_sync = time.time()
                except Exception as exc:
                    print(f"Time sync failed: {exc}")

            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))

            # Convert to millimeters (matches the SDK example behavior)
            depth_mm = depth_raw.astype(np.float32) * float(scale)
            depth_mm = np.where((depth_mm > MIN_DEPTH_MM) & (depth_mm < MAX_DEPTH_MM), depth_mm, 0)
            depth_mm = depth_mm.astype(np.uint16)

            now = time.time()
            if now - last_print >= PRINT_INTERVAL_S:
                center = int(depth_mm[height // 2, width // 2])
                print(f"center distance: {center} mm")
                last_print = now

            if SHOW_WINDOW:
                # Colorize depth: near/far -> different colors.
                # AUTO_SCALE gives more visible gradients if distances are clustered.
                valid = depth_mm[(depth_mm > 0)]
                if AUTO_SCALE and valid.size:
                    vmin = float(np.min(valid))
                    vmax = float(np.max(valid))
                    if vmax <= vmin:
                        vmax = vmin + 1e-3
                    depth_vis = np.clip(depth_mm, vmin, vmax)
                    depth_8u = ((depth_vis - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    depth_vis = np.clip(depth_mm, 0, MAX_DEPTH_MM)
                    depth_8u = (depth_vis / MAX_DEPTH_MM * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                cv2.imshow(WINDOW_NAME, depth_color)
                key = cv2.waitKey(1)
                if key in (27, ord('q')):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
