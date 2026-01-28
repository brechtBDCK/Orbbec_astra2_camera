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
    OBFormat,
    OBSensorType,
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
ENABLE_SPATIAL = True
# (No simple parameters exposed here in the Python binding.)

# --- TemporalFilter ---
ENABLE_TEMPORAL = True
# (No simple parameters exposed here in the Python binding.)

# --- HoleFillingFilter ---
ENABLE_HOLE_FILLING = False
# (No simple parameters exposed here in the Python binding.)

# --- ThresholdFilter (min/max depth clamp, in millimeters) ---
ENABLE_THRESHOLD = True
THRESHOLD_MIN = 200 #20cm
THRESHOLD_MAX = 1000 #100cm

SHOW_WINDOW = True
PRINT_INTERVAL_S = 1.0
TIMEOUT_MS = 100
MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000

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

    # Build filter chain based on toggles
    filters = []
    if ENABLE_DISPARITY:
        filters.append(DisparityTransform())
    if ENABLE_SPATIAL:
        filters.append(SpatialAdvancedFilter())
    if ENABLE_TEMPORAL:
        filters.append(TemporalFilter())
    if ENABLE_HOLE_FILLING:
        filters.append(HoleFillingFilter())
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
                cv2.imshow("Depth", depth_color)
                key = cv2.waitKey(1)
                if key in (27, ord('q')):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
