"""Depth/color visualization helpers (OpenCV).

These functions convert SDK frames into numpy arrays for display.
"""

import numpy as np
import cv2

from pyorbbecsdk import FormatConvertFilter, OBConvertFormat, OBFormat


def determine_convert_format(fmt: OBFormat) -> OBConvertFormat | None:
    """Map camera color formats to converter formats."""
    mapping = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
        OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
        OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
    }
    return mapping.get(fmt)


def color_frame_to_bgr(color_frame) -> np.ndarray | None:
    """Convert a color frame to BGR for OpenCV display."""
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()

    if fmt == OBFormat.RGB:
        data = np.asanyarray(color_frame.get_data())
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if fmt == OBFormat.BGR:
        data = np.asanyarray(color_frame.get_data())
        return np.resize(data, (height, width, 3))

    # For non-RGB/BGR formats, use the SDK format converter.
    convert_format = determine_convert_format(fmt)
    if convert_format is None:
        return None

    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(color_frame)
    if rgb_frame is None:
        return None

    data = np.asanyarray(rgb_frame.get_data())
    rgb = np.resize(data, (height, width, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_depth_frame(obj):
    """Return a depth frame from either a frameset or a single frame."""
    if obj is None:
        return None
    if hasattr(obj, "get_depth_frame"):
        try:
            return obj.get_depth_frame()
        except Exception:
            pass
    if hasattr(obj, "as_depth_frame"):
        try:
            return obj.as_depth_frame()
        except Exception:
            pass
    return None


def extract_color_frame(obj):
    """Return a color frame from either a frameset or a single frame."""
    if obj is None:
        return None
    if hasattr(obj, "get_color_frame"):
        try:
            return obj.get_color_frame()
        except Exception:
            pass
    if hasattr(obj, "as_color_frame"):
        try:
            return obj.as_color_frame()
        except Exception:
            pass
    return None


def depth_frame_to_mm(depth_frame) -> np.ndarray | None:
    """Convert a Y16 depth frame into millimeters as uint16 (SDK-style)."""
    if depth_frame is None:
        return None
    if depth_frame.get_format() != OBFormat.Y16:
        return None
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_mm = depth_raw.astype(np.float32) * float(scale)
    return depth_mm.astype(np.uint16)


def stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize to a common height and stack horizontally."""
    if left is None:
        return right
    if right is None:
        return left
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    if lh != rh:
        target_h = max(lh, rh)
        if lh != target_h:
            new_w = max(1, int(lw * (target_h / float(lh))))
            left = cv2.resize(left, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if rh != target_h:
            new_w = max(1, int(rw * (target_h / float(rh))))
            right = cv2.resize(right, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([left, right])


def depth_to_vis(
    depth: np.ndarray,
    *,
    min_depth: float | None = None,
    max_depth: float | None = None,
    auto_scale: bool = True,
) -> np.ndarray:
    """Convert depth to a JET colormap for visualization."""
    depth_f = depth.astype(np.float32)
    valid = depth_f[depth_f > 0]

    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        if auto_scale:
            vmin = float(np.min(valid))
            vmax = float(np.max(valid)) if max_depth is None else float(max_depth)
        else:
            vmin = 0.0 if min_depth is None else float(min_depth)
            vmax = float(max_depth) if max_depth is not None else float(np.max(valid))
        if vmax <= vmin:
            vmax = vmin + 1.0

    depth_clipped = np.clip(depth_f, vmin, vmax)
    depth_8u = ((depth_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def depth_mm_to_vis(depth_mm: np.ndarray, display_cfg: dict) -> np.ndarray:
    """Convert a depth-mm image to a colored visualization."""
    display_cfg = display_cfg or {}
    auto_scale = bool(display_cfg.get("auto_scale", True))
    max_depth_mm = int(display_cfg.get("max_depth_mm", 10000))
    # Auto-scale uses the current valid depth range for better contrast.
    if auto_scale:
        return depth_to_vis(depth_mm, auto_scale=True)
    return depth_to_vis(depth_mm, auto_scale=False, min_depth=0.0, max_depth=max_depth_mm)


def clamp_depth_mm(depth_mm: np.ndarray, display_cfg: dict) -> np.ndarray:
    """Zero-out depth values outside the configured min/max range."""
    min_depth = display_cfg.get("min_depth_mm", 20)
    max_depth = display_cfg.get("max_depth_mm", 10000)
    return np.where((depth_mm > min_depth) & (depth_mm < max_depth), depth_mm, 0).astype(
        np.uint16
    )
