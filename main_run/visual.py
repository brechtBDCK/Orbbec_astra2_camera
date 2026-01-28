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


def depth_frame_to_meters(depth_frame) -> np.ndarray | None:
    """Convert a Y16 depth frame into meters as float32."""
    if depth_frame is None:
        return None
    if depth_frame.get_format() != OBFormat.Y16:
        return None
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    return depth_raw.astype(np.float32) * float(scale)


def depth_to_colormap(depth_m: np.ndarray, max_vis_m: float) -> np.ndarray:
    """Render depth as a simple color map for quick visualization."""
    clipped = np.clip(depth_m, 0.0, max_vis_m)
    depth_8u = (clipped / max_vis_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
