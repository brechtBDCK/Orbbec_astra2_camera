"""Runtime helper functions for main run loop."""

from __future__ import annotations

import cv2
import numpy as np

from pyorbbecsdk import Pipeline


def print_device_info(pipeline: Pipeline) -> None:
    """Print a quick summary of the connected device."""
    try:
        info = pipeline.get_device().get_device_info()
    except Exception as exc:
        print(f"Device info unavailable: {exc}")
        return
    print("\nDevice:")
    print(f"  name: {info.get_name()}")
    print(f"  pid:  {info.get_pid()}  vid: {info.get_vid()}")
    print(f"  sn:   {info.get_serial_number()}")
    print(f"  fw:   {info.get_firmware_version()}")



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


def depth_mm_to_vis(depth_mm: np.ndarray, display_cfg: dict) -> np.ndarray:
    """Convert a depth-mm image to a colored visualization."""
    display_cfg = display_cfg or {}
    auto_scale = bool(display_cfg.get("auto_scale", True))
    max_depth_mm = int(display_cfg.get("max_depth_mm", 10000))
    valid = depth_mm[depth_mm > 0]
    if auto_scale and valid.size:
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            vmax = vmin + 1e-3
        depth_vis = np.clip(depth_mm, vmin, vmax)
        depth_8u = ((depth_vis - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        depth_vis = np.clip(depth_mm, 0, max_depth_mm)
        depth_8u = (depth_vis / max_depth_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
