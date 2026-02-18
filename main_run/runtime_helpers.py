"""Runtime helper functions for main run loop."""

from __future__ import annotations

import cv2
import numpy as np

from pyorbbecsdk import Pipeline

import main_run.config as cfg


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


def summarize_depth(depth_m: np.ndarray) -> None:
    """Print simple statistics for a depth image in meters."""
    h, w = depth_m.shape
    center_m = float(depth_m[h // 2, w // 2])
    finite = depth_m[np.isfinite(depth_m)]
    d_min = float(np.min(finite)) if finite.size else float("nan")
    d_med = float(np.median(finite)) if finite.size else float("nan")
    d_max = float(np.max(finite)) if finite.size else float("nan")

    print("\nDepth summary:")
    print(f"  size:   {w}x{h}")
    print(f"  center: {center_m:.3f} m")
    print(f"  min:    {d_min:.3f} m")
    print(f"  median: {d_med:.3f} m")
    print(f"  max:    {d_max:.3f} m")


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


def depth_mm_to_vis(depth_mm: np.ndarray) -> np.ndarray:
    """Convert a depth-mm image to a colored visualization."""
    valid = depth_mm[depth_mm > 0]
    if cfg.AUTO_SCALE and valid.size:
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            vmax = vmin + 1e-3
        depth_vis = np.clip(depth_mm, vmin, vmax)
        depth_8u = ((depth_vis - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        depth_vis = np.clip(depth_mm, 0, cfg.MAX_DEPTH_MM)
        depth_8u = (depth_vis / cfg.MAX_DEPTH_MM * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
