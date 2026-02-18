"""Undistort a single color image and a single depth image using intrinsics JSON.

Edit COLOR_IMAGE_PATH / DEPTH_IMAGE_PATH below and run.
"""
import json
import os

import cv2
import numpy as np

from orbbec.visual import depth_to_vis, stack_side_by_side


COLOR_IMAGE_PATH = "/home/bdck/PROJECTS_WSL/Orbbec_astra2_camera/captures/rgbd/capture_20260218_112553_000001_color.png"
DEPTH_IMAGE_PATH = "/home/bdck/PROJECTS_WSL/Orbbec_astra2_camera/captures/rgbd/capture_20260218_112553_000001_depth.png"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_JSON_PATH = os.path.join(SCRIPT_DIR, "camera_intrinsics.json")

SHOW_WINDOWS = True
SAVE_DIR = None  # e.g. "outputs"
ALPHA = None  # getOptimalNewCameraMatrix alpha; None uses original K
DEPTH_MAX = None  # e.g. 5000 for 5m if depth is in mm


def load_calibration(path: str) -> dict:
    """Load a combined intrinsics JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_intrinsics(cfg: dict) -> dict:
    if "intrinsic" in cfg and isinstance(cfg["intrinsic"], dict):
        return cfg["intrinsic"]
    return cfg


def _extract_distortion(cfg: dict) -> dict:
    if "distortion" in cfg and isinstance(cfg["distortion"], dict):
        return cfg["distortion"]
    return cfg


def _extract_size(cfg: dict) -> tuple[int, int]:
    if "size" in cfg and isinstance(cfg["size"], (list, tuple)) and len(cfg["size"]) == 2:
        return int(cfg["size"][0]), int(cfg["size"][1])
    intr = _extract_intrinsics(cfg)
    w = intr.get("width")
    h = intr.get("height")
    if w is not None and h is not None:
        return int(w), int(h)
    w = cfg.get("width")
    h = cfg.get("height")
    if w is not None and h is not None:
        return int(w), int(h)
    return 0, 0


def _pick_cfg_by_size(calib: dict, kind: str, width: int, height: int) -> tuple[dict, str, bool]:
    key = f"{int(width)}x{int(height)}"
    by_size = calib.get(f"{kind}_by_size")
    if isinstance(by_size, dict) and key in by_size:
        return by_size[key], key, True
    print(f"No {kind} intrinsics for {key} in {CALIB_JSON_PATH}")
    return calib.get(kind, {}), key, False


def _format_intrinsics(intr: dict) -> str:
    keys = ["fx", "fy", "cx", "cy", "width", "height"]
    parts = []
    for k in keys:
        if k in intr:
            parts.append(f"{k}={intr[k]}")
    return ", ".join(parts) if parts else "n/a"


def _format_distortion(dist: dict) -> str:
    keys = ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"]
    parts = []
    for k in keys:
        if k in dist:
            parts.append(f"{k}={dist[k]}")
    return ", ".join(parts) if parts else "n/a"


def undistort_image(img: np.ndarray, K: np.ndarray, D: np.ndarray, alpha: float | None) -> np.ndarray:
    """Undistort an image using OpenCV, optionally keeping full FOV with alpha."""
    h, w = img.shape[:2]
    if alpha is None:
        new_K = K
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha)

    return cv2.undistort(img, K, D, None, new_K)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    """Overlay a small label in the top-left corner."""
    out = img.copy()
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def depth_to_vis_with_limit(depth: np.ndarray, max_depth: float | None) -> np.ndarray:
    """Wrapper for depth visualization with optional max clamp."""
    return depth_to_vis(depth, max_depth=max_depth, auto_scale=True)


def build_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Standard pinhole camera matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_distortion(dist: dict) -> np.ndarray:
    """Build OpenCV distortion vector (supports 5 or 8 params)."""
    k1 = float(dist.get("k1", 0.0))
    k2 = float(dist.get("k2", 0.0))
    p1 = float(dist.get("p1", 0.0))
    p2 = float(dist.get("p2", 0.0))
    k3 = float(dist.get("k3", 0.0))
    k4 = dist.get("k4")
    k5 = dist.get("k5")
    k6 = dist.get("k6")
    if k4 is not None or k5 is not None or k6 is not None:
        k4 = float(0.0 if k4 is None else k4)
        k5 = float(0.0 if k5 is None else k5)
        k6 = float(0.0 if k6 is None else k6)
        return np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
    return np.array([k1, k2, p1, p2, k3], dtype=np.float64)


def _warn_size(name: str, img: np.ndarray, expected: tuple[int, int]) -> None:
    h, w = img.shape[:2]
    if (w, h) != expected:
        print(f"{name} size is {w}x{h}, expected {expected[0]}x{expected[1]}.")


def main() -> int:
    calib = load_calibration(CALIB_JSON_PATH)

    if COLOR_IMAGE_PATH:
        color = cv2.imread(COLOR_IMAGE_PATH, cv2.IMREAD_COLOR)
        if color is None:
            print(f"Failed to read color image: {COLOR_IMAGE_PATH}")
            return 1
        color_cfg, color_key, color_match = _pick_cfg_by_size(calib, "color", color.shape[1], color.shape[0])
        color_size = _extract_size(color_cfg)
        if color_size != (0, 0):
            _warn_size("Color", color, (int(color_size[0]), int(color_size[1])))

        color_intr = _extract_intrinsics(color_cfg)
        color_dist = _extract_distortion(color_cfg)
        color_K = build_camera_matrix(
            float(color_intr.get("fx", 0.0)),
            float(color_intr.get("fy", 0.0)),
            float(color_intr.get("cx", 0.0)),
            float(color_intr.get("cy", 0.0)),
        )
        color_D = build_distortion(color_dist)

        print("\nColor image:")
        print(f"  path: {COLOR_IMAGE_PATH}")
        print(f"  size: {color.shape[1]}x{color.shape[0]}")
        print(f"  profile key: {color_key} (matched={color_match})")
        print(f"  intrinsics: {_format_intrinsics(color_intr)}")
        print(f"  distortion: {_format_distortion(color_dist)}")

        color_undist = undistort_image(color, color_K, color_D, ALPHA)
        color_orig = add_label(color, "color: original")
        color_ud = add_label(color_undist, "color: undistorted")
        color_pair = stack_side_by_side(color_orig, color_ud)

        if SAVE_DIR:
            out_path = f"{SAVE_DIR.rstrip('/')}/color_undistorted.png"
            cv2.imwrite(out_path, color_pair)
            print(f"Saved color comparison: {out_path}")

        if SHOW_WINDOWS:
            cv2.imshow("Color undistort", color_pair)

    if DEPTH_IMAGE_PATH:
        depth = cv2.imread(DEPTH_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Failed to read depth image: {DEPTH_IMAGE_PATH}")
            return 1
        depth_cfg, depth_key, depth_match = _pick_cfg_by_size(calib, "depth", depth.shape[1], depth.shape[0])
        depth_size = _extract_size(depth_cfg)
        if depth_size != (0, 0):
            _warn_size("Depth", depth, (int(depth_size[0]), int(depth_size[1])))

        depth_intr = _extract_intrinsics(depth_cfg)
        depth_dist = _extract_distortion(depth_cfg)
        depth_K = build_camera_matrix(
            float(depth_intr.get("fx", 0.0)),
            float(depth_intr.get("fy", 0.0)),
            float(depth_intr.get("cx", 0.0)),
            float(depth_intr.get("cy", 0.0)),
        )
        depth_D = build_distortion(depth_dist)

        print("\nDepth image:")
        print(f"  path: {DEPTH_IMAGE_PATH}")
        print(f"  size: {depth.shape[1]}x{depth.shape[0]}")
        print(f"  profile key: {depth_key} (matched={depth_match})")
        print(f"  intrinsics: {_format_intrinsics(depth_intr)}")
        print(f"  distortion: {_format_distortion(depth_dist)}")

        depth_undist = undistort_image(depth, depth_K, depth_D, ALPHA)
        depth_vis = depth_to_vis_with_limit(depth, DEPTH_MAX)
        depth_ud_vis = depth_to_vis_with_limit(depth_undist, DEPTH_MAX)
        depth_orig = add_label(depth_vis, "depth: original")
        depth_ud = add_label(depth_ud_vis, "depth: undistorted")
        depth_pair = stack_side_by_side(depth_orig, depth_ud)

        if SAVE_DIR:
            out_path = f"{SAVE_DIR.rstrip('/')}/depth_undistorted.png"
            cv2.imwrite(out_path, depth_pair)
            print(f"Saved depth comparison: {out_path}")

        if SHOW_WINDOWS:
            cv2.imshow("Depth undistort", depth_pair)

    if SHOW_WINDOWS:
        print("Press any key in an image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
