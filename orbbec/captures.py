"""Capture helpers for RGB-D images and point clouds."""

import json
import os

import cv2
import numpy as np

_CALIB_CACHE: dict | None = None
_CALIB_PATH: str | None = None  # cache keyed by the resolved JSON path


def _cap_get(capture_cfg: dict | None, key: str, default):
    """Small helper: allow missing capture_cfg without lots of None checks."""
    if not isinstance(capture_cfg, dict):
        return default
    return capture_cfg.get(key, default)


def ensure_capture_dir(capture_cfg: dict | None, subdir: str | None = None) -> str:
    """Create (if needed) and return the output directory for captures."""
    out_dir = os.path.join(os.getcwd(), _cap_get(capture_cfg, "dir", "captures"))
    if subdir:
        out_dir = os.path.join(out_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _capture_basename(capture_cfg: dict | None, index: int) -> str:
    prefix = _cap_get(capture_cfg, "prefix", "capture")
    return f"{prefix}_{index:06d}"


def next_capture_index(capture_cfg: dict | None) -> int:
    """Return the next capture index by scanning existing files."""
    base_dir = os.path.join(os.getcwd(), _cap_get(capture_cfg, "dir", "captures"))
    if not os.path.isdir(base_dir):
        return 1

    prefix = _cap_get(capture_cfg, "prefix", "capture")
    prefix_tag = f"{prefix}_"
    max_idx = 0

    for root, _dirs, files in os.walk(base_dir):
        for name in files:
            if not name.startswith(prefix_tag):
                continue
            rest = name[len(prefix_tag) :]
            idx_part = rest.split("_", 1)[0]
            if not idx_part.isdigit():
                continue
            idx = int(idx_part)
            if idx > max_idx:
                max_idx = idx

    return max_idx + 1


def save_rgbd_capture(
    depth_mm: np.ndarray | None,
    color_bgr: np.ndarray | None,
    depth_vis: np.ndarray | None,
    index: int,
    capture_cfg: dict | None,
) -> list[str]:
    """Save RGB (optional) + depth (optional) images. Returns saved paths."""
    out_dir = ensure_capture_dir(capture_cfg, _cap_get(capture_cfg, "rgbd_subdir", "rgbd"))
    base = _capture_basename(capture_cfg, index)
    saved: list[str] = []

    if color_bgr is not None:
        color_path = os.path.join(out_dir, f"{base}_color.png")
        cv2.imwrite(color_path, color_bgr)
        saved.append(color_path)

    if depth_mm is not None:
        depth_path = os.path.join(out_dir, f"{base}_depth.png")
        cv2.imwrite(depth_path, depth_mm.astype(np.uint16))
        saved.append(depth_path)

    if depth_vis is not None:
        depth_vis_path = os.path.join(out_dir, f"{base}_depth_vis.png")
        cv2.imwrite(depth_vis_path, depth_vis)
        saved.append(depth_vis_path)

    return saved


def save_pointcloud_capture(points_xyz: np.ndarray, index: int, capture_cfg: dict | None) -> str:
    """Save XYZ point cloud to an ASCII PLY file. Returns the path."""
    out_dir = ensure_capture_dir(
        capture_cfg, _cap_get(capture_cfg, "pointcloud_subdir", "pointcloud")
    )
    base = _capture_basename(capture_cfg, index)
    path = os.path.join(out_dir, f"{base}_points.ply")
    _save_point_cloud_to_ply(path, points_xyz)
    return path


def save_rgbd_undistorted(
    depth_mm: np.ndarray | None,
    color_bgr: np.ndarray | None,
    depth_vis_fn,
    index: int,
    capture_cfg: dict | None,
) -> list[str]:
    """Save undistorted RGB + depth (and depth visualization)."""
    if depth_mm is None and color_bgr is None:
        return []

    calib = _load_calibration(capture_cfg)
    if calib is None:
        print("Undistort: calibration JSON not found.")
        return []

    saved: list[str] = []
    out_dir = ensure_capture_dir(capture_cfg, _cap_get(capture_cfg, "rgbd_subdir", "rgbd"))
    base = _capture_basename(capture_cfg, index)

    if color_bgr is not None:
        color_ud = _undistort_color(color_bgr, calib)
        if color_ud is not None:
            color_path = os.path.join(out_dir, f"{base}_color_undist.png")
            cv2.imwrite(color_path, color_ud)
            saved.append(color_path)
        else:
            print("Undistort: no matching color intrinsics for this size.")

    if depth_mm is not None:
        depth_ud = _undistort_depth(depth_mm, calib)
        if depth_ud is not None:
            depth_path = os.path.join(out_dir, f"{base}_depth_undist.png")
            cv2.imwrite(depth_path, depth_ud.astype(np.uint16))
            saved.append(depth_path)

            if depth_vis_fn is not None:
                depth_vis = depth_vis_fn(depth_ud)
                depth_vis_path = os.path.join(out_dir, f"{base}_depth_vis_undist.png")
                cv2.imwrite(depth_vis_path, depth_vis)
                saved.append(depth_vis_path)
        else:
            print("Undistort: no matching depth intrinsics for this size.")

    return saved


def save_pointcloud_undistorted(
    depth_mm: np.ndarray, index: int, capture_cfg: dict | None
) -> str | None:
    """Save a point cloud derived from undistorted depth."""
    calib = _load_calibration(capture_cfg)
    if calib is None:
        print("Undistort: calibration JSON not found.")
        return None

    depth_ud = _undistort_depth(depth_mm, calib)
    if depth_ud is None:
        print("Undistort: no matching depth intrinsics for this size.")
        return None

    K = _get_intrinsic_matrix(calib, "depth", depth_mm.shape[1], depth_mm.shape[0])
    if K is None:
        print("Undistort: missing depth intrinsics.")
        return None

    xyz = _pointcloud_from_depth(depth_ud, K)
    out_dir = ensure_capture_dir(
        capture_cfg, _cap_get(capture_cfg, "pointcloud_subdir", "pointcloud")
    )
    base = _capture_basename(capture_cfg, index)
    path = os.path.join(out_dir, f"{base}_points_undist.ply")
    _save_point_cloud_to_ply(path, xyz)
    return path


def _save_point_cloud_to_ply(path: str, points_xyz: np.ndarray) -> None:
    """Write a minimal ASCII PLY with XYZ points."""
    n = points_xyz.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for (x, y, z) in points_xyz:
            f.write(f"{x} {y} {z}\n")


def _resolve_calib_path(capture_cfg: dict | None) -> str:
    """Resolve calibration path relative to CWD if not absolute."""
    path = _cap_get(capture_cfg, "calib_json", "camera_intrinsics.json")
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _load_calibration(capture_cfg: dict | None) -> dict | None:
    """Load and cache intrinsics/distortion JSON."""
    global _CALIB_CACHE, _CALIB_PATH
    path = _resolve_calib_path(capture_cfg)
    if _CALIB_CACHE is not None and _CALIB_PATH == path:
        return _CALIB_CACHE
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        _CALIB_CACHE = json.load(f)
        _CALIB_PATH = path
    return _CALIB_CACHE


def _pick_cfg_by_size(calib: dict, kind: str, width: int, height: int) -> dict | None:
    """Pick intrinsics by WxH key if available."""
    key = f"{int(width)}x{int(height)}"
    by_size = calib.get(f"{kind}_by_size", {})
    if isinstance(by_size, dict) and key in by_size:
        return by_size[key]
    return None


def _get_intrinsic_matrix(calib: dict, kind: str, width: int, height: int) -> np.ndarray | None:
    """Return K for the given stream size, or None if missing."""
    cfg = _pick_cfg_by_size(calib, kind, width, height)
    if cfg is None:
        return None
    intr = cfg.get("intrinsic", cfg)
    return build_camera_matrix(
        float(intr.get("fx", 0.0)),
        float(intr.get("fy", 0.0)),
        float(intr.get("cx", 0.0)),
        float(intr.get("cy", 0.0)),
    )


def _get_distortion(calib: dict, kind: str, width: int, height: int) -> np.ndarray | None:
    """Return distortion vector for the given stream size, or None if missing."""
    cfg = _pick_cfg_by_size(calib, kind, width, height)
    if cfg is None:
        return None
    dist = cfg.get("distortion", cfg)
    return build_distortion(dist)


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


def _undistort_color(color_bgr: np.ndarray, calib: dict) -> np.ndarray | None:
    """Undistort color frame using matching intrinsics."""
    h, w = color_bgr.shape[:2]
    K = _get_intrinsic_matrix(calib, "color", w, h)
    D = _get_distortion(calib, "color", w, h)
    if K is None or D is None:
        return None
    return cv2.undistort(color_bgr, K, D, None, K)


def _undistort_depth(depth_mm: np.ndarray, calib: dict) -> np.ndarray | None:
    """Undistort depth (in mm) using matching intrinsics."""
    h, w = depth_mm.shape[:2]
    K = _get_intrinsic_matrix(calib, "depth", w, h)
    D = _get_distortion(calib, "depth", w, h)
    if K is None or D is None:
        return None
    depth_f = depth_mm.astype(np.float32)
    depth_ud = cv2.undistort(depth_f, K, D, None, K)
    depth_ud = np.clip(depth_ud, 0, 65535).astype(np.uint16)
    return depth_ud


def _pointcloud_from_depth(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project depth to XYZ using the pinhole model."""
    h, w = depth_mm.shape[:2]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    z = depth_mm.astype(np.float32)
    mask = z > 0
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32)

    v, u = np.indices((h, w), dtype=np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    xyz = np.stack([x, y, z], axis=-1)
    return xyz[mask]
