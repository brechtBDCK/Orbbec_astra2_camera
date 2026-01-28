"""Point cloud helpers.

These functions generate and optionally save point clouds.
"""

import os
import time
from typing import Tuple

import numpy as np

from pyorbbecsdk import OBFormat, PointCloudFilter, Pipeline

import main_run.config as cfg


def ensure_pointcloud_dir() -> str:
    """Ensure the output directory exists."""
    out_dir = os.path.join(os.getcwd(), cfg.POINTCLOUD_SAVE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def pointcloud_output_path(frame_index: int) -> str:
    """Create a timestamped output path for a PLY file."""
    out_dir = ensure_pointcloud_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{cfg.POINTCLOUD_SAVE_PREFIX}_{ts}_{frame_index:06d}.ply")


def save_point_cloud_to_ply(path: str, points_xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """Save XYZ (and optional RGB) to an ASCII PLY file."""
    n = points_xyz.shape[0]
    has_color = rgb is not None

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if has_color:
            for (x, y, z), (r, g, b) in zip(points_xyz, rgb, strict=False):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z) in points_xyz:
                f.write(f"{x} {y} {z}\n")


def decode_point_cloud_frame_to_numpy(point_cloud_frame, point_format: OBFormat):
    """Decode an SDK point cloud frame into numpy arrays."""
    data = point_cloud_frame.get_data()
    if data is None:
        return None, None

    buf = np.frombuffer(data, dtype=np.float32)

    if point_format == OBFormat.POINT:
        if buf.size % 3 != 0:
            raise ValueError(f"Unexpected POINT buffer length: {buf.size}")
        xyz = buf.reshape(-1, 3)
        return xyz, None

    if point_format == OBFormat.RGB_POINT:
        if buf.size % 6 != 0:
            raise ValueError(f"Unexpected RGB_POINT buffer length: {buf.size}")
        pts = buf.reshape(-1, 6)
        xyz = pts[:, 0:3]
        rgb_f = pts[:, 3:6]
        rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)
        return xyz, rgb

    raise ValueError(f"Unsupported point format: {point_format}")


def create_pointcloud_filter(pipeline: Pipeline) -> PointCloudFilter:
    """Create and configure the SDK point cloud filter."""
    pc = PointCloudFilter()
    pc.set_color_data_normalization(bool(cfg.POINTCLOUD_COLOR_NORMALIZE))
    try:
        cam_param = pipeline.get_camera_param()
        pc.set_camera_param(cam_param)
    except Exception:
        pass
    return pc


def summarize_pointcloud(xyz: np.ndarray) -> None:
    """Print simple statistics about the point cloud."""
    if xyz.size == 0:
        print("Point cloud is empty.")
        return
    z = xyz[:, 2]
    finite_z = z[np.isfinite(z)]
    z_min = float(np.min(finite_z)) if finite_z.size else float("nan")
    z_med = float(np.median(finite_z)) if finite_z.size else float("nan")
    z_max = float(np.max(finite_z)) if finite_z.size else float("nan")
    print("\nPoint cloud summary:")
    print(f"  points: {xyz.shape[0]}")
    print(f"  z min:  {z_min:.3f}")
    print(f"  z med:  {z_med:.3f}")
    print(f"  z max:  {z_max:.3f}")
