"""One-shot capture helper for RGB-D + point cloud saves."""

import numpy as np
from pyorbbecsdk import OBFormat

from orbbec.captures import (
    next_capture_index,
    save_pointcloud_capture,
    save_pointcloud_undistorted,
    save_rgbd_capture,
    save_rgbd_undistorted,
)
from orbbec.pointcloud import decode_point_cloud_frame_to_numpy
from orbbec.session import CaptureSession
from orbbec.visual import (
    clamp_depth_mm,
    color_frame_to_bgr,
    depth_frame_to_mm,
    depth_mm_to_vis,
    extract_color_frame,
)


def take_images(config_path: str | None = None, capture_index: int | None = None) -> dict:
    """Capture RGB-D + point cloud (and undistorted versions) and save to disk."""
    session = CaptureSession(config_path=config_path)
    try:
        session.warmup()
        frameset, aligned_frames, depth_frame = session.next_frames()
        return save_all_from_frames(
            depth_frame=depth_frame,
            aligned_frames=aligned_frames,
            frameset=frameset,
            have_color=session.have_color,
            point_cloud_filter=session.point_cloud_filter,
            display_cfg=session.display_cfg,
            capture_cfg=session.capture_cfg,
            capture_index=capture_index,
        )
    finally:
        session.close()


def save_all_from_frames(
    *,
    depth_frame,
    aligned_frames,
    frameset,
    have_color: bool,
    point_cloud_filter,
    display_cfg: dict,
    capture_cfg: dict,
    capture_index: int | None,
) -> dict | None:
    """Save RGB-D + point cloud outputs for an already-captured depth frame."""
    if capture_index is None:
        capture_index = next_capture_index(capture_cfg)
    depth_mm = depth_frame_to_mm(depth_frame)
    if depth_mm is None:
        return None

    depth_mm = clamp_depth_mm(depth_mm, display_cfg)

    # Use aligned color when available, otherwise fall back to raw frameset.
    color_bgr = None
    if have_color:
        color_frame = extract_color_frame(aligned_frames)
        if color_frame is None:
            color_frame = extract_color_frame(frameset)
        if color_frame is not None:
            color_bgr = color_frame_to_bgr(color_frame)

    # Save RGB + depth + a quick visualization.
    depth_vis = depth_mm_to_vis(depth_mm, display_cfg)
    saved_rgbd = save_rgbd_capture(depth_mm, color_bgr, depth_vis, capture_index, capture_cfg)
    saved_undist = save_rgbd_undistorted(
        depth_mm,
        color_bgr,
        lambda d: depth_mm_to_vis(d, display_cfg),
        capture_index,
        capture_cfg,
    )

    # Point cloud from filtered depth (XYZ only).
    point_format = OBFormat.POINT
    point_cloud_filter.set_create_point_format(point_format)
    pc_frame = point_cloud_filter.process(depth_frame)
    saved_pc = None
    if pc_frame is not None:
        xyz, _ = decode_point_cloud_frame_to_numpy(pc_frame, point_format)
        if xyz is not None:
            saved_pc = save_pointcloud_capture(xyz, capture_index, capture_cfg)

    saved_pc_undist = save_pointcloud_undistorted(depth_mm, capture_index, capture_cfg)

    return {
        "rgbd": saved_rgbd,
        "rgbd_undist": saved_undist,
        "pointcloud": saved_pc,
        "pointcloud_undist": saved_pc_undist,
    }
