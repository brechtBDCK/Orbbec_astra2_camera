#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, OBError,
    OBSensorType, OBFormat, OBStreamType,
    AlignFilter, PointCloudFilter
)

def save_point_cloud_to_ply(path: str, points_xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """
    Save point cloud to an ASCII PLY.
    - points_xyz: (N,3) float32 [x,y,z]
    - rgb:        (N,3) uint8   [r,g,b] optional
    """
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
    """
    Orbbec point cloud formats map to SDK structs:
      - OBPoint:      float x,y,z
      - OBColorPoint: float x,y,z,r,g,b
    We'll decode from the frame's raw byte buffer as float32. :contentReference[oaicite:1]{index=1}
    """
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
        # In the SDK, r/g/b are floats in the OBColorPoint struct. :contentReference[oaicite:2]{index=2}
        # Most devices produce 0..255-like values; clamp & convert to uint8.
        rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)
        return xyz, rgb

    raise ValueError(f"Unsupported point format: {point_format}")

def pick_default_profile(pipeline: Pipeline, sensor_type: OBSensorType):
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    if profile_list is None:
        return None
    # safest: just ask the device for its default profile
    return profile_list.get_default_video_stream_profile()

def main():
    out_dir = os.path.join(os.getcwd(), "point_clouds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "point_cloud.ply")

    pipeline = Pipeline()
    config = Config()

    # --- Depth stream (required)
    depth_profile = pick_default_profile(pipeline, OBSensorType.DEPTH_SENSOR)
    if depth_profile is None:
        print("No depth profile available (is the camera visible to this OS/WSL instance?)")
        return 1
    config.enable_stream(depth_profile)

    # --- Color stream (optional)
    has_color = False
    try:
        color_profile = pick_default_profile(pipeline, OBSensorType.COLOR_SENSOR)
        if color_profile is not None:
            config.enable_stream(color_profile)
            has_color = True
    except OBError as e:
        print(f"Color stream not available: {e}")
        has_color = False

    # (Optional) sync frames (recommended when using both depth & color)
    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass

    # Start pipeline WITH config (important)
    pipeline.start(config)

    try:
        # Point cloud pipeline
        point_cloud_filter = PointCloudFilter()

        # If you have camera parameters available, set them (deprecated but harmless)
        try:
            cam_param = pipeline.get_camera_param()
            point_cloud_filter.set_camera_param(cam_param)
        except Exception:
            pass

        # If we have color, align depth->color before generating RGB point cloud
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) if has_color else None

        # Grab a single good frameset (depth required, color optional)
        deadline = time.time() + 5.0
        frames = None
        depth_frame = None
        color_frame = None
        while time.time() < deadline:
            candidate = pipeline.wait_for_frames(1000)  # ms
            if candidate is None:
                continue
            depth = candidate.get_depth_frame()
            if depth is None:
                continue
            frames = candidate
            depth_frame = depth
            if has_color:
                color = candidate.get_color_frame()
                if color is None:
                    continue
                color_frame = color
            break

        if depth_frame is None:
            print("Timed out waiting for depth frames. (In WSL, verify the USB device is actually attached to WSL.)")
            return 2

        if has_color and color_frame is None:
            print("Color frames not available; falling back to depth-only point cloud.")
            has_color = False
            align_filter = None

        point_format = OBFormat.RGB_POINT if has_color else OBFormat.POINT
        point_cloud_filter.set_create_point_format(point_format)

        # Optionally align
        if align_filter is not None:
            aligned = align_filter.process(frames)
            if aligned is None:
                print("Frame alignment failed.")
                return 3
            pc_input = aligned
        else:
            pc_input = depth_frame

        # Generate point cloud frame
        pc_frame = point_cloud_filter.process(pc_input)
        if pc_frame is None:
            print("Point cloud generation failed (pc_frame is None).")
            return 4

        # Decode and save
        xyz, rgb = decode_point_cloud_frame_to_numpy(pc_frame, point_format)
        if xyz is None:
            print("Point cloud frame had no data.")
            return 5

        save_point_cloud_to_ply(out_path, xyz, rgb)
        print(f"Saved: {out_path}")
        return 0
    finally:
        pipeline.stop()

if __name__ == "__main__":
    sys.exit(main())
