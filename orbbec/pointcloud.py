"""Point cloud helpers."""

import numpy as np

from pyorbbecsdk import OBFormat, PointCloudFilter, Pipeline


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
    try:
        cam_param = pipeline.get_camera_param()
        pc.set_camera_param(cam_param)
    except Exception:
        pass
    return pc
