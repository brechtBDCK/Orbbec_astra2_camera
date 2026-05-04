import cv2
import numpy as np
import pyorbbecsdk as sdk

FormatConvertFilter = sdk.FormatConvertFilter
OBConvertFormat = sdk.OBConvertFormat
OBFormat = sdk.OBFormat

def color_frame_to_bgr(color_frame):
    if color_frame is None:
        return None

    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if fmt == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))

    convert_map = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
        OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
    }
    convert_format = convert_map.get(fmt)
    if convert_format is None:
        return None

    converter = FormatConvertFilter()
    converter.set_format_convert_format(convert_format)
    rgb_frame = converter.process(color_frame)
    if rgb_frame is None:
        return None

    rgb = np.resize(np.asanyarray(rgb_frame.get_data()), (height, width, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def depth_frame_to_bgr(depth_frame):
    if depth_frame is None or depth_frame.get_format() != OBFormat.Y16:
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    depth_scale = float(depth_frame.get_depth_scale())
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_mm = depth_raw.astype(np.float32) * depth_scale

    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        depth_8u = np.zeros((height, width), dtype=np.uint8)
    else:
        min_valid = float(valid.min())
        max_valid = float(valid.max())
        if max_valid <= min_valid:
            max_valid = min_valid + 1.0
        depth_8u = (
            (np.clip(depth_mm, min_valid, max_valid) - min_valid)
            / (max_valid - min_valid)
            * 255.0
        ).astype(np.uint8)

    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
