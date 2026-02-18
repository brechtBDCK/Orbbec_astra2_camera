from __future__ import annotations

import json
import os

import numpy as np

from pyorbbecsdk import OBSensorType, Pipeline

# -----------------------------------------------------------------------------
# Options (edit these)
# -----------------------------------------------------------------------------
PRINT_ALL_PROFILES = True
PRINT_EXTRINSICS = True

# If set, choose a specific profile index; otherwise use default.
EXTRINSIC_COLOR_INDEX = None
EXTRINSIC_DEPTH_INDEX = None

# Save intrinsics + a single depth->color extrinsic to JSON.
SAVE_JSON = True
OUTPUT_JSON_NAME = "/home/bdck/PROJECTS_WSL/Orbbec_astra2_camera/camera_intrinsics.json"


def _safe_get(obj, name: str):
    return getattr(obj, name) if hasattr(obj, name) else None


def format_intrinsic(intr) -> str:
    keys = ["fx", "fy", "cx", "cy", "width", "height"]
    data = {k: _safe_get(intr, k) for k in keys if _safe_get(intr, k) is not None}
    return str(data) if data else str(intr)


def format_distortion(dist) -> str:
    keys = ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"]
    data = {k: _safe_get(dist, k) for k in keys if _safe_get(dist, k) is not None}
    return str(data) if data else str(dist)


def format_extrinsic(extr) -> str:
    rot = _safe_get(extr, "rotation") or _safe_get(extr, "rot")
    trans = _safe_get(extr, "translation") or _safe_get(extr, "transform")
    if rot is not None or trans is not None:
        return str({"rotation": rot, "translation": trans})
    return str(extr)


def intrinsic_to_dict(intr) -> dict:
    fx = _safe_get(intr, "fx")
    fy = _safe_get(intr, "fy")
    cx = _safe_get(intr, "cx")
    cy = _safe_get(intr, "cy")
    width = _safe_get(intr, "width")
    height = _safe_get(intr, "height")
    return {
        "fx": float(fx) if fx is not None else None,
        "fy": float(fy) if fy is not None else None,
        "cx": float(cx) if cx is not None else None,
        "cy": float(cy) if cy is not None else None,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
    }


def distortion_to_dict(dist) -> dict:
    k1 = _safe_get(dist, "k1")
    k2 = _safe_get(dist, "k2")
    k3 = _safe_get(dist, "k3")
    k4 = _safe_get(dist, "k4")
    k5 = _safe_get(dist, "k5")
    k6 = _safe_get(dist, "k6")
    
    p1 = _safe_get(dist, "p1")
    p2 = _safe_get(dist, "p2")
    return {
        "k1": float(k1) if k1 is not None else None,
        "k2": float(k2) if k2 is not None else None,
        "k3": float(k3) if k3 is not None else None,
        "k4": float(k4) if k4 is not None else None,
        "k5": float(k5) if k5 is not None else None,
        "k6": float(k6) if k6 is not None else None,
        "p1": float(p1) if p1 is not None else None,
        "p2": float(p2) if p2 is not None else None,
    }


def extrinsic_to_dict(extr) -> dict:
    rot = _safe_get(extr, "rotation") or _safe_get(extr, "rot")
    trans = _safe_get(extr, "translation") or _safe_get(extr, "transform")
    rot_arr = np.asarray(rot, dtype=np.float32).reshape(-1).tolist() if rot is not None else []
    trans_arr = np.asarray(trans, dtype=np.float32).reshape(-1).tolist() if trans is not None else []
    return {"rotation": rot_arr, "translation": trans_arr}


def list_profiles(pipeline: Pipeline, sensor_type: OBSensorType, label: str) -> list:
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    count = profile_list.get_count()
    profiles = []
    print(f"\n{label} profiles:")
    for i in range(count):
        sp = profile_list.get_stream_profile_by_index(i)
        if sp is None or not sp.is_video_stream_profile():
            continue
        vsp = sp.as_video_stream_profile()
        width = vsp.get_width()
        height = vsp.get_height()
        fps = vsp.get_fps()
        fmt = sp.get_format()
        fmt_name = getattr(fmt, "name", str(fmt))
        profiles.append(vsp)
        print(f"  [{i:02d}] {width}x{height}@{fps} {fmt_name}")
    return profiles


def get_profile_by_index(profile_list, index: int | None):
    if index is None:
        return profile_list.get_default_video_stream_profile()
    sp = profile_list.get_stream_profile_by_index(int(index))
    if sp is None or not sp.is_video_stream_profile():
        raise RuntimeError(f"Profile index {index} is not a video profile.")
    return sp.as_video_stream_profile()


def profile_to_dict(sp, vsp, index: int) -> dict:
    fmt = sp.get_format()
    fmt_name = getattr(fmt, "name", str(fmt))
    return {
        "index": index,
        "width": int(vsp.get_width()),
        "height": int(vsp.get_height()),
        "fps": int(vsp.get_fps()),
        "format": fmt_name,
        "intrinsic": intrinsic_to_dict(vsp.get_intrinsic()),
        "distortion": distortion_to_dict(vsp.get_distortion()),
    }


def size_key(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def main() -> None:
    pipeline = Pipeline()

    color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    if PRINT_ALL_PROFILES:
        list_profiles(pipeline, OBSensorType.DEPTH_SENSOR, "Depth")
        list_profiles(pipeline, OBSensorType.COLOR_SENSOR, "Color")

    # Intrinsics + distortion per profile
    if PRINT_ALL_PROFILES:
        print("\nDepth intrinsics/distortion:")
        for i in range(depth_list.get_count()):
            sp = depth_list.get_stream_profile_by_index(i)
            if sp is None or not sp.is_video_stream_profile():
                continue
            vsp = sp.as_video_stream_profile()
            fmt = sp.get_format()
            fmt_name = getattr(fmt, "name", str(fmt))
            print(f"  [{i:02d}] {vsp.get_width()}x{vsp.get_height()}@{vsp.get_fps()} {fmt_name}")
            print(f"       intr: {format_intrinsic(vsp.get_intrinsic())}")
            print(f"       dist: {format_distortion(vsp.get_distortion())}")

        print("\nColor intrinsics/distortion:")
        for i in range(color_list.get_count()):
            sp = color_list.get_stream_profile_by_index(i)
            if sp is None or not sp.is_video_stream_profile():
                continue
            vsp = sp.as_video_stream_profile()
            fmt = sp.get_format()
            fmt_name = getattr(fmt, "name", str(fmt))
            print(f"  [{i:02d}] {vsp.get_width()}x{vsp.get_height()}@{vsp.get_fps()} {fmt_name}")
            print(f"       intr: {format_intrinsic(vsp.get_intrinsic())}")
            print(f"       dist: {format_distortion(vsp.get_distortion())}")

    if PRINT_EXTRINSICS:
        color_profile = get_profile_by_index(color_list, EXTRINSIC_COLOR_INDEX)
        depth_profile = get_profile_by_index(depth_list, EXTRINSIC_DEPTH_INDEX)
        extr = depth_profile.get_extrinsic_to(color_profile)
        print("\nExtrinsics (depth -> color):")
        print(
            f"  depth[{EXTRINSIC_DEPTH_INDEX or 'default'}] -> "
            f"color[{EXTRINSIC_COLOR_INDEX or 'default'}]: {format_extrinsic(extr)}"
        )

    if SAVE_JSON:
        depth_by_size: dict[str, dict] = {}
        for i in range(depth_list.get_count()):
            sp = depth_list.get_stream_profile_by_index(i)
            if sp is None or not sp.is_video_stream_profile():
                continue
            vsp = sp.as_video_stream_profile()
            key = size_key(vsp.get_width(), vsp.get_height())
            if key not in depth_by_size:
                depth_by_size[key] = {
                    "width": int(vsp.get_width()),
                    "height": int(vsp.get_height()),
                    "intrinsic": intrinsic_to_dict(vsp.get_intrinsic()),
                    "distortion": distortion_to_dict(vsp.get_distortion()),
                }

        color_by_size: dict[str, dict] = {}
        for i in range(color_list.get_count()):
            sp = color_list.get_stream_profile_by_index(i)
            if sp is None or not sp.is_video_stream_profile():
                continue
            vsp = sp.as_video_stream_profile()
            key = size_key(vsp.get_width(), vsp.get_height())
            if key not in color_by_size:
                color_by_size[key] = {
                    "width": int(vsp.get_width()),
                    "height": int(vsp.get_height()),
                    "intrinsic": intrinsic_to_dict(vsp.get_intrinsic()),
                    "distortion": distortion_to_dict(vsp.get_distortion()),
                }

        color_profile = get_profile_by_index(color_list, EXTRINSIC_COLOR_INDEX)
        depth_profile = get_profile_by_index(depth_list, EXTRINSIC_DEPTH_INDEX)
        extr = depth_profile.get_extrinsic_to(color_profile)

        data = {
            "depth_by_size": depth_by_size,
            "color_by_size": color_by_size,
            "extrinsics": {"depth_to_color": extrinsic_to_dict(extr)},
        }

        out_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON_NAME)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved profiles JSON: {out_path}")


if __name__ == "__main__":
    main()
