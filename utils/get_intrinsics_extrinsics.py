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

# If True, print extrinsics for ALL depth profiles to the chosen/default color.
PRINT_EXTRINSICS_FOR_ALL_DEPTH = False

# Save all intrinsics/extrinsics to JSON.
SAVE_JSON = True
OUTPUT_JSON_NAME = "camera_intrinsics.json"
SAVE_EXTRINSICS_ALL_PAIRS = True


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
    return {
        "fx": float(_safe_get(intr, "fx")),
        "fy": float(_safe_get(intr, "fy")),
        "cx": float(_safe_get(intr, "cx")),
        "cy": float(_safe_get(intr, "cy")),
        "width": int(_safe_get(intr, "width")),
        "height": int(_safe_get(intr, "height")),
    }


def distortion_to_dict(dist) -> dict:
    return {
        "k1": float(_safe_get(dist, "k1")),
        "k2": float(_safe_get(dist, "k2")),
        "k3": float(_safe_get(dist, "k3")),
        "k4": float(_safe_get(dist, "k4")),
        "k5": float(_safe_get(dist, "k5")),
        "k6": float(_safe_get(dist, "k6")),
        "p1": float(_safe_get(dist, "p1")),
        "p2": float(_safe_get(dist, "p2")),
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


def _find_default_index(profile_list, default_profile) -> int | None:
    if default_profile is None:
        return None
    count = profile_list.get_count()
    for i in range(count):
        sp = profile_list.get_stream_profile_by_index(i)
        if sp is None or not sp.is_video_stream_profile():
            continue
        vsp = sp.as_video_stream_profile()
        if (
            vsp.get_width() == default_profile.get_width()
            and vsp.get_height() == default_profile.get_height()
            and vsp.get_fps() == default_profile.get_fps()
            and sp.get_format() == default_profile.get_format()
        ):
            return i
    return None


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

        if PRINT_EXTRINSICS_FOR_ALL_DEPTH:
            print("\nExtrinsics (depth -> selected color):")
            for i in range(depth_list.get_count()):
                sp = depth_list.get_stream_profile_by_index(i)
                if sp is None or not sp.is_video_stream_profile():
                    continue
                depth_profile = sp.as_video_stream_profile()
                extr = depth_profile.get_extrinsic_to(color_profile)
                print(f"  depth[{i:02d}] -> color[{EXTRINSIC_COLOR_INDEX or 'default'}]: {format_extrinsic(extr)}")
        else:
            depth_profile = get_profile_by_index(depth_list, EXTRINSIC_DEPTH_INDEX)
            extr = depth_profile.get_extrinsic_to(color_profile)
            print("\nExtrinsics (depth -> color):")
            print(f"  depth[{EXTRINSIC_DEPTH_INDEX or 'default'}] -> color[{EXTRINSIC_COLOR_INDEX or 'default'}]: {format_extrinsic(extr)}")

    if SAVE_JSON:
        depth_by_size: dict[str, dict] = {}
        depth_profile_by_size: dict[str, object] = {}
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
                depth_profile_by_size[key] = vsp

        color_by_size: dict[str, dict] = {}
        color_profile_by_size: dict[str, object] = {}
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
                color_profile_by_size[key] = vsp

        data = {
            "depth_by_size": depth_by_size,
            "color_by_size": color_by_size,
            "extrinsics": {"depth_to_color": []},
        }

        extr_list = []
        if SAVE_EXTRINSICS_ALL_PAIRS:
            for d_key, d_vsp in depth_profile_by_size.items():
                for c_key, c_vsp in color_profile_by_size.items():
                    extr = d_vsp.get_extrinsic_to(c_vsp)
                    extr_list.append(
                        {
                            "depth_size": d_key,
                            "color_size": c_key,
                            **extrinsic_to_dict(extr),
                        }
                    )
        else:
            color_profile = get_profile_by_index(color_list, EXTRINSIC_COLOR_INDEX)
            depth_profile = get_profile_by_index(depth_list, EXTRINSIC_DEPTH_INDEX)
            extr = depth_profile.get_extrinsic_to(color_profile)
            d_key = size_key(depth_profile.get_width(), depth_profile.get_height())
            c_key = size_key(color_profile.get_width(), color_profile.get_height())
            extr_list.append(
                {
                    "depth_size": d_key,
                    "color_size": c_key,
                    **extrinsic_to_dict(extr),
                }
            )

        data["extrinsics"]["depth_to_color"] = extr_list

        out_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON_NAME)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved profiles JSON: {out_path}")


if __name__ == "__main__":
    main()
