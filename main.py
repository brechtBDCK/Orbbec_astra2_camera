"""Astra2 main entry point.

This file focuses on control-flow only. Configuration lives in config.toml.
"""

import os
import sys
import time

import cv2
import numpy as np
from pyorbbecsdk import AlignFilter, Config, OBError, OBFormat, OBSensorType, OBStreamType, Pipeline

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib
from orbbec.captures import (
    save_pointcloud_capture,
    save_pointcloud_undistorted,
    save_rgbd_capture,
    save_rgbd_undistorted,
)
from orbbec.filters import apply_depth_filters, build_depth_filters
from orbbec.pointcloud import create_pointcloud_filter, decode_point_cloud_frame_to_numpy
from orbbec.profiles import choose_video_profile, list_video_profiles
from orbbec.runtime_helpers import (
    depth_mm_to_vis,
    extract_color_frame,
    extract_depth_frame,
    print_device_info,
    stack_side_by_side,
)
from orbbec.visual import color_frame_to_bgr, depth_frame_to_mm


def load_config(config_path: str | None = None) -> dict:
    default_path = os.path.join(os.path.dirname(__file__), "config.toml")
    path = config_path or os.environ.get("ASTRA2_CONFIG_PATH", default_path)
    cfg: dict = {}
    if not os.path.exists(path):
        print(f"Config file not found: {path} (using defaults)")
    else:
        try:
            with open(path, "rb") as f:
                cfg = tomllib.load(f)
        except Exception as exc:
            print(f"Config load failed: {exc} (using defaults)")
    return cfg


def main() -> int:
    """Program entry. Returns a process exit code."""
    cfg = load_config("config.toml")
    run_cfg = cfg.get("run", {})
    streams_cfg = cfg.get("streams", {})
    filters_cfg = cfg.get("filters", {})
    display_cfg = cfg.get("display", {})
    capture_cfg = cfg.get("capture", {})

    pipeline = Pipeline()
    live = not run_cfg.get("print_info_offline", False)

    if run_cfg.get("print_info_offline", False):
        # Print everything and exit (no live run).
        print_device_info(pipeline)
        list_video_profiles(pipeline, OBSensorType.DEPTH_SENSOR, "Depth")
        list_video_profiles(pipeline, OBSensorType.COLOR_SENSOR, "Color")
        list_video_profiles(pipeline, OBSensorType.LEFT_IR_SENSOR, "Left IR")
        list_video_profiles(pipeline, OBSensorType.RIGHT_IR_SENSOR, "Right IR")
        try:
            depth_sensor = pipeline.get_device().get_sensor(OBSensorType.DEPTH_SENSOR)
            rec = list(depth_sensor.get_recommended_filters())
            print("\nAvailable (recommended) filters:")
            for f in rec:
                try:
                    print(f"  - {f.get_name()} (enabled={f.is_enabled()})")
                except Exception:
                    print(f"  - {f}")
        except Exception as exc:
            print(f"\nAvailable filters: unable to query ({exc})")
        return 0

    config = Config()

    # Depth is required.
    try:
        # If DEPTH_PROFILE_INDEX is set, it already defines width/height/fps.
        # Only set width/height/fps when you want to request a specific profile.
        depth_cfg = streams_cfg.get("depth", {})
        depth_profile_index = depth_cfg.get("profile_index")
        if depth_profile_index is not None and depth_profile_index < 0:
            depth_profile_index = None
        depth_width = depth_cfg.get("width", 0) or None
        depth_height = depth_cfg.get("height", 0) or None
        depth_fps = depth_cfg.get("fps", 0) or None
        depth_profile = choose_video_profile(
            pipeline,
            OBSensorType.DEPTH_SENSOR,
            index=depth_profile_index,
            width=depth_width,
            height=depth_height,
            fps=depth_fps,
            format_name=depth_cfg.get("format", "Y16"),
            default_format=OBFormat.Y16,
        )
        config.enable_stream(depth_profile)
        print(f"Depth profile: {depth_profile}")
    except Exception as exc:
        print(f"Failed to configure depth stream: {exc}")
        return 1

    # Color is optional. If unavailable, keep running with depth only.
    have_color = False
    try:
        color_cfg = streams_cfg.get("color", {})
        color_profile_index = color_cfg.get("profile_index")
        if color_profile_index is not None and color_profile_index < 0:
            color_profile_index = None
        color_width = color_cfg.get("width", 0) or None
        color_height = color_cfg.get("height", 0) or None
        color_fps = color_cfg.get("fps", 0) or None
        color_profile = choose_video_profile(
            pipeline,
            OBSensorType.COLOR_SENSOR,
            index=color_profile_index,
            width=color_width,
            height=color_height,
            fps=color_fps,
            format_name=color_cfg.get("format", "RGB"),
            default_format=OBFormat.RGB,
        )
        config.enable_stream(color_profile)
        have_color = True
        print(f"Color profile: {color_profile}")
    except OBError as exc:
        print(f"Color stream unavailable: {exc}")
    except Exception as exc:
        print(f"Failed to configure color stream: {exc}")

    if have_color:
        # Frame sync improves depth/color alignment stability.
        try:
            pipeline.enable_frame_sync()
        except Exception:
            pass

    try:
        pipeline.start(config)
    except Exception as exc:
        print(f"Failed to start pipeline: {exc}")
        return 1

    capture_index = 0
    try:
        dec_cfg = filters_cfg.get("decimation", {})
        if dec_cfg.get("scale_value") == 0:
            dec_cfg["scale_value"] = None
        depth_filters = build_depth_filters(pipeline, filters_cfg)
        if depth_filters.description:
            print("\nDepth filters (applied to depth + point clouds):")
            for line in depth_filters.description:
                print(f"  - {line}")
        else:
            print("\nDepth filters: none")

        # Depth->color alignment is on when color is available.
        align_filter = AlignFilter(OBStreamType.COLOR_STREAM) if have_color else None
        if not have_color:
            print("Color stream unavailable; running depth-only preview.")

        # PointCloudFilter needs camera params to correctly scale output points.
        point_cloud_filter = create_pointcloud_filter(pipeline)

        if display_cfg.get("show_window", True):
            window_name = display_cfg.get("window_name", "Astra2")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(
                window_name,
                int(display_cfg.get("window_pos_x", 50)),
                int(display_cfg.get("window_pos_y", 50)),
            )

        last_print = 0.0
        warmup_total = max(0, int(display_cfg.get("warmup_frames", 50)))
        warmup_remaining = warmup_total
        warmup_seen = 0
        depth_vis_fn = lambda d: depth_mm_to_vis(d, display_cfg)
        while True:
            frameset = pipeline.wait_for_frames(int(display_cfg.get("timeout_ms", 100)))
            if frameset is None:
                continue

            aligned_frames = frameset
            if align_filter is not None:
                aligned = align_filter.process(frameset)
                if aligned is not None:
                    aligned_frames = aligned

            depth_frame = extract_depth_frame(aligned_frames)
            if depth_frame is None:
                depth_frame = extract_depth_frame(frameset)
            if depth_frame is None:
                continue
            depth_frame = apply_depth_filters(depth_frame, depth_filters.filters)

            if warmup_remaining > 0:
                warmup_seen += 1
                print(f"Warmup frame {warmup_seen}/{warmup_total}")
                warmup_remaining -= 1
                continue

            depth_mm = depth_frame_to_mm(depth_frame)
            if depth_mm is None:
                continue

            depth_mm = np.where(
                (depth_mm > display_cfg.get("min_depth_mm", 20))
                & (depth_mm < display_cfg.get("max_depth_mm", 10000)),
                depth_mm,
                0,
            ).astype(np.uint16)

            color_bgr = None
            if have_color:
                color_frame = extract_color_frame(aligned_frames)
                if color_frame is None:
                    color_frame = extract_color_frame(frameset)
                if color_frame is not None:
                    color_bgr = color_frame_to_bgr(color_frame)

            if live and display_cfg.get("show_window", True):
                depth_vis = depth_mm_to_vis(depth_mm, display_cfg)
                combined = stack_side_by_side(depth_vis, color_bgr) if color_bgr is not None else depth_vis
                cv2.imshow(display_cfg.get("window_name", "Astra2"), combined)

            now = time.time()
            if now - last_print >= float(display_cfg.get("print_interval_s", 1.0)):
                h, w = depth_mm.shape
                center_distance_mm = int(depth_mm[h // 2, w // 2])
                print(f"center distance: {center_distance_mm} mm")
                last_print = now

            if live and display_cfg.get("show_window", True):
                key = cv2.waitKey(1)
                if key in (27, ord("q")):  # 27 is the ESC key
                    break
                if key == ord("s"):
                    capture_index += 1

                    depth_vis = depth_mm_to_vis(depth_mm, display_cfg)
                    saved = save_rgbd_capture(
                        depth_mm, color_bgr, depth_vis, capture_index, capture_cfg
                    )
                    if saved:
                        print(f"Saved capture: {', '.join(saved)}")
                    else:
                        print("Capture: no RGB-D data saved.")

                    saved_undist = save_rgbd_undistorted(
                        depth_mm, color_bgr, depth_vis_fn, capture_index, capture_cfg
                    )
                    if saved_undist:
                        print(f"Saved undistorted RGB-D: {', '.join(saved_undist)}")

                    point_format = OBFormat.POINT
                    point_cloud_filter.set_create_point_format(point_format)
                    pc_frame = point_cloud_filter.process(depth_frame)
                    if pc_frame is None:
                        print("Point cloud: failed to generate.")
                    else:
                        xyz, _ = decode_point_cloud_frame_to_numpy(pc_frame, point_format)
                        if xyz is not None:
                            path = save_pointcloud_capture(xyz, capture_index, capture_cfg)
                            print(f"Saved point cloud: {path}")
                        else:
                            print("Point cloud: empty.")

                    undist_path = save_pointcloud_undistorted(
                        depth_mm, capture_index, capture_cfg
                    )
                    if undist_path:
                        print(f"Saved undistorted point cloud: {undist_path}")

        return 0
    finally:
        try:
            pipeline.stop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
