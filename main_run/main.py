"""Astra2 main entry point.

This file focuses on control-flow only. Configuration lives in config.py.

Flow overview:
- Build and start the pipeline
- Apply optional alignment and depth filters
- Either visualize depth frames or generate point clouds
"""

import sys
import time

import cv2
import numpy as np

from pyorbbecsdk import AlignFilter, Config, OBError, OBFormat, OBSensorType, OBStreamType, Pipeline

import main_run.config as cfg
from main_run.filters import apply_depth_filters, build_depth_filters
from main_run.pointcloud import (
    create_pointcloud_filter,
    decode_point_cloud_frame_to_numpy,
    ensure_pointcloud_dir,
    pointcloud_output_path,
    save_point_cloud_to_ply,
    summarize_pointcloud,
)
from main_run.profiles import choose_video_profile, list_video_profiles
from main_run.visual import color_frame_to_bgr, depth_frame_to_meters, depth_to_colormap


ESC_KEY = 27


def print_device_info(pipeline: Pipeline) -> None:
    """Print a quick summary of the connected device."""
    try:
        info = pipeline.get_device().get_device_info()
    except Exception as exc:
        print(f"Device info unavailable: {exc}")
        return
    print("\nDevice:")
    print(f"  name: {info.get_name()}")
    print(f"  pid:  {info.get_pid()}  vid: {info.get_vid()}")
    print(f"  sn:   {info.get_serial_number()}")
    print(f"  fw:   {info.get_firmware_version()}")


def summarize_depth(depth_m):
    """Print simple statistics for a depth image in meters."""
    h, w = depth_m.shape
    center_m = float(depth_m[h // 2, w // 2])
    finite = depth_m[np.isfinite(depth_m)]
    d_min = float(np.min(finite)) if finite.size else float("nan")
    d_med = float(np.median(finite)) if finite.size else float("nan")
    d_max = float(np.max(finite)) if finite.size else float("nan")

    print("\nDepth summary:")
    print(f"  size:   {w}x{h}")
    print(f"  center: {center_m:.3f} m")
    print(f"  min:    {d_min:.3f} m")
    print(f"  median: {d_med:.3f} m")
    print(f"  max:    {d_max:.3f} m")


def main() -> int:
    """Program entry. Returns a process exit code."""
    if cfg.MODE not in {"depth", "pointcloud"}:
        print(f"Unsupported MODE: {cfg.MODE}")
        return 2

    pipeline = Pipeline()

    if cfg.LIST_PROFILES:
        # Handy for finding profile indexes you want to select in config.py
        list_video_profiles(pipeline, OBSensorType.DEPTH_SENSOR, "Depth")
        list_video_profiles(pipeline, OBSensorType.COLOR_SENSOR, "Color")
        list_video_profiles(pipeline, OBSensorType.LEFT_IR_SENSOR, "Left IR")
        list_video_profiles(pipeline, OBSensorType.RIGHT_IR_SENSOR, "Right IR")
        return 0

    config = Config()

    # Depth is required.
    try:
        # If DEPTH_PROFILE_INDEX is set, it already defines width/height/fps.
        # Only set width/height/fps when you want to request a specific profile.
        depth_profile = choose_video_profile(
            pipeline,
            OBSensorType.DEPTH_SENSOR,
            index=cfg.DEPTH_PROFILE_INDEX,
            width=cfg.DEPTH_WIDTH,
            height=cfg.DEPTH_HEIGHT,
            fps=cfg.DEPTH_FPS,
            format_name=cfg.DEPTH_FORMAT_NAME,
            default_format=OBFormat.Y16,
        )
        config.enable_stream(depth_profile)
        print(f"Depth profile: {depth_profile}")
    except Exception as exc:
        print(f"Failed to configure depth stream: {exc}")
        return 1

    # Color is optional, but required for alignment and RGB point clouds.
    want_color = bool(
        cfg.ENABLE_COLOR or cfg.ALIGN_DEPTH_TO_COLOR or (cfg.MODE == "pointcloud" and cfg.POINTCLOUD_WITH_COLOR)
    )
    have_color = False
    if want_color:
        try:
            color_profile = choose_video_profile(
                pipeline,
                OBSensorType.COLOR_SENSOR,
                index=cfg.COLOR_PROFILE_INDEX,
                width=cfg.COLOR_WIDTH,
                height=cfg.COLOR_HEIGHT,
                fps=cfg.COLOR_FPS,
                format_name=cfg.COLOR_FORMAT_NAME,
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

    try:
        if cfg.PRINT_DEVICE_INFO:
            print_device_info(pipeline)

        if cfg.PRINT_AVAILABLE_PROFILES:
            # Lists all profiles the device reports for each sensor.
            list_video_profiles(pipeline, OBSensorType.DEPTH_SENSOR, "Depth")
            list_video_profiles(pipeline, OBSensorType.COLOR_SENSOR, "Color")
            list_video_profiles(pipeline, OBSensorType.LEFT_IR_SENSOR, "Left IR")
            list_video_profiles(pipeline, OBSensorType.RIGHT_IR_SENSOR, "Right IR")

        depth_filters = build_depth_filters(pipeline)
        if depth_filters.description:
            print("\nDepth filters:")
            for line in depth_filters.description:
                print(f"  - {line}")
        else:
            print("\nDepth filters: none")

        if cfg.PRINT_AVAILABLE_FILTERS:
            # Print the device's recommended post-processing filters.
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

        # Depth->color alignment (recommended for RGB point clouds)
        want_align = bool(
            cfg.ALIGN_DEPTH_TO_COLOR or (cfg.MODE == "pointcloud" and cfg.POINTCLOUD_WITH_COLOR and have_color)
        )
        align_filter = AlignFilter(OBStreamType.COLOR_STREAM) if (want_align and have_color) else None
        if want_align and not have_color:
            print("Alignment requested but color stream is not available.")
        elif align_filter is not None and cfg.MODE == "pointcloud" and cfg.POINTCLOUD_WITH_COLOR:
            print("Using depth->color alignment for RGB point clouds.")

        point_cloud_filter = None
        warned_rgb_with_filters = False
        if cfg.MODE == "pointcloud":
            # PointCloudFilter needs camera params to correctly scale output points.
            point_cloud_filter = create_pointcloud_filter(pipeline)
            if cfg.POINTCLOUD_SAVE_PLY:
                ensure_pointcloud_dir()

        last_print = 0.0
        saved_once = False
        frame_index = 0
        while True:
            frames = pipeline.wait_for_frames(int(cfg.TIMEOUT_MS))
            if frames is None:
                continue

            if align_filter is not None:
                aligned = align_filter.process(frames)
                if aligned is not None:
                    frames = aligned

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            depth_frame = apply_depth_filters(depth_frame, depth_filters.filters)

            if cfg.MODE == "depth":
                depth_m = depth_frame_to_meters(depth_frame)
                if depth_m is None:
                    continue

                if cfg.LIVE:
                    # Render depth in a simple color map for quick visualization.
                    depth_vis = depth_to_colormap(depth_m, float(cfg.MAX_VIS_M))
                    cv2.imshow("Astra2 Depth", depth_vis)

                    if have_color:
                        color_frame = frames.get_color_frame()
                        if color_frame is not None:
                            color_bgr = color_frame_to_bgr(color_frame)
                            if color_bgr is not None:
                                cv2.imshow("Astra2 Color", color_bgr)

                now = time.time()
                if now - last_print >= float(cfg.PRINT_INTERVAL_S):
                    h, w = depth_m.shape
                    center_distance_m = float(depth_m[h // 2, w // 2])
                    print(f"center distance: {center_distance_m:.3f} mm")
                    last_print = now

                if cfg.LIVE:
                    key = cv2.waitKey(1)
                    if key in (ESC_KEY, ord("q")):
                        break
                else:
                    # Single-shot summary then exit.
                    summarize_depth(depth_m)
                    return 0

            else:
                assert point_cloud_filter is not None

                want_rgb_pc = bool(cfg.POINTCLOUD_WITH_COLOR and have_color and not depth_filters.filters)
                if cfg.POINTCLOUD_WITH_COLOR and have_color and depth_filters.filters and not warned_rgb_with_filters:
                    # RGB point clouds require the full frameset. If depth is filtered,
                    # we fall back to depth-only point clouds for consistency.
                    print("Depth filters are enabled; generating depth-only point clouds to honor filters.")
                    warned_rgb_with_filters = True

                point_format = OBFormat.RGB_POINT if want_rgb_pc else OBFormat.POINT
                point_cloud_filter.set_create_point_format(point_format)

                pc_input = frames if want_rgb_pc else depth_frame
                pc_frame = point_cloud_filter.process(pc_input)
                if pc_frame is None:
                    continue

                xyz, rgb = decode_point_cloud_frame_to_numpy(pc_frame, point_format)
                if xyz is None:
                    continue

                now = time.time()
                if now - last_print >= float(cfg.PRINT_INTERVAL_S):
                    summarize_pointcloud(xyz)
                    last_print = now

                if cfg.POINTCLOUD_SAVE_PLY:
                    should_save = False
                    if cfg.POINTCLOUD_SAVE_EVERY_N_FRAMES <= 0:
                        should_save = not saved_once
                    else:
                        should_save = (frame_index % int(cfg.POINTCLOUD_SAVE_EVERY_N_FRAMES) == 0)

                    if should_save:
                        out_path = pointcloud_output_path(frame_index)
                        save_point_cloud_to_ply(out_path, xyz, rgb)
                        print(f"Saved point cloud: {out_path}")
                        saved_once = True
                        if not cfg.LIVE and cfg.POINTCLOUD_SAVE_EVERY_N_FRAMES <= 0:
                            return 0

                if not cfg.LIVE and not cfg.POINTCLOUD_SAVE_PLY:
                    return 0

                if cfg.LIVE:
                    key = cv2.waitKey(1)
                    if key in (ESC_KEY, ord("q")):
                        break

            frame_index += 1

        return 0
    finally:
        try:
            pipeline.stop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
