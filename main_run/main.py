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
    summarize_pointcloud,
)
from main_run.captures import (
    save_pointcloud_capture,
    save_pointcloud_undistorted,
    save_rgbd_capture,
    save_rgbd_undistorted,
)
from main_run.profiles import choose_video_profile, list_video_profiles
from main_run.runtime_helpers import (
    depth_mm_to_vis,
    extract_color_frame,
    extract_depth_frame,
    print_device_info,
    stack_side_by_side,
    summarize_depth,
)
from main_run.visual import color_frame_to_bgr, depth_frame_to_mm


def main() -> int:
    """Program entry. Returns a process exit code."""
    if cfg.MODE not in {"depth_with_color", "pointcloud_only_depth"}:
        print(f"Unsupported MODE: {cfg.MODE}")
        return 2

    pipeline = Pipeline()
    live = not cfg.PRINT_INFO_OFFLINE

    if cfg.PRINT_INFO_OFFLINE:
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

    # Color is required only for depth preview mode.
    want_color = cfg.MODE == "depth_with_color"
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

    capture_index = 0
    try:
        depth_filters = build_depth_filters(pipeline)
        if depth_filters.description:
            print("\nDepth filters (applied to depth + point clouds):")
            for line in depth_filters.description:
                print(f"  - {line}")
        else:
            print("\nDepth filters: none")

        # Depth->color alignment is always on.
        align_filter = AlignFilter(OBStreamType.COLOR_STREAM) if have_color else None
        if want_color and not have_color:
            print("Color stream unavailable; cannot align depth to color.")

        point_cloud_filter = None
        if cfg.MODE == "pointcloud_only_depth":
            # PointCloudFilter needs camera params to correctly scale output points.
            point_cloud_filter = create_pointcloud_filter(pipeline)

        last_print = 0.0
        warmup_total = max(0, int(cfg.WARMUP_FRAMES))
        warmup_remaining = warmup_total
        warmup_seen = 0
        while True:
            frameset = pipeline.wait_for_frames(int(cfg.TIMEOUT_MS))
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

            if cfg.MODE == "depth_with_color":
                depth_mm = depth_frame_to_mm(depth_frame)
                if depth_mm is None:
                    continue

                # Clamp to a reasonable mm range (matches SDK example)
                depth_mm = np.where(
                    (depth_mm > cfg.MIN_DEPTH_MM) & (depth_mm < cfg.MAX_DEPTH_MM),
                    depth_mm,
                    0,
                ).astype(np.uint16)

                color_bgr = None
                depth_vis = None
                if live and cfg.SHOW_WINDOW:
                    # Render depth in a simple color map for quick visualization.
                    depth_vis = depth_mm_to_vis(depth_mm)

                    if have_color:
                        color_frame = extract_color_frame(aligned_frames)
                        if color_frame is None:
                            color_frame = extract_color_frame(frameset)
                        if color_frame is not None:
                            color_bgr = color_frame_to_bgr(color_frame)

                    combined = stack_side_by_side(depth_vis, color_bgr) if color_bgr is not None else depth_vis
                    cv2.imshow("Astra2", combined)

                now = time.time()
                if now - last_print >= float(cfg.PRINT_INTERVAL_S):
                    h, w = depth_mm.shape
                    center_distance_mm = int(depth_mm[h // 2, w // 2])
                    print(f"center distance: {center_distance_mm} mm")
                    last_print = now

                if live and cfg.SHOW_WINDOW:
                    key = cv2.waitKey(1)
                    if key in (27, ord("q")): #27 is the ESC key
                        break
                    if key == ord("s"):
                        capture_index += 1
                        saved = save_rgbd_capture(depth_mm, color_bgr, depth_vis, capture_index)
                        if saved:
                            print(f"Saved capture: {', '.join(saved)}")
                        saved_undist = save_rgbd_undistorted(depth_mm, color_bgr, depth_mm_to_vis, capture_index)
                        if saved_undist:
                            print(f"Saved undistorted: {', '.join(saved_undist)}")
                else:
                    # Single-shot summary then exit.
                    summarize_depth(depth_mm.astype(np.float32) / 1000.0)
                    return 0

            else:
                assert point_cloud_filter is not None

                point_format = OBFormat.POINT
                point_cloud_filter.set_create_point_format(point_format)

                pc_frame = point_cloud_filter.process(depth_frame)
                if pc_frame is None:
                    continue

                xyz, _ = decode_point_cloud_frame_to_numpy(pc_frame, point_format)
                if xyz is None:
                    continue

                now = time.time()
                if now - last_print >= float(cfg.PRINT_INTERVAL_S):
                    summarize_pointcloud(xyz)
                    last_print = now

                if live and cfg.SHOW_WINDOW:
                    depth_mm = depth_frame_to_mm(depth_frame)
                    if depth_mm is not None:
                        depth_mm = np.where(
                            (depth_mm > cfg.MIN_DEPTH_MM) & (depth_mm < cfg.MAX_DEPTH_MM),
                            depth_mm,
                            0,
                        ).astype(np.uint16)
                        depth_vis = depth_mm_to_vis(depth_mm)
                        cv2.imshow("Astra2", depth_vis)

                    key = cv2.waitKey(1)
                    if key in (27, ord("q")):
                        break
                    if key == ord("s"):
                        capture_index += 1
                        path = save_pointcloud_capture(xyz, capture_index)
                        print(f"Saved point cloud: {path}")
                        depth_mm = depth_frame_to_mm(depth_frame)
                        if depth_mm is not None:
                            undist_path = save_pointcloud_undistorted(depth_mm, capture_index)
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
