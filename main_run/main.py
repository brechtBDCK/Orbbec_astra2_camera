"""Astra2 main entry point.

This file focuses on control-flow only. Configuration lives in config.py.

Flow overview:
- Build and start the pipeline
- Apply optional alignment and depth filters
- Either visualize depth frames or generate point clouds
"""

import os
import sys
import time

if "QT_QPA_FONTDIR" not in os.environ:
    for _candidate in ("/usr/share/fonts/truetype", "/usr/share/fonts", "/usr/local/share/fonts"):
        if os.path.isdir(_candidate):
            os.environ["QT_QPA_FONTDIR"] = _candidate
            break

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
from main_run.visual import color_frame_to_bgr, depth_frame_to_mm


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


def extract_depth_frame(obj):
    """Return a depth frame from either a frameset or a single frame."""
    if obj is None:
        return None
    if hasattr(obj, "get_depth_frame"):
        try:
            return obj.get_depth_frame()
        except Exception:
            pass
    if hasattr(obj, "as_depth_frame"):
        try:
            return obj.as_depth_frame()
        except Exception:
            pass
    return None


def extract_color_frame(obj):
    """Return a color frame from either a frameset or a single frame."""
    if obj is None:
        return None
    if hasattr(obj, "get_color_frame"):
        try:
            return obj.get_color_frame()
        except Exception:
            pass
    if hasattr(obj, "as_color_frame"):
        try:
            return obj.as_color_frame()
        except Exception:
            pass
    return None


def stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Resize to a common height and stack horizontally."""
    if left is None:
        return right
    if right is None:
        return left
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    if lh != rh:
        target_h = max(lh, rh)
        if lh != target_h:
            new_w = max(1, int(lw * (target_h / float(lh))))
            left = cv2.resize(left, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if rh != target_h:
            new_w = max(1, int(rw * (target_h / float(rh))))
            right = cv2.resize(right, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([left, right])


def depth_mm_to_vis(depth_mm: np.ndarray) -> np.ndarray:
    """Convert a depth-mm image to a colored visualization."""
    valid = depth_mm[depth_mm > 0]
    if cfg.AUTO_SCALE and valid.size:
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            vmax = vmin + 1e-3
        depth_vis = np.clip(depth_mm, vmin, vmax)
        depth_8u = ((depth_vis - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        depth_vis = np.clip(depth_mm, 0, cfg.MAX_DEPTH_MM)
        depth_8u = (depth_vis / cfg.MAX_DEPTH_MM * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def main() -> int:
    """Program entry. Returns a process exit code."""
    if cfg.MODE not in {"depth_with_color", "pointcloud_with_color"}:
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
        if cfg.MODE == "pointcloud_with_color":
            # PointCloudFilter needs camera params to correctly scale output points.
            point_cloud_filter = create_pointcloud_filter(pipeline)
            if cfg.POINTCLOUD_SAVE_PLY:
                ensure_pointcloud_dir()

        last_print = 0.0
        save_index = 0
        warned_pc_size_mismatch = False
        pc_average_target = max(1, int(cfg.POINTCLOUD_AVERAGE_N_FRAMES))
        pc_xyz_sum: np.ndarray | None = None
        pc_xyz_count = 0
        pc_frame_times_ms: list[float] = []
        pc_batch_start: float | None = None
        warmup_remaining = max(0, int(cfg.WARMUP_FRAMES))
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

                if live and cfg.SHOW_WINDOW:
                    # Render depth in a simple color map for quick visualization.
                    depth_vis = depth_mm_to_vis(depth_mm)

                    color_bgr = None
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
                else:
                    # Single-shot summary then exit.
                    summarize_depth(depth_mm.astype(np.float32) / 1000.0)
                    return 0

            else:
                assert point_cloud_filter is not None

                point_format = OBFormat.POINT
                point_cloud_filter.set_create_point_format(point_format)

                if pc_batch_start is None:
                    pc_batch_start = time.perf_counter()
                frame_start = time.perf_counter()
                pc_frame = point_cloud_filter.process(depth_frame)
                if pc_frame is None:
                    if pc_xyz_count == 0:
                        pc_batch_start = None
                    continue

                xyz, _ = decode_point_cloud_frame_to_numpy(pc_frame, point_format)
                if xyz is None:
                    if pc_xyz_count == 0:
                        pc_batch_start = None
                    continue

                pc_frame_times_ms.append((time.perf_counter() - frame_start) * 1000.0)

                if pc_xyz_sum is not None and xyz.shape != pc_xyz_sum.shape:
                    if not warned_pc_size_mismatch:
                        print("Point cloud size changed; resetting average buffer.")
                        warned_pc_size_mismatch = True
                    pc_xyz_sum = None
                    pc_xyz_count = 0
                    pc_frame_times_ms.clear()
                    pc_batch_start = None

                if pc_xyz_sum is None:
                    pc_xyz_sum = xyz.astype(np.float32, copy=True)
                    pc_xyz_count = 1
                else:
                    pc_xyz_sum += xyz
                    pc_xyz_count += 1

                if pc_xyz_count < pc_average_target:
                    continue

                xyz_avg = pc_xyz_sum / float(pc_xyz_count)
                pc_xyz_sum = None
                pc_xyz_count = 0
                batch_ms = None
                if pc_batch_start is not None:
                    batch_ms = (time.perf_counter() - pc_batch_start) * 1000.0
                pc_batch_start = None

                now = time.time()
                if now - last_print >= float(cfg.PRINT_INTERVAL_S):
                    summarize_pointcloud(xyz_avg)
                    if pc_frame_times_ms:
                        avg_frame_ms = sum(pc_frame_times_ms) / float(len(pc_frame_times_ms))
                        if batch_ms is None:
                            print(f"Point cloud timing: frame avg {avg_frame_ms:.2f} ms")
                        else:
                            print(
                                "Point cloud timing: "
                                f"frame avg {avg_frame_ms:.2f} ms, "
                                f"batch {batch_ms:.2f} ms (N={len(pc_frame_times_ms)})"
                            )
                    pc_frame_times_ms.clear()
                    last_print = now

                if cfg.POINTCLOUD_SAVE_PLY:
                    out_path = pointcloud_output_path(save_index)
                    save_point_cloud_to_ply(out_path, xyz_avg, None)
                    print(f"Saved point cloud: {out_path}")
                    save_index += 1

        return 0
    finally:
        try:
            pipeline.stop()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
