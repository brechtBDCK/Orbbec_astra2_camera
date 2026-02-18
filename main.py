"""Astra2 main entry point.

This file focuses on control-flow only. Configuration lives in config.toml.
"""

import sys
import time

import cv2
import numpy as np
from pyorbbecsdk import OBSensorType, Pipeline

from orbbec.profiles import list_video_profiles
from orbbec.visual import (
    clamp_depth_mm,
    depth_mm_to_vis,
    extract_color_frame,
    stack_side_by_side,
)
from orbbec.session import CaptureSession, load_config
from orbbec.take_images import save_all_from_frames
from orbbec.visual import color_frame_to_bgr, depth_frame_to_mm

def print_device_info(pipeline: Pipeline) -> None:
    try:
        info = pipeline.get_device().get_device_info()
        print("\nDevice:")
        print(f"  name: {info.get_name()}")
        print(f"  pid:  {info.get_pid()}  vid: {info.get_vid()}")
        print(f"  sn:   {info.get_serial_number()}")
        print(f"  fw:   {info.get_firmware_version()}")
    except Exception as exc:
        print(f"Device info unavailable: {exc}")

def main() -> int:
    """Program entry. Returns a process exit code."""
    cfg = load_config()
    run_cfg = cfg.get("run", {})
    display_cfg = cfg.get("display", {})

    if run_cfg.get("print_info_offline", False):
        # Query device/profiles once and exit (no live streaming).
        pipeline = Pipeline()
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

    live = True
    session = None
    try:
        session = CaptureSession(cfg=cfg)
        if session.depth_filters.description:
            print("\nDepth filters (applied to depth + point clouds):")
            for line in session.depth_filters.description:
                print(f"  - {line}")
        else:
            print("\nDepth filters: none")

        if not session.have_color:
            print("Color stream unavailable; running depth-only preview.")

        if display_cfg.get("show_window", True):
            window_name = display_cfg.get("window_name", "Astra2")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(
                window_name,
                int(display_cfg.get("window_pos_x", 50)),
                int(display_cfg.get("window_pos_y", 50)),
            )

        last_print = 0.0
        session.warmup(verbose=True)

        while True:
            frameset, aligned_frames, depth_frame = session.next_frames()

            # Convert filtered depth to mm (uint16) for display and saving.
            depth_mm = depth_frame_to_mm(depth_frame)
            if depth_mm is None:
                continue

            depth_mm = clamp_depth_mm(depth_mm, display_cfg)

            color_bgr = None
            if session.have_color:
                color_frame = extract_color_frame(aligned_frames)
                if color_frame is None:
                    color_frame = extract_color_frame(frameset)
                if color_frame is not None:
                    color_bgr = color_frame_to_bgr(color_frame)

            if live and display_cfg.get("show_window", True):
                # Visual preview: depth colormap (and color, if available).
                depth_vis = depth_mm_to_vis(depth_mm, display_cfg)
                combined = (
                    stack_side_by_side(depth_vis, color_bgr)
                    if color_bgr is not None
                    else depth_vis
                )
                cv2.imshow(display_cfg.get("window_name", "Astra2"), combined)

            now = time.time()
            if now - last_print >= float(display_cfg.get("print_interval_s", 1.0)):
                # Lightweight feedback: center pixel depth.
                h, w = depth_mm.shape
                center_distance_mm = int(depth_mm[h // 2, w // 2])
                print(f"center distance: {center_distance_mm} mm")
                last_print = now

            if live and display_cfg.get("show_window", True):
                key = cv2.waitKey(1)
                if key in (27, ord("q")):  # 27 is the ESC key
                    break
                if key == ord("s"):
                    # One-shot capture: RGB-D + point cloud + undistorted versions.
                    result = save_all_from_frames(
                        depth_frame=depth_frame,
                        aligned_frames=aligned_frames,
                        frameset=frameset,
                        have_color=session.have_color,
                        point_cloud_filter=session.point_cloud_filter,
                        display_cfg=session.display_cfg,
                        capture_cfg=session.capture_cfg,
                        capture_index=None,
                    )
                    if result is None:
                        print("Capture: failed.")
                    else:
                        if result["rgbd"]:
                            print(f"Saved capture: {', '.join(result['rgbd'])}")
                        else:
                            print("Capture: no RGB-D data saved.")
                        if result["rgbd_undist"]:
                            print(
                                f"Saved undistorted RGB-D: {', '.join(result['rgbd_undist'])}"
                            )
                        if result["pointcloud"]:
                            print(f"Saved point cloud: {result['pointcloud']}")
                        if result["pointcloud_undist"]:
                            print(
                                f"Saved undistorted point cloud: {result['pointcloud_undist']}"
                            )

        return 0
    finally:
        if session is not None:
            session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
