"""Shared camera session setup for live capture."""

from __future__ import annotations

import os

from pyorbbecsdk import AlignFilter, Config, OBError, OBFormat, OBSensorType, OBStreamType, Pipeline

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from orbbec.filters import apply_depth_filters, build_depth_filters
from orbbec.pointcloud import create_pointcloud_filter
from orbbec.profiles import choose_video_profile
from orbbec.visual import extract_depth_frame


def load_config(config_path: str | None = None) -> dict:
    """Load config.toml into a raw dict (no schema coercion)."""
    default_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
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


class CaptureSession:
    """Owns pipeline setup + common per-frame helpers for live or one-shot use."""
    def __init__(self, config_path: str | None = None, cfg: dict | None = None) -> None:
        self.cfg = cfg if cfg is not None else load_config(config_path)
        self.streams_cfg = self.cfg.get("streams", {})
        self.filters_cfg = self.cfg.get("filters", {})
        self.display_cfg = self.cfg.get("display", {})
        self.capture_cfg = self.cfg.get("capture", {})

        self.pipeline = Pipeline()
        self.config = Config()

        # Depth stream is required; index=-1 or width/height/fps=0 means "use default".
        depth_cfg = self.streams_cfg.get("depth", {})
        depth_profile_index = depth_cfg.get("profile_index")
        if depth_profile_index is not None and depth_profile_index < 0:
            depth_profile_index = None
        depth_width = depth_cfg.get("width", 0) or None
        depth_height = depth_cfg.get("height", 0) or None
        depth_fps = depth_cfg.get("fps", 0) or None
        depth_profile = choose_video_profile(
            self.pipeline,
            OBSensorType.DEPTH_SENSOR,
            index=depth_profile_index,
            width=depth_width,
            height=depth_height,
            fps=depth_fps,
            format_name=depth_cfg.get("format", "Y16"),
            default_format=OBFormat.Y16,
        )
        self.config.enable_stream(depth_profile)

        # Color stream is optional; if unavailable we still run depth-only.
        self.have_color = False
        try:
            color_cfg = self.streams_cfg.get("color", {})
            color_profile_index = color_cfg.get("profile_index")
            if color_profile_index is not None and color_profile_index < 0:
                color_profile_index = None
            color_width = color_cfg.get("width", 0) or None
            color_height = color_cfg.get("height", 0) or None
            color_fps = color_cfg.get("fps", 0) or None
            color_profile = choose_video_profile(
                self.pipeline,
                OBSensorType.COLOR_SENSOR,
                index=color_profile_index,
                width=color_width,
                height=color_height,
                fps=color_fps,
                format_name=color_cfg.get("format", "RGB"),
                default_format=OBFormat.RGB,
            )
            self.config.enable_stream(color_profile)
            self.have_color = True
        except OBError as exc:
            print(f"Color stream unavailable: {exc}")
        except Exception as exc:
            print(f"Failed to configure color stream: {exc}")

        if self.have_color:
            # Frame sync improves depth/color alignment stability.
            try:
                self.pipeline.enable_frame_sync()
            except Exception:
                pass

        self.pipeline.start(self.config)

        # Normalize decimation scale: 0 means SDK default.
        dec_cfg = self.filters_cfg.get("decimation", {})
        if dec_cfg.get("scale_value") == 0:
            dec_cfg["scale_value"] = None
        self.depth_filters = build_depth_filters(self.pipeline, self.filters_cfg)

        # Align depth to color if color is available.
        self.align_filter = AlignFilter(OBStreamType.COLOR_STREAM) if self.have_color else None
        self.point_cloud_filter = create_pointcloud_filter(self.pipeline)

    def warmup(self, *, verbose: bool = False) -> None:
        """Discard initial frames to let auto-exposure/filters stabilize."""
        warmup_total = max(0, int(self.display_cfg.get("warmup_frames", 0)))
        timeout_ms = int(self.display_cfg.get("timeout_ms", 100))
        for i in range(warmup_total):
            if verbose:
                print(f"Warmup frame {i + 1}/{warmup_total}")
            self.pipeline.wait_for_frames(timeout_ms)

    def next_frames(self):
        """Return (frameset, aligned_frames, filtered_depth_frame)."""
        timeout_ms = int(self.display_cfg.get("timeout_ms", 100))
        while True:
            frameset = self.pipeline.wait_for_frames(timeout_ms)
            if frameset is None:
                continue

            aligned_frames = frameset
            if self.align_filter is not None:
                aligned = self.align_filter.process(frameset)
                if aligned is not None:
                    aligned_frames = aligned

            depth_frame = extract_depth_frame(aligned_frames)
            if depth_frame is None:
                depth_frame = extract_depth_frame(frameset)
            if depth_frame is None:
                continue

            depth_frame = apply_depth_filters(depth_frame, self.depth_filters.filters)
            return frameset, aligned_frames, depth_frame

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
